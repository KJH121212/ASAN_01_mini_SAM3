import os
import glob
import json
import cv2
import torch
import numpy as np
import gc
from sam3.model_builder import build_sam3_video_model

# =========================================================
# 🛠️ RLE 유틸리티 함수
# =========================================================
def mask_to_rle(mask):
    """이진 마스크를 RLE 형식 리스트로 변환합니다."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()

def rle_to_mask(rle, shape):
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if not rle: return mask.reshape((h, w))
    rle = np.array(rle)
    starts = rle[0::2] - 1
    lengths = rle[1::2]
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape((h, w))

def get_box_from_mask(mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0: return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return [x_min, y_min, x_max, y_max]

# =========================================================
# 🚀 메인 트래킹 함수
# =========================================================
def run_sam3_tracking(video_path, json_path, output_video_path, output_json_dir, fps=60):
    
    # 결과 JSON 저장 디렉토리 생성
    os.makedirs(output_json_dir, exist_ok=True)
    
    # 1. 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Using device: {device}")
    
    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True # SAM3 최적화

    print("🔄 Loading SAM 3 Model...")
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    
    # 2. 비디오/폴더 확인
    is_directory = os.path.isdir(video_path)
    if is_directory:
        frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        if not frame_files:
            frame_files = sorted(glob.glob(os.path.join(video_path, "*.png")))
        if not frame_files:
            print(f"❌ Error: 폴더에 이미지가 없습니다: {video_path}")
            return
        total_frames = len(frame_files)
        temp_img = cv2.imread(frame_files[0])
        height, width = temp_img.shape[:2]
        print(f"📂 Folder Detected: {total_frames} frames.")
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"🎬 Video File Detected: {total_frames} frames.")

    # 3. Inference State 초기화
    print("🔄 Initializing Inference State (CPU Offload)...")
    inference_state = predictor.init_state(
        video_path=video_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=True
    )
    predictor.clear_all_points_in_video(inference_state)

    # 4. JSON 로드 및 프롬프트 입력
    print(f"📂 Loading Input JSON from {json_path}...")
    with open(json_path, 'r') as f:
        frame_0_data = json.load(f)

    print("✨ Adding prompts to Frame 0...")
    for obj_id_str, info in frame_0_data.items():
        try:
            obj_id = int(obj_id_str)
            if "rle" in info:
                mask_np = rle_to_mask(info['rle'], (height, width))
                box_xyxy = get_box_from_mask(mask_np)
                
                if box_xyxy is not None:
                    x_min, y_min, x_max, y_max = box_xyxy
                    rel_box = np.array([
                        [x_min / width, y_min / height, x_max / width, y_max / height]
                    ], dtype=np.float32)

                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=rel_box
                    )
                    print(f"   - Object {obj_id}: Added Box {box_xyxy}")
        except ValueError:
            continue

    # 5. 트래킹 및 저장
    print("🚀 Running Tracking & Saving Results...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not is_directory:
        cap = cv2.VideoCapture(video_path)

    np.random.seed(42)
    color_map = {i: np.random.randint(100, 255, (3,), dtype=np.uint8).tolist() for i in range(100)}

    propagation_gen = predictor.propagate_in_video(
        inference_state, start_frame_idx=0, max_frame_num_to_track=None,
        reverse=False, propagate_preflight=True
    )

    for frame_idx, obj_ids, _, video_res_masks, _ in propagation_gen:
        # 프레임 이미지 읽기
        if is_directory:
            frame = cv2.imread(frame_files[frame_idx]) if frame_idx < len(frame_files) else None
        else:
            ret, frame = cap.read()
            if not ret: frame = None
        
        if frame is None: break

        overlay = frame.copy()
        frame_dict = {} # JSON 저장을 위한 딕셔너리
        
        for i, out_obj_id in enumerate(obj_ids):
            # 마스크 처리 (이진화)
            mask_bool = (video_res_masks[i] > 0.0).cpu().numpy().squeeze()
            
            if mask_bool.sum() > 0:
                # ⭐ JSON 저장용 RLE 변환
                frame_dict[int(out_obj_id)] = {"rle": mask_to_rle(mask_bool)}
                
                # 영상 시각화 로직
                color = color_map.get(out_obj_id, [0, 255, 0])
                overlay[mask_bool] = color
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 2)
                
                y, x = np.where(mask_bool)
                if len(y) > 0:
                    cy, cx = int(np.mean(y)), int(np.mean(x))
                    cv2.putText(frame, f"ID {out_obj_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # ⭐ 프레임별 JSON 파일 저장 (00000.json, 00001.json...)
        if frame_dict:
            save_path = os.path.join(output_json_dir, f"{frame_idx:05d}.json")
            with open(save_path, 'w') as f:
                json.dump(frame_dict, f)

        # 영상 합성 및 쓰기
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"   Processing {frame_idx}/{total_frames}...")

    if not is_directory: cap.release()
    out.release()
    
    del inference_state
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✅ Completed!")
    print(f"🎬 Video: {output_video_path}")
    print(f"📂 JSONs: {output_json_dir}")

# =========================================================
# 실행
# =========================================================
if __name__ == "__main__":
    MY_VIDEO_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/1_FRAME/Won_Kim_research_at_Bosanjin/M01/M01_VISIT2_LowerLimb"
    MY_JSON_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/2_KEYPOINTS/Won_Kim_research_at_Bosanjin/M01/M01_VISIT2_LowerLimb/000000.json"
    
    # 출력 경로 설정
    MY_OUTPUT_VIDEO = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/final_result.mp4"
    MY_OUTPUT_JSON_DIR = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/tracking_results_json"

    run_sam3_tracking(
        video_path=MY_VIDEO_PATH,
        json_path=MY_JSON_PATH,
        output_video_path=MY_OUTPUT_VIDEO,
        output_json_dir=MY_OUTPUT_JSON_DIR, # 추가된 매개변수
        fps=60
    )