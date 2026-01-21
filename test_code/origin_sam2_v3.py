import os
import glob
import json
import shutil
import cv2
import torch
import numpy as np
import gc
import time
from tqdm import tqdm
from sam3.model_builder import build_sam3_video_model

# =========================================================
# 🛠️ 유틸리티 함수 (RLE 변환 & RAM 복사)
# =========================================================
def mask_to_rle(mask):
    """이진 마스크를 RLE(Run-Length Encoding) 리스트로 변환"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()

def rle_to_mask(rle, shape):
    """RLE 리스트를 이진 마스크로 복원"""
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
    """마스크에서 Bounding Box(xyxy) 추출"""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0: return None
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return [x_min, y_min, x_max, y_max]

def copy_frames_to_ram(source_dir, ram_dir_name="sam3_temp_frames"):
    """
    NAS(디스크)에 있는 이미지들을 리눅스 RAM 디스크(/dev/shm)로 고속 복사합니다.
    I/O 병목 현상을 해결하는 핵심 함수입니다.
    """
    ram_root = "/dev/shm"
    if not os.path.exists(ram_root):
        print("⚠️ Warning: /dev/shm이 없습니다. /tmp를 사용합니다.")
        ram_root = "/tmp"
        
    dest_dir = os.path.join(ram_root, ram_dir_name)

    # 이미지 리스트 확보
    print(f"🔍 원본 경로 검색 중: {source_dir}")
    files = sorted(glob.glob(os.path.join(source_dir, "*.jpg")))
    if not files:
        files = sorted(glob.glob(os.path.join(source_dir, "*.png")))
    
    if not files:
        raise FileNotFoundError(f"❌ 해당 경로에 이미지가 없습니다: {source_dir}")
        
    total_files = len(files)
    print(f"📦 총 {total_files}개의 프레임을 RAM으로 복사합니다...")

    # 기존 데이터 정리 후 생성
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    # 고속 복사 진행
    start_time = time.time()
    for src_path in tqdm(files, desc="🚀 RAM Copying"):
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copyfile(src_path, dest_path)

    duration = time.time() - start_time
    print(f"✅ 복사 완료! ({duration:.2f}초 소요)")
    print(f"📂 RAM 데이터 경로: {dest_dir}")
    
    return dest_dir

def cleanup_ram(ram_path):
    """사용이 끝난 RAM 데이터를 삭제"""
    if os.path.exists(ram_path) and ("/dev/shm" in ram_path or "/tmp" in ram_path):
        print(f"🗑️ RAM 메모리 반환 중: {ram_path}")
        shutil.rmtree(ram_path)
        print("✨ 메모리 정리 완료.")

# =========================================================
# 🚀 메인 트래킹 로직
# =========================================================
def run_sam3_tracking(video_path, json_path, output_video_path, output_json_dir, fps=60):
    
    # 결과 JSON 저장 디렉토리 생성
    os.makedirs(output_json_dir, exist_ok=True)
    
    # 1. 장치 및 모델 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Using device: {device}")
    
    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    print("🔄 Loading SAM 3 Model...")
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    
    # 2. 이미지 정보 확인
    frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    if not frame_files:
        frame_files = sorted(glob.glob(os.path.join(video_path, "*.png")))
    
    total_frames = len(frame_files)
    temp_img = cv2.imread(frame_files[0])
    height, width = temp_img.shape[:2]
    print(f"🎬 Tracking Target: {total_frames} frames ({width}x{height})")

    # 3. Inference State 초기화 (RAM 경로 + 비동기 로딩)
    print("🔄 Initializing Inference State...")
    inference_state = predictor.init_state(
        video_path=video_path,       # RAM 경로가 들어감
        offload_video_to_cpu=True,   # VRAM 절약
        offload_state_to_cpu=True,   # 메모리 절약
        async_loading_frames=True    # 비동기 로딩 (속도 향상)
    )
    predictor.clear_all_points_in_video(inference_state)

    # 4. JSON 프롬프트 로드 (0번 프레임)
    print(f"📂 Loading Input Prompts from {json_path}...")
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
                    # 좌표 정규화
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

    # 5. 트래킹 시작 및 결과 저장
    print("🚀 Running Tracking & Saving Results...")
    
    # Video Writer 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 색상 맵 생성
    np.random.seed(42)
    color_map = {i: np.random.randint(100, 255, (3,), dtype=np.uint8).tolist() for i in range(100)}

    # 전파(Propagation) 실행
    propagation_gen = predictor.propagate_in_video(
        inference_state, start_frame_idx=0, max_frame_num_to_track=None,
        reverse=False, propagate_preflight=True
    )

    for frame_idx, obj_ids, _, video_res_masks, _ in propagation_gen:
        # 프레임 이미지 읽기 (RAM에서)
        if frame_idx < len(frame_files):
            frame = cv2.imread(frame_files[frame_idx])
        else:
            break
        
        if frame is None: break

        overlay = frame.copy()
        frame_dict = {} # JSON 저장용
        
        for i, out_obj_id in enumerate(obj_ids):
            # 마스크 이진화
            mask_bool = (video_res_masks[i] > 0.0).cpu().numpy().squeeze()
            
            if mask_bool.sum() > 0:
                # 1) JSON 데이터 저장 (RLE)
                frame_dict[int(out_obj_id)] = {"rle": mask_to_rle(mask_bool)}
                
                # 2) 영상 시각화 (Overlay)
                color = color_map.get(out_obj_id, [0, 255, 0])
                overlay[mask_bool] = color
                contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 2)
                
                # ID 텍스트
                y, x = np.where(mask_bool)
                if len(y) > 0:
                    cy, cx = int(np.mean(y)), int(np.mean(x))
                    cv2.putText(frame, f"ID {out_obj_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 프레임별 JSON 파일 저장
        if frame_dict:
            save_path = os.path.join(output_json_dir, f"{frame_idx:05d}.json")
            with open(save_path, 'w') as f:
                json.dump(frame_dict, f)

        # 영상 합성 및 저장
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        out.write(frame)
        
    out.release()
    
    # 메모리 정리
    del inference_state
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"✅ All Jobs Completed!")
    print(f"   🎬 Video: {output_video_path}")
    print(f"   📂 JSONs: {output_json_dir}")

# =========================================================
# 실행 블록
# =========================================================
if __name__ == "__main__":
    
    # 1. 원본 데이터 경로 (NAS)
    NAS_VIDEO_DIR = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/walking_data/FRAME/frontal__walking__1"
    
    # 2. 입력 및 출력 설정
    INPUT_JSON_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/output_per_frame_json/00000.json"
    OUTPUT_VIDEO_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/final_result_v3.mp4"
    OUTPUT_JSON_DIR = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/tracking_results_json"
    
    ram_path = None
    try:
        # ⭐ 1단계: NAS -> RAM 고속 복사
        ram_path = copy_frames_to_ram(NAS_VIDEO_DIR, ram_dir_name="sam3_frames_cache")
        
        # ⭐ 2단계: 트래킹 실행 (RAM 경로 사용)
        run_sam3_tracking(
            video_path=ram_path,        # 복사된 RAM 경로 전달
            json_path=INPUT_JSON_PATH,
            output_video_path=OUTPUT_VIDEO_PATH,
            output_json_dir=OUTPUT_JSON_DIR,
            fps=60
        )
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # ⭐ 3단계: RAM 메모리 반환 (필수)
        if ram_path:
            cleanup_ram(ram_path)