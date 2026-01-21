import os
import glob
import json
import torch
import numpy as np
import gc
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm
from sam3.model_builder import build_sam3_video_predictor

# ==============================================================================
# ⚙️ 설정
# ==============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 원본 데이터 경로
ORIGIN_FRAME_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/walking_data/FRAME/frontal__walking__1")

# 결과 저장 경로
OUTPUT_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/output_per_frame_json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 가상 폴더 경로
TEMP_DIR = OUTPUT_DIR / "temp_chunk_frames"

STEP_SIZE = 200   # 보폭 (한 번에 처리할 양)
LOOK_AHEAD = 10   # 겹침 구간

# ==============================================================================
# 🛠️ 유틸리티
# ==============================================================================
def mask_to_rle(mask_bool_np):
    pixels = mask_bool_np.flatten()
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

def create_virtual_chunk_folder(start_idx, end_idx, all_files):
    if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR) 
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    chunk_files = all_files[start_idx : end_idx]
    for f in chunk_files:
        os.symlink(f, TEMP_DIR / Path(f).name)
    return len(chunk_files)

def get_box_from_mask(mask_np):
    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) == 0: return None 
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    # Numpy array [x1, y1, x2, y2]
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

# ==============================================================================
# 🚀 실행 로직
# ==============================================================================
all_frame_files = sorted([str(p) for p in ORIGIN_FRAME_PATH.glob("*.jpg")])
total_frames = len(all_frame_files)
gpus = range(torch.cuda.device_count())

_img = cv2.imread(all_frame_files[0])
IMG_HEIGHT, IMG_WIDTH = _img.shape[:2]

print("🔄 모델 로딩 중...")
predictor = build_sam3_video_predictor(gpus_to_use=gpus)

print(f"🎬 전체 {total_frames} 프레임 | 보폭 {STEP_SIZE} | 저장 후 읽기 방식")

for chunk_start in range(0, total_frames, STEP_SIZE):
    
    run_end = min(chunk_start + STEP_SIZE + LOOK_AHEAD, total_frames)
    num_frames_in_chunk = create_virtual_chunk_folder(chunk_start, run_end, all_frame_files)
    
    print(f"\n📁 [Virtual Chunk] {chunk_start}~{run_end} ({num_frames_in_chunk}장) 처리 중...")

    # 1. 세션 초기화
    response = predictor.handle_request(request=dict(type="start_session", resource_path=str(TEMP_DIR)))
    session_id = response["session_id"]
    
    with torch.inference_mode():
        # [Case A] 맨 처음 시작
        if chunk_start == 0:
            print("   ✨ (Start) 텍스트 프롬프트 'person' 적용")
            predictor.handle_request(
                request=dict(type="add_prompt", session_id=session_id, frame_index=0, text="person")
            )
            
        # [Case B] 이어달리기 (JSON 로드)
        else:
            prev_json_path = OUTPUT_DIR / f"{chunk_start:05d}.json"
            
            if prev_json_path.exists():
                print(f"   📂 (Load) {prev_json_path.name} 파일 로드")
                with open(prev_json_path, 'r') as f:
                    prev_data = json.load(f)
                
                if not prev_data:
                    print("   ⚠️ 빈 JSON: 텍스트로 재시작")
                    predictor.handle_request(dict(type="add_prompt", session_id=session_id, frame_index=0, text="person"))
                
                for obj_id_str, info in prev_data.items():
                    obj_id = int(obj_id_str)
                    rle = info['rle']
                    
                    # 마스크 복원
                    mask_np = rle_to_mask(rle, (IMG_HEIGHT, IMG_WIDTH))
                    box_xyxy = get_box_from_mask(mask_np) # 박스도 계산 (옵션)
                    
                    if box_xyxy is not None:
                        mask_tensor = torch.from_numpy(mask_np)
                        if torch.cuda.is_available(): mask_tensor = mask_tensor.cuda()
                        
                        # ⭐ [핵심 수정] text="person"을 같이 넣어줍니다!
                        # 이렇게 하면 "Text or Box required" 에러를 무조건 통과합니다.
                        # SAM3는 Mask를 우선적으로 참고하여 Text("person")에 맞는 객체를 찾습니다.
                        req = dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0, 
                            obj_id=obj_id,
                            mask=mask_tensor,
                            text="person",      # 👈 치트키 추가 (에러 방지용)
                            boxes=np.array([box_xyxy]) # 👈 박스도 같이 (Array 형태)
                        )
                        predictor.handle_request(request=req)
                        
            else:
                print(f"   ⚠️ 파일 없음: {prev_json_path.name}. 텍스트로 재시작.")
                predictor.handle_request(
                    request=dict(type="add_prompt", session_id=session_id, frame_index=0, text="person")
                )

    # 2. 트래킹 실행
    with torch.inference_mode():
        gen = predictor.handle_stream_request(request=dict(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=0, 
        ))
        
        for response in tqdm(gen, total=num_frames_in_chunk, desc=f"Chunk {chunk_start}"):
            if "outputs" not in response: continue
            
            relative_idx = int(response["frame_index"]) 
            real_idx = chunk_start + relative_idx       
            
            outputs = response["outputs"]
            obj_ids = outputs.get("out_obj_ids", outputs.get("obj_ids", []))
            
            if "out_mask_logits" in outputs: mask_data = outputs["out_mask_logits"]; is_logits=True
            elif "mask_logits" in outputs: mask_data = outputs["mask_logits"]; is_logits=True
            else: mask_data = outputs.get("out_binary_masks"); is_logits=False
            
            if mask_data is None: continue

            frame_res = {}
            for i, obj_id in enumerate(obj_ids):
                m = mask_data[i] if isinstance(mask_data, list) else mask_data[i]
                if isinstance(m, torch.Tensor): m_cpu = m.detach().cpu().numpy()
                else: m_cpu = np.array(m)

                mask_bool = (m_cpu > 0.0).astype(np.uint8) if is_logits else m_cpu.astype(np.uint8)
                if mask_bool.sum() < 30: continue
                
                frame_res[int(obj_id)] = {"rle": mask_to_rle(mask_bool)}
            
            # 파일로 저장 (Stateless)
            if frame_res:
                save_path = OUTPUT_DIR / f"{real_idx:05d}.json"
                with open(save_path, 'w') as f:
                    json.dump(frame_res, f)

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    gc.collect()
    torch.cuda.empty_cache()

if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
print("\n🎉 모든 처리가 완료되었습니다!")
predictor.shutdown()