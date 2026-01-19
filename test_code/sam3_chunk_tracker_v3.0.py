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

ORIGIN_FRAME_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/walking_data/FRAME/frontal__walking__1")
OUTPUT_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/output_per_frame_json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = OUTPUT_DIR / "temp_chunk_frames"

STEP_SIZE = 200   # 실제 처리 진도
LOOK_AHEAD = 10   # 겹침 구간 (Handshake Zone)

# ==============================================================================
# 🛠️ 유틸리티
# ==============================================================================
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0.0
    return intersection / union

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

# ==============================================================================
# 🚀 메인 실행 로직
# ==============================================================================
all_frame_files = sorted([str(p) for p in ORIGIN_FRAME_PATH.glob("*.jpg")])
total_frames = len(all_frame_files)
gpus = range(torch.cuda.device_count())

_img = cv2.imread(all_frame_files[0])
IMG_HEIGHT, IMG_WIDTH = _img.shape[:2]

# 전체 비디오에서 사용된 ID 중 가장 큰 값 (신규 ID 발급용)
GLOBAL_MAX_ID = -1

print("🔄 모델 로딩 중...")
predictor = build_sam3_video_predictor(gpus_to_use=gpus)
print(f"🎬 전체 {total_frames} 프레임 | Overlap Handshake (구간 매칭) 모드")

for chunk_start in range(0, total_frames, STEP_SIZE):
    
    run_end = min(chunk_start + STEP_SIZE + LOOK_AHEAD, total_frames)
    num_frames_in_chunk = create_virtual_chunk_folder(chunk_start, run_end, all_frame_files)
    
    print(f"\n📁 [Chunk] {chunk_start}~{run_end} ({num_frames_in_chunk}장)")

    # 1. 세션 시작
    response = predictor.handle_request(request=dict(type="start_session", resource_path=str(TEMP_DIR)))
    session_id = response["session_id"]
    
    # ⭐ 이번 청크의 ID 매핑 테이블 (Model ID -> Global ID)
    local_id_map = {} 

    # 2. 프롬프트 (단순화: 그냥 'person'만 주고 모델이 찾게 함 -> 이후 Overlap에서 매핑)
    with torch.inference_mode():
        # 시작할 때 텍스트 프롬프트로 객체 탐지 유도
        predictor.handle_request(
            request=dict(type="add_prompt", session_id=session_id, frame_index=0, text="person")
        )
        
        # (옵션) 만약 Overlap 구간에 대해 힌트를 더 주고 싶다면,
        # 이전 청크의 마지막 마스크를 프롬프트로 넣어줄 수도 있음.
        # 하지만 "IOU를 통해 ID를 부여하는 방식"을 원하셨으므로, 
        # 모델이 스스로 찾게 두고 결과단에서 매핑하는 것이 더 깔끔함.

    # 3. 트래킹 및 Handshake
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
            current_model_ids = outputs.get("out_obj_ids", outputs.get("obj_ids", []))
            
            # 마스크 데이터 로드
            if "out_mask_logits" in outputs: mask_data = outputs["out_mask_logits"]; is_logits=True
            elif "mask_logits" in outputs: mask_data = outputs["mask_logits"]; is_logits=True
            else: mask_data = outputs.get("out_binary_masks"); is_logits=False
            
            if mask_data is None: continue

            # ==================================================================
            # 🤝 Handshake Logic (겹치는 0~10 구간에서만 수행)
            # ==================================================================
            if chunk_start > 0 and relative_idx < LOOK_AHEAD:
                # 이미 저장된 '정답지(이전 청크의 결과)' 로드
                saved_json_path = OUTPUT_DIR / f"{real_idx:05d}.json"
                
                if saved_json_path.exists():
                    with open(saved_json_path, 'r') as f:
                        saved_data = json.load(f) # {saved_id: {rle: ...}}
                    
                    # 현재 모델이 찾은 각 객체에 대해
                    for i, model_id in enumerate(current_model_ids):
                        model_id_int = int(model_id)
                        
                        # 이미 매핑 끝난 놈은 패스
                        if model_id_int in local_id_map: continue
                        
                        # 현재 마스크 디코딩
                        m = mask_data[i] if isinstance(mask_data, list) else mask_data[i]
                        if isinstance(m, torch.Tensor): m_cpu = m.detach().cpu().numpy()
                        else: m_cpu = np.array(m)
                        curr_mask_bool = (m_cpu > 0.0) if is_logits else m_cpu.astype(bool)

                        # 정답지들과 IoU 비교
                        best_iou = 0
                        matched_global_id = None
                        
                        for saved_id_str, info in saved_data.items():
                            saved_id = int(saved_id_str)
                            saved_mask = rle_to_mask(info['rle'], (IMG_HEIGHT, IMG_WIDTH)).astype(bool)
                            
                            iou = calculate_iou(curr_mask_bool, saved_mask)
                            # 겹침이 50% 이상이면 같은 객체로 인정
                            if iou > 0.5 and iou > best_iou:
                                best_iou = iou
                                matched_global_id = saved_id
                        
                        # 매핑 등록
                        if matched_global_id is not None:
                            local_id_map[model_id_int] = matched_global_id
                            # print(f"     🤝 [Handshake] Frame {real_idx}: Model {model_id_int} == Saved {matched_global_id} (IoU {best_iou:.2f})")

            # ==================================================================
            # 💾 저장 및 신규 ID 처리
            # ==================================================================
            frame_res = {}
            for i, raw_id in enumerate(current_model_ids):
                raw_id_int = int(raw_id)
                
                # 1. 매핑 확인
                if raw_id_int in local_id_map:
                    final_id = local_id_map[raw_id_int]
                else:
                    # 2. 매핑이 안 된 경우 (Overlap 구간 지났는데도 없음 = 신규 등장)
                    # 혹은 첫 번째 청크인 경우
                    
                    # 전역 ID 카운터 증가 및 새 번호표 발급
                    GLOBAL_MAX_ID += 1
                    local_id_map[raw_id_int] = GLOBAL_MAX_ID
                    final_id = GLOBAL_MAX_ID
                    print(f"     🆕 [New Object] Frame {real_idx}: Model {raw_id_int} -> New Global {final_id}")
                
                # 데이터 처리
                m = mask_data[i] if isinstance(mask_data, list) else mask_data[i]
                if isinstance(m, torch.Tensor): m_cpu = m.detach().cpu().numpy()
                else: m_cpu = np.array(m)

                mask_bool = (m_cpu > 0.0).astype(np.uint8) if is_logits else m_cpu.astype(np.uint8)
                if mask_bool.sum() < 30: continue
                
                frame_res[final_id] = {"rle": mask_to_rle(mask_bool)}
            
            # 저장 (Overlap 구간에서는 이전 파일을 덮어쓰게 되는데, 
            # 이는 최신(현재 청크)의 추론이 더 연속적일 수 있으므로 괜찮음.
            # 하지만 Handshake의 목적은 ID를 맞추는 것이므로, 
            # ID가 맞춰졌다면 덮어써도 논리적 오류는 없음)
            if frame_res:
                save_path = OUTPUT_DIR / f"{real_idx:05d}.json"
                with open(save_path, 'w') as f:
                    json.dump(frame_res, f)

    # 세션 정리
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    gc.collect()
    torch.cuda.empty_cache()

if TEMP_DIR.exists(): shutil.rmtree(TEMP_DIR)
print(f"\n🎉 완료! Overlap Handshake 방식이 적용되었습니다. (최종 Max ID: {GLOBAL_MAX_ID})")
predictor.shutdown()