import glob  
import os  
import cv2  
import matplotlib.pyplot as plt  
import numpy as np  
import json
import random
import sam3
import torch

from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import login
from tqdm import tqdm


# 🐱 1. device 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("no GPU")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# 🐱 2. path 정리
frame_path = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/walking_data/FRAME/frontal__walking__1")
# frame_path = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/1_FRAME/Won_Kim_research_at_Bosanjin/M01/M01_VISIT2_UpperLimb")
data_path = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
output_path = data_path / "walking_data/sam"

# checkpoint path
bpe_path = data_path / "checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
checkpoint_path = data_path / "checkpoints/SAM3/sam3.pt"
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# # path 검증
# paths_to_check = {
#     "frame_path": frame_path,
#     "data_path": data_path,
#     "output_path": output_path,
#     "bpe_path": bpe_path,
#     "checkpoint_path": checkpoint_path
# }

# for name, path in paths_to_check.items():
#     # 경로가 존재하면 True, 없으면 False 출력
#     print(f"{name} exists?: {os.path.exists(path)}")


# 🐱3. hugging face 로그인

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACE_TOKEN을 env 파일에서 찾을 수 없음")


login(token=hf_token)

# 🐱4. SAM3  모델 빌딩
from sam3.model_builder import build_sam3_video_model  # SAM3 비디오 모델을 생성하기 위한 빌더 함수를 임포트합니다.

sam3_model = build_sam3_video_model(
    checkpoint_path=checkpoint_path,
    bpe_path=bpe_path
)               # 정의된 설정에 따라 SAM3 비디오 모델 인스턴스를 생성하고 초기화합니다.
predictor = sam3_model.tracker                      # 모델 내부에서 비디오 내 객체 추적을 담당하는 'tracker' 컴포넌트를 predictor 변수로 가져옵니다.
predictor.backbone = sam3_model.detector.backbone   # 메모리 효율성을 위해 Detector(탐지기)가 사용하는 백본 네트워크를 Tracker와 공유하도록 참조를 연결합니다.

# # infrence state 초기화 
# # 비디오 프레임 로드해서 ram에 올림->추적에 필요한 memory bank 초기화, 반환된 inference state 객체는 이후의 모든 작업에 필수적 사용
# inference_state = predictor.init_state(video_path=str(frame_path), offload_video_to_cpu=True)
# # 현재 추론 상태(inference_state)에 저장된 모든 사용자 입력 포인트와 상호작용 정보를 삭제하여 초기화합니다.
# predictor.clear_all_points_in_video(inference_state)  

# [수정된 전략 1] 0번 프레임에서만 YOLO로 사람을 찾고 끝낸다.
print("🚀 [Initial Prompt] 0번 프레임에서만 사람을 찾습니다.")


# # ==============================================================================
# # 🛠️ Helper Function: Mask RLE Encoding
# # ==============================================================================
# def mask_to_rle(mask_bool_np):
#     pixels = mask_bool_np.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return runs.tolist()

# # ==============================================================================
# # 1. 📂 데이터 로드 (Lazy Loading 준비)
# # ==============================================================================
# # 프레임 파일 리스트 생성
# frame_files = sorted([
#     os.path.join(str(frame_path), f) 
#     for f in os.listdir(frame_path) 
#     if f.lower().endswith(('.jpg', '.jpeg', '.png'))
# ])

# if not frame_files:
#     raise FileNotFoundError("폴더에 이미지가 없습니다.")

# # 첫 프레임 읽어서 해상도 정보 획득
# frame0 = cv2.imread(frame_files[0])
# height, width = frame0.shape[:2]
# total_frames = len(frame_files)

# print(f"✅ 데이터 준비 완료: 총 {total_frames} 프레임 ({width}x{height})")


# # ==============================================================================
# # 2. 🤖 주기적 텍스트 프롬프트 적용 (Multiple People & New Entries)
# # ==============================================================================
# # 중간에 새로 등장하는 사람을 찾기 위해, 0번 뿐만 아니라 일정 간격으로 스캔합니다.
# PROMPT_INTERVAL = 30  # 30프레임(약 1초)마다 "person" 있는지 스캔
# txt_prompt = "person"

# print(f"🚀 [Prompting] '{txt_prompt}'를 {PROMPT_INTERVAL}프레임 간격으로 검색합니다...")

# # 스캔할 프레임 인덱스 목록 (0, 30, 60, ... 끝까지)
# scan_frame_indices = list(range(0, total_frames, PROMPT_INTERVAL))

# # 진행률 표시와 함께 프롬프트 주입
# for scan_idx in tqdm(scan_frame_indices, desc="Scanning for people"):
    
#     # 해당 프레임에 "person" 텍스트 프롬프트 적용
#     # add_new_points_or_box는 해당 텍스트에 맞는 객체들을 찾아 새로운 ID를 부여하고 반환합니다.
#     # (이미 추적 중인 객체와 겹치더라도, SAM 모델 특성상 새로운 ID가 발급될 수 있습니다)
#     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#         inference_state=inference_state,
#         frame_idx=scan_idx,
#         obj_id=None,     # None으로 주면 모델이 알아서 새 ID 할당 (또는 리스트 반환)
#         text=txt_prompt 
#     )

# print("✅ 모든 구간 프롬프트 입력 완료. 이제 전체 경로를 연결(Propagation)합니다.")


# # ==============================================================================
# # 3. 🏃‍♂️ 전체 비디오 트래킹 (Propagation)
# # ==============================================================================
# # 저장할 디렉토리 설정
# vis_output_dir = output_path / "visualizations"
# vis_output_dir.mkdir(parents=True, exist_ok=True)
# json_output_path = output_path / "tracking_results.json"

# # 시각화할 랜덤 프레임 선정 (10장)
# num_vis = min(10, total_frames)
# vis_indices = set(random.sample(range(total_frames), k=num_vis))
# vis_indices.add(0) # 0번은 꼭 포함
# sorted_vis_indices = sorted(list(vis_indices))

# print(f"🏃‍♂️ [Tracking] 트래킹 시작... (결과는 {json_output_path}에 저장)")

# video_tracking_results = {}

# # 전체 비디오 전파 (Propagation)
# # propagate_in_video는 입력된 모든 프롬프트(0, 30, 60...)를 고려하여 마스크를 생성합니다.
# for i, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(predictor.propagate_in_video(inference_state)):
    
#     if i % 50 == 0:
#         print(f"   Processing frame {out_frame_idx}/{total_frames}...")

#     # --- A. JSON 데이터 구조화 ---
#     frame_results = {}
    
#     # 감지된 모든 객체(ID)에 대해 반복
#     for k, obj_id in enumerate(out_obj_ids):
#         # 마스크 추출 (Threshold 0.0)
#         mask_bool_np = (out_mask_logits[k] > 0.0).cpu().numpy().astype(np.uint8)
        
#         # 마스크가 너무 작거나 없으면 저장 생략 (노이즈 방지)
#         if mask_bool_np.sum() < 10: 
#             continue

#         # RLE 인코딩
#         rle_mask = mask_to_rle(mask_bool_np)
        
#         frame_results[int(obj_id)] = {
#             "rle_mask": rle_mask,
#             "height": height,
#             "width": width
#         }
    
#     if frame_results: # 감지된 객체가 있을 때만 저장
#         video_tracking_results[out_frame_idx] = frame_results

#     # --------------------------------------------------------------------------
#     # B. 랜덤 시각화 (JPG 저장) + 🆔 ID 표시 추가
#     # --------------------------------------------------------------------------
#     if out_frame_idx in sorted_vis_indices:
#         # 1. 해당 프레임 이미지 로드
#         current_frame = cv2.imread(frame_files[out_frame_idx])
#         current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

#         plt.figure(figsize=(12, 8))
#         plt.title(f"Frame {out_frame_idx} | Detected IDs: {out_obj_ids}")
#         plt.imshow(current_frame)
#         plt.axis('off')

#         # 2. 마스크 및 ID 그리기
#         for k, obj_id in enumerate(out_obj_ids):
#             mask_bool_np = (out_mask_logits[k] > 0.0).cpu().numpy()
            
#             # 마스크가 너무 작으면(노이즈) 건너뛰기
#             if mask_bool_np.sum() < 10:
#                 continue

#             # (1) 마스크 색칠하기 (기존 함수)
#             show_mask(mask_bool_np, plt.gca(), obj_id=obj_id)

#             # (2) 🆔 ID 텍스트 표시 (New!)
#             # 마스크의 좌표(True인 픽셀들)를 찾습니다.
#             y_coords, x_coords = np.where(mask_bool_np)
            
#             # 마스크의 중심점(Centroid) 계산
#             x_center = int(np.mean(x_coords))
#             y_center = int(np.mean(y_coords))
            
#             # 중심점에 텍스트 출력 (흰색 글씨 + 검은색 배경박스)
#             plt.text(x_center, y_center, f"ID {obj_id}", 
#                      color='white', 
#                      fontsize=12, 
#                      fontweight='bold', 
#                      ha='center', va='center',
#                      bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.5))

#         # 3. 저장 및 닫기
#         save_name = vis_output_dir / f"result_{out_frame_idx:05d}.jpg"
#         plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
#         plt.close() # 메모리 해제

# # ==============================================================================
# # 4. 💾 결과 저장
# # ==============================================================================
# with open(json_output_path, 'w') as f:
#     json.dump(video_tracking_results, f) # 용량 문제로 indent 제거 권장

# print("\n🎉 완료!")
# print(f"📊 총 {len(video_tracking_results)}개 프레임에서 객체 감지됨")
# print(f"📁 JSON 경로: {json_output_path}")
# print(f"🖼️ 시각화 경로: {vis_output_dir}")