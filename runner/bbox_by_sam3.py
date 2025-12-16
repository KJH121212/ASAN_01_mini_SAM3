from pathlib import Path
import pandas as pd
import json
import time
from tqdm import tqdm
import numpy as np
from sam3.model_builder import build_sam3_video_model
import sys
sys.path.append('/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sam3')
from func.mask_to_bbox import mask_to_bbox

# -----------------------------------------------------------------------------
# 1. 설정 및 데이터 로드 (Configuration & Data Loading)
# -----------------------------------------------------------------------------

# Path 정리
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_sam3/")
CSV_PATH = DATA_DIR / "metadata.csv"
OUTPUT_PATH = DATA_DIR / "test"
CHECKPOINT_DIR = DATA_DIR / "checkpoints/SAM3"
CHECKPOINT_PT = CHECKPOINT_DIR / "sam3.pt"

# CSV 불러오기 및 타겟 설정
df = pd.read_csv(CSV_PATH)
target = 3                             # 원하는 행 인덱스 설정

for target in range(1,2):
    start_time = time.time()

    # 데이터 추출
    COMMON_PATH = df.loc[target, "common_path"]   # COMMON_PATH 추출
    VIDEO_PTH = df.loc[target, "video_path"]
    N_FRAMES = df.loc[target, "n_frames"]         # 프레임 수 추출

    print("segment video : ", COMMON_PATH)

    # 세부 경로 설정
    FRAME_DIR = DATA_DIR / "1_FRAME" / COMMON_PATH          # 프레임 디렉토리 경로 설정
    KPT_DIR = DATA_DIR / "2_KEYPOINTS" / COMMON_PATH        # 키포인트 디렉토리 경로 설정
    MP4_PATH = DATA_DIR / "3_MP4" / f"{COMMON_PATH}.mp4"    # MP4 디렉토리 경로 설정    
    INTERP_DIR = DATA_DIR / "4_INTERP_DATA" / COMMON_PATH   # 보간 데이터 디렉토리 경로 설정
    SAVE_DIR = OUTPUT_PATH / COMMON_PATH                    # 저장 경로 설정
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_SAVE_PATH = SAVE_DIR / "video_segments.json"
    TIME_LOG_PATH = SAVE_DIR / "elapsed_time.txt"  # 시간 기록을 저장할 파일 경로 설정

    # Keypoint 데이터 로드 및 BBox 계산
    with open(KPT_DIR / "000000.json", 'r') as f:
        kpt_data = json.load(f)

    ori_bbox = kpt_data['instance_info'][0]['bbox'][0]
    ori_bbox = np.array(ori_bbox, dtype=np.float32)

    kpt_width, kpt_height = 1280, 720

    rel_box = [[
        ori_bbox[0] / kpt_width,
        ori_bbox[1] / kpt_height,
        ori_bbox[2] / kpt_width,
        ori_bbox[3] / kpt_height
    ]]
    rel_box = np.array(rel_box, dtype=np.float32)

    # -----------------------------------------------------------------------------
    # 2. 모델 초기화 (Model Initialization)
    # -----------------------------------------------------------------------------
    sam3_model = build_sam3_video_model(checkpoint_path=CHECKPOINT_PT)  # SAM3 비디오 모델 빌드
    predictor = sam3_model.tracker                                      # SAM3 비디오 예측기 초기화
    predictor.backbone = sam3_model.detector.backbone                   # 백본 설정

    print("SAM3 비디오 모델과 예측기가 성공적으로 초기화되었습니다.")

    # -----------------------------------------------------------------------------
    # 3. 추론 준비 및 프롬프트 입력 (Inference Setup & Prompting)
    # -----------------------------------------------------------------------------
    inference_state = predictor.init_state(video_path=VIDEO_PTH)    # 비디오 추론 상태 초기화
    predictor.clear_all_points_in_video(inference_state)            # 비디오의 모든 포인트 지우기

    ann_frame_idx = 0   # the frame index we interact with
    ann_obj_id = 4      # give a unique id to each object we interact with

    # 초기 포인트/박스 추가
    _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=rel_box,
    )

    # 해상도 복원용 변수 (사용자 코드 유지)
    width = 1920
    height = 1080

    box = np.array([[ 
        rel_box[0][0] * width,
        rel_box[0][1] * height,
        rel_box[0][2] * width,
        rel_box[0][3] * height
    ]], dtype=np.float32)

    # -----------------------------------------------------------------------------
    # 4. 비디오 전파 및 결과 수집 (Video Propagation)
    # -----------------------------------------------------------------------------
    video_segments = {}  # video_segments contains the per-frame segmentation results

    # run propagation throughout the video and collect the results in a dict
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=N_FRAMES, reverse=False, propagate_preflight=True):
        video_segments[frame_idx] = {
            out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # -----------------------------------------------------------------------------
    # 5. 결과 변환 및 저장 (Result Processing & Saving)
    # -----------------------------------------------------------------------------
    print("🔄 JSON 변환 및 저장 준비 중...")

    json_output = {}

    for frame_idx, segments in tqdm(video_segments.items()):
        json_output[str(frame_idx)] = {} # JSON 키는 문자열이어야 안전함
        
        for obj_id, mask in segments.items():
            # 마스크 차원 정리
            if mask.ndim == 3:
                mask = mask.squeeze()
                
            bbox = mask_to_bbox(mask)
            
            json_output[str(frame_idx)][str(obj_id)] = {
                "bbox": bbox,
            }

    try:
        print(f"💾 JSON 파일 저장 시작: {JSON_SAVE_PATH}")
        
        with open(JSON_SAVE_PATH, 'w') as f:
            json.dump(json_output, f) # indent=4를 빼면 용량이 줄어듭니다.
            
        print(f"✅ 저장 완료! : {JSON_SAVE_PATH}")
        
    except Exception as e:
        print(f"❌ 저장 실패: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"총 걸린 시간 = {elapsed_time:.2f} seconds")
    # -----------------------------------------------------------------------------
    # 7. 실행 시간 별도 저장 (Save Elapsed Time)
    # -----------------------------------------------------------------------------

    try:
        with open(TIME_LOG_PATH, 'w') as f:
            f.write(f"Total execution time: {elapsed_time:.2f} seconds\n")  # 시간을 소수점 둘째 자리까지 기록
            f.write(f"Processed Frames: {N_FRAMES}\n")  # (선택사항) 처리한 프레임 수도 함께 적으면 분석에 더 좋습니다.
            
        print(f"⏱️ 실행 시간 기록 저장 완료: {TIME_LOG_PATH}")  # 저장 완료 메시지 출력

    except Exception as e:
        print(f"❌ 시간 기록 저장 실패: {e}")  # 에러 발생 시 예외 처리