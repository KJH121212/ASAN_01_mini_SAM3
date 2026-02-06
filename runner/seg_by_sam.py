import sys
sys.path.append("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/")

from func.huggingface_login import login_to_huggingface
my_env_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/.env"
login_to_huggingface(my_env_path)

import os
import pandas as pd
import numpy as np
from pathlib import Path
import time

data_dir = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
metadata_path = data_dir / "metadata.csv"

df = pd.read_csv(metadata_path)

common_path = df["common_path"][0]
frame_path = data_dir / "1_FRAME" / common_path
output_path = data_dir / "8_SAM" / common_path
checkpoint_path = data_dir / "checkpoints/SAM3" / "sam3.pt"
bpe_path = data_dir / "checkpoints/SAM3" / "bpe_simple_vocab_16e6.txt.gz"

from func.text_tracking import detect_objects, run_bidirectional_tracking

# ==============================================================================
# [Main] 메인 실행부
# ==============================================================================
def main():
    # ▼ 경로 설정 ▼
    prompt = "person"
    
    # ⭐ [중요] 시작할 프레임 번호 (이 프레임에서 객체를 찾고 앞뒤로 퍼짐)
    START_FRAME_IDX = 0  # 예: 400번 프레임에서 시작
    # ▲ 경로 설정 ▲

    print("="*60)
    print(f"🚀 SAM3 Bidirectional Tracking 시작")
    print(f"📂 입력 폴더: {frame_path}")
    print(f"🎯 시작 프레임: {START_FRAME_IDX}")
    print("="*60)

    start_time = time.time()

    # 1. 지정된 프레임에서 Detection
    detection_res = detect_objects(frame_path, prompt, target_frame_idx=START_FRAME_IDX)
    
    # 2. 양방향 Tracking
    if detection_res:
        run_bidirectional_tracking(frame_path, detection_res, output_path, start_frame_idx=START_FRAME_IDX)
        
        end_time = time.time()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        
        print("\n" + "="*60)
        print(f"✅ [완료] 모든 작업이 정상적으로 끝났습니다.")
        print(f"⏱️  총 소요 시간: {minutes}분 {seconds:.2f}초")
        print("="*60)
    else:
        print("❌ 객체를 찾지 못해 종료합니다.")

if __name__ == "__main__":
    main()