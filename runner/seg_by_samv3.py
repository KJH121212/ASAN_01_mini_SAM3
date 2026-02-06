import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
import traceback
import torch
import multiprocessing as mp  # 멀티프로세싱 모듈

# --- 1. 환경 설정 ---
# (중요) PyTorch 메모리 단편화 방지 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/")
from func.huggingface_login import login_to_huggingface

# --- 2. 경로 설정 (전역 변수) ---
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "new_metadata.csv"
NEW_METADATA_PATH = DATA_DIR / "new_metadata.csv"
CHECKPOINT_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/sam3.pt"
BPE_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
ENV_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/.env"

# ==============================================================================
# [Worker] 개별 비디오를 처리하는 작업자 (별도 프로세스에서 실행됨)
# ==============================================================================
def process_video_worker(common_path, start_frame_idx=0):
    """
    하나의 비디오를 처리하고 종료되는 함수입니다.
    이 함수가 끝나면 프로세스가 소멸되며 GPU 메모리가 100% 반환됩니다.
    """
    try:
        # 라이브러리 임포트 (프로세스 내부에서 로드)
        import torch
        import gc
        from func.text_tracking_v2 import detect_objects, run_bidirectional_tracking
        from sam3 import build_sam3_image_model
        from sam3.model_builder import build_sam3_video_model

        # 로그인 (필요시)
        login_to_huggingface(ENV_PATH)

        # 경로 설정
        curr_frame_path = DATA_DIR / "1_FRAME" / common_path
        curr_output_path = DATA_DIR / "8_SAM" / common_path
        prompt = "person"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"   ▶ [Worker] 시작: {common_path}")

        # ------------------------------------------------------
        # [Step 1] Image Model Load -> Detect -> Unload
        # ------------------------------------------------------
        image_model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH, bpe_path=BPE_PATH)
        image_model.to(device)
        
        detection_res = detect_objects(
            str(curr_frame_path), prompt, start_frame_idx, 
            model=image_model
        )
        
        # 모델 삭제 및 캐시 정리
        del image_model
        gc.collect()
        torch.cuda.empty_cache()

        if not detection_res:
            print(f"   ❌ [Worker] 객체 미검출")
            return False

        # ------------------------------------------------------
        # [Step 2] Video Model Load -> Tracking -> Unload
        # ------------------------------------------------------
        video_model = build_sam3_video_model(checkpoint_path=CHECKPOINT_PATH, apply_temporal_disambiguation=True, device=device)
        
        run_bidirectional_tracking(
            str(curr_frame_path), detection_res, str(curr_output_path), start_frame_idx,
            model=video_model
        )
        
        del video_model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"   ✅ [Worker] 처리 완료")
        return True

    except Exception as e:
        print(f"   🔥 [Worker Error] {e}")
        traceback.print_exc()
        return False

# ==============================================================================
# [Main] 메인 프로세스 (작업 관리 및 진행상황 기록)
# ==============================================================================
def main():
    # 1. 데이터 로드
    if NEW_METADATA_PATH.exists():
        print(f"📂 이어하기: '{NEW_METADATA_PATH.name}' 로드")
        load_path = NEW_METADATA_PATH
    else:
        print(f"📂 초기 시작: '{METADATA_PATH.name}' 로드")
        load_path = METADATA_PATH

    if not load_path.exists():
        print("[ERROR] 메타데이터 없음")
        return

    df = pd.read_csv(load_path)
    if "sam_done" not in df.columns:
        df["sam_done"] = False
    
    df["sam_done"] = df["sam_done"].fillna(False).astype(bool)
    todo_df = df[~df["sam_done"]] 
    
    total_count = len(df)
    todo_count = len(todo_df)

    print("="*60)
    print(f"🚀 SAM3 Batch Processing (Process Isolation Mode)")
    print(f"🛡️  메모리 누수 방지를 위해 '프로세스 격리' 방식으로 실행합니다.")
    print(f"🔥 남은 작업량: {todo_count}개")
    print("="*60)

    success_count = 0
    fail_count = 0

    # 2. 반복 처리 (Process Spawning)
    # multiprocessing의 'spawn' 방식을 사용하여 GPU 컨텍스트 충돌 방지
    ctx = mp.get_context('spawn')

    for idx, row in todo_df.iterrows():
        common_path = row["common_path"]
        curr_frame_path = DATA_DIR / "1_FRAME" / common_path
        
        print(f"\n[Progress] {success_count + fail_count + 1}/{todo_count} | Path: {common_path}")

        if not curr_frame_path.exists():
            print(f"   ⚠️ [Skip] 폴더 없음")
            fail_count += 1
            continue

        start_time = time.time()

        # ⭐ 핵심: 별도 프로세스 생성 및 실행
        # 메인 프로세스는 여기서 대기(join)하고, 워커 프로세스가 GPU를 쓰고 죽습니다.
        p = ctx.Process(target=process_video_worker, args=(common_path, 0))
        p.start()
        p.join() # 프로세스가 끝날 때까지 대기
        
        # 종료 코드 확인 (0이면 정상 종료 -> 성공으로 간주하지 않고, 결과 파일 확인이 더 정확하지만 일단 exitcode 활용)
        # process_video_worker는 성공/실패 여부를 반환할 수 없으므로(Process 특징),
        # 여기서는 단순히 에러 없이 끝났는지를 체크합니다.
        # 더 확실한 방법: SAM 결과 파일(JSON)이 생성되었는지 확인
        
        sam_output_dir = DATA_DIR / "8_SAM" / common_path
        json_files = list(sam_output_dir.glob("*.json")) if sam_output_dir.exists() else []

        if p.exitcode == 0 and len(json_files) > 0:
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"   ✅ 메인: 성공 확인 ({minutes}분 {seconds:.2f}초)")
            
            df.at[idx, "sam_done"] = True
            df.to_csv(NEW_METADATA_PATH, index=False)
            success_count += 1
        else:
            print(f"   ❌ 메인: 실패 또는 비정상 종료 (ExitCode: {p.exitcode})")
            fail_count += 1
            # 실패했어도 기록 저장은 선택 (여기선 False 유지)

    print("\n🏁 모든 작업 종료")

if __name__ == "__main__":
    main()