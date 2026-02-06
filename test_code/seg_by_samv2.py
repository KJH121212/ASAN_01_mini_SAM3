import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
import traceback
import torch
import multiprocessing as mp
import math

# --- 1. 환경 설정 ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 경로 설정 (사용자 환경에 맞게 유지)
sys.path.append("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/")
from func.huggingface_login import login_to_huggingface

DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
METADATA_PATH = DATA_DIR / "new_metadata.csv"
NEW_METADATA_PATH = DATA_DIR / "new_metadata.csv"
CHECKPOINT_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/sam3.pt"
BPE_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
ENV_PATH = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/.env"

# ⭐ 배치 사이즈 설정 (한 번 프로세스를 띄울 때 처리할 비디오 개수)
# OOM이 걱정되면 3~5 정도로 시작, 괜찮으면 10으로 늘리세요.
BATCH_SIZE = 5 

# ==============================================================================
# [Worker] 배치(여러 개) 비디오를 처리하는 작업자
# ==============================================================================
def process_batch_worker(paths_list, start_frame_idx=0):
    """
    여러 개의 비디오(paths_list)를 순차적으로 처리하고 종료되는 함수
    """
    try:
        # 라이브러리 로드는 프로세스 당 1회만 수행 (오버헤드 감소)
        import torch
        import gc
        from func.text_tracking_v2 import detect_objects, run_bidirectional_tracking
        from sam3 import build_sam3_image_model
        from sam3.model_builder import build_sam3_video_model

        login_to_huggingface(ENV_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        results = {} # 결과 저장용 딕셔너리 {path: success_bool}

        print(f"   ▶ [Worker] 배치 시작 (총 {len(paths_list)}개)")

        for common_path in paths_list:
            try:
                curr_frame_path = DATA_DIR / "1_FRAME" / common_path
                curr_output_path = DATA_DIR / "8_SAM" / common_path
                prompt = "person"

                print(f"     ▷ 처리 중: {common_path}")

                # ------------------------------------------------------
                # [Step 1] Image Model
                # ------------------------------------------------------
                image_model = build_sam3_image_model(checkpoint_path=CHECKPOINT_PATH, bpe_path=BPE_PATH)
                image_model.to(device)
                
                detection_res = detect_objects(
                    str(curr_frame_path), prompt, start_frame_idx, 
                    model=image_model
                )
                
                del image_model
                gc.collect()
                torch.cuda.empty_cache()

                if not detection_res:
                    print(f"     ❌ 객체 미검출: {common_path}")
                    results[common_path] = False
                    continue

                # ------------------------------------------------------
                # [Step 2] Video Model
                # ------------------------------------------------------
                video_model = build_sam3_video_model(checkpoint_path=CHECKPOINT_PATH, apply_temporal_disambiguation=True, device=device)
                
                run_bidirectional_tracking(
                    str(curr_frame_path), detection_res, str(curr_output_path), start_frame_idx,
                    model=video_model
                )
                
                del video_model
                gc.collect()
                torch.cuda.empty_cache()
                
                # (중요) IPC 메모리 정리 (공유 메모리 누수 방지)
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()

                print(f"     ✅ 완료: {common_path}")
                results[common_path] = True

            except Exception as e:
                print(f"     🔥 개별 에러 ({common_path}): {e}")
                traceback.print_exc()
                results[common_path] = False
                
                # 에러 발생 시에도 메모리 비우기 시도
                gc.collect()
                torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"   🔥 [Worker Critical Error] 프로세스 전체 실패: {e}")
        traceback.print_exc()
        return {}

# ==============================================================================
# [Main] 메인 프로세스
# ==============================================================================
def main():
    if NEW_METADATA_PATH.exists():
        print(f"📂 이어하기: '{NEW_METADATA_PATH.name}' 로드")
        load_path = NEW_METADATA_PATH
    else:
        print(f"📂 초기 시작: '{METADATA_PATH.name}' 로드")
        load_path = METADATA_PATH

    df = pd.read_csv(load_path)
    if "sam_done" not in df.columns:
        df["sam_done"] = False
    
    df["sam_done"] = df["sam_done"].fillna(False).astype(bool)
    todo_df = df[~df["sam_done"]]
    
    # 전체 경로 리스트 추출
    todo_paths = todo_df["common_path"].tolist()
    total_count = len(todo_paths)
    
    # ⭐ 배치(Chunk)로 나누기
    # 예: [path1, path2, path3, ...] -> [[p1, p2, p3], [p4, p5, p6], ...]
    num_batches = math.ceil(total_count / BATCH_SIZE)
    path_chunks = [todo_paths[i:i + BATCH_SIZE] for i in range(0, total_count, BATCH_SIZE)]

    print("="*60)
    print(f"🚀 SAM3 Batch Processing (Batch Size: {BATCH_SIZE})")
    print(f"🔥 총 작업: {total_count}개 | 총 배치 수: {num_batches}회")
    print("="*60)

    ctx = mp.get_context('spawn')
    
    processed_count = 0

    for i, batch_paths in enumerate(path_chunks):
        print(f"\n📦 [Batch {i+1}/{num_batches}] 시작 (포함된 비디오: {len(batch_paths)}개)")
        
        start_time = time.time()

        # 워커 프로세스 실행
        # (주의: Process는 리턴값을 직접 받을 수 없으므로, 파일 시스템으로 성공 여부 확인)
        p = ctx.Process(target=process_batch_worker, args=(batch_paths, 0))
        p.start()
        p.join()

        # 배치 처리 후 메타데이터 업데이트 확인
        success_in_batch = 0
        
        # 이번 배치에 포함된 경로들 확인
        for path in batch_paths:
            sam_output_dir = DATA_DIR / "8_SAM" / path
            # 파일이 생성되었는지 확인
            json_files = list(sam_output_dir.glob("*.json")) if sam_output_dir.exists() else []
            
            if len(json_files) > 0:
                # 성공 처리
                df.loc[df["common_path"] == path, "sam_done"] = True
                success_in_batch += 1
            else:
                # 실패 처리 (이미 False 겠지만 확실히)
                print(f"   ⚠️ 실패 감지: {path}")

        # CSV 저장 (배치 끝날 때마다 저장)
        df.to_csv(NEW_METADATA_PATH, index=False)
        
        elapsed = time.time() - start_time
        processed_count += len(batch_paths)
        
        print(f"   🏁 배치 {i+1} 종료 | 성공: {success_in_batch}/{len(batch_paths)} | 소요시간: {elapsed:.1f}초")
        print(f"   💾 메타데이터 저장 완료 ({processed_count}/{total_count} 진행 중)")

    print("\n🏁 모든 작업 종료")

if __name__ == "__main__":
    main()