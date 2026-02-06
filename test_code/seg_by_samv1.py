import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
import traceback

# --- 1. 환경 설정 및 라이브러리 로드 ---
sys.path.append("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/")

from func.huggingface_login import login_to_huggingface
from func.text_tracking import detect_objects, run_bidirectional_tracking

# 환경 변수 로드 및 로그인
my_env_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/.env"
login_to_huggingface(my_env_path)

# --- 2. 경로 및 데이터 설정 ---
data_dir = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
metadata_path = data_dir / "metadata.csv"
new_metadata_path = data_dir / "new_metadata.csv"

# ==============================================================================
# [Main] 메인 실행부
# ==============================================================================
def main():
    # ----------------------------------------------------------------------
    # 1. 데이터 로드 (이어하기 로직)
    # ----------------------------------------------------------------------
    # 이미 생성된 new_metadata.csv가 있다면 그것을 로드하여 진행 상황을 이어갑니다.
    if new_metadata_path.exists():
        print(f"📂 기존 진행 기록 발견! '{new_metadata_path.name}'을 로드합니다.")
        load_path = new_metadata_path
    else:
        print(f"📂 초기 시작: '{metadata_path.name}'을 로드합니다.")
        load_path = metadata_path

    if not load_path.exists():
        print(f"[ERROR] 메타데이터 파일을 찾을 수 없습니다: {load_path}")
        return

    df = pd.read_csv(load_path)

    # ----------------------------------------------------------------------
    # 2. 데이터 전처리 (Boolean 변환 및 필터링)
    # ----------------------------------------------------------------------
    # 컬럼이 없으면 생성
    if "sam_done" not in df.columns:
        df["sam_done"] = False
    
    # 확실한 필터링을 위해 NaN값을 False로 채우고, Boolean 타입으로 강제 변환
    df["sam_done"] = df["sam_done"].fillna(False).astype(bool)

    # 해야 할 작업만 필터링 (sam_done이 False인 행들)
    # 주의: iterate는 todo_df로 하지만, 상태 업데이트는 원본 df에 해야 합니다.
    todo_df = df[~df["sam_done"]] 

    total_count = len(df)
    todo_count = len(todo_df)
    done_count = total_count - todo_count

    print("="*60)
    print(f"🚀 SAM3 Batch Processing 시작")
    print(f"📊 전체 데이터: {total_count}개")
    print(f"✅ 이미 완료됨: {done_count}개")
    print(f"🔥 남은 작업량: {todo_count}개 (여기서부터 시작합니다)")
    print("="*60)

    if todo_count == 0:
        print("🎉 모든 작업이 이미 완료되어 있습니다! 종료합니다.")
        return

    success_count = 0
    fail_count = 0

    # ----------------------------------------------------------------------
    # 3. 반복 처리 (Filter된 리스트만 순회)
    # ----------------------------------------------------------------------
    # todo_df의 index는 원본 df의 index와 동일하게 유지됩니다.
    for idx, row in todo_df.iterrows():
        try:
            # --- 개별 비디오 정보 추출 ---
            common_path = row["common_path"]
            
            # 경로 동적 생성
            curr_frame_path = data_dir / "1_FRAME" / common_path
            curr_output_path = data_dir / "8_SAM" / common_path
            
            # 파라미터 설정
            prompt = "person"
            start_frame_idx = 0 
            
            # --- 진행 상황 로그 ---
            # (현재 몇 번째 처리 중인지 / 남은 개수)
            print(f"\n[Progress] 남은 작업 {success_count + fail_count + 1}/{todo_count} | ID: {idx} | Path: {common_path}")

            if not curr_frame_path.exists():
                print(f"   ⚠️ [Skip] 프레임 폴더 없음: {curr_frame_path}")
                fail_count += 1
                continue

            # --- 처리 시작 ---
            start_time = time.time()

            # 1. 객체 검출
            detection_res = detect_objects(
                str(curr_frame_path), 
                prompt, 
                target_frame_idx=start_frame_idx
            )
            
            # 2. 양방향 트래킹
            if detection_res:
                run_bidirectional_tracking(
                    str(curr_frame_path), 
                    detection_res, 
                    str(curr_output_path), 
                    start_frame_idx=start_frame_idx
                )
                
                # --- [성공 시 원본 DF 업데이트 및 저장] ---
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = elapsed % 60
                print(f"   ✅ [성공] 처리 완료 ({minutes}분 {seconds:.2f}초)")
                
                # ⭐ 중요: 원본 df에 기록해야 전체 상태가 유지됨
                df.at[idx, "sam_done"] = True
                df.to_csv(new_metadata_path, index=False)  # 실시간 저장
                
                success_count += 1
            else:
                print(f"   ❌ [실패] 객체를 찾지 못했습니다.")
                # 실패 시 False 유지 (혹은 실패 표시를 따로 하려면 로직 추가)
                fail_count += 1

        except Exception as e:
            print(f"   🔥 [ERROR] 처리 중 오류 발생: {e}")
            traceback.print_exc()
            fail_count += 1
            # 에러 발생 시에도 안전하게 저장
            df.to_csv(new_metadata_path, index=False)
            continue

    # ----------------------------------------------------------------------
    # 4. 최종 리포트
    # ----------------------------------------------------------------------
    print("\n" + "="*60)
    print("🏁 금일 할당된 작업이 종료되었습니다.")
    print(f"📊 시도한 작업: {todo_count}개")
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {fail_count}개")
    print(f"💾 최종 상태 저장됨: {new_metadata_path}")
    print("="*60)

if __name__ == "__main__":
    main()