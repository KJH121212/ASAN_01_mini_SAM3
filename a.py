from func.data import get_paths, get_total_count
import os

def check_system():
    print("🔎 설정 검증을 시작합니다...\n")

    # 1. CSV 로드 확인
    try:
        total = get_total_count()
        print(f"✅ 메타데이터 CSV 로드 성공! (총 {total}개의 데이터)")
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return

    if total == 0:
        print("⚠️ 데이터가 0개입니다. CSV 파일을 확인해주세요.")
        return

    # 2. 첫 번째 타겟(0번)으로 경로 검증
    target_num = 0
    print(f"\n[{target_num}번 데이터] 경로 생성 테스트 중...")

    try:
        paths = get_paths(target_num)
        
        # 3. 데이터 내용 출력 (눈으로 확인)
        print("-" * 50)
        print(f"📌 Common Path : {paths.common_path}")
        print(f"📌 Frame Count : {paths.n_frames}")
        print(f"📌 MP4 Path    : {paths.mp4_path}")
        print(f"📌 Save Dir    : {paths.save_dir}")
        print("-" * 50)

        # 4. 실제 파일/폴더 존재 여부 체크 (핵심!)
        print("\n[물리적 파일 확인]")
        
        # 영상 파일 확인
        if paths.mp4_path.exists():
            print(f"✅ 원본 영상 파일이 존재합니다.")
        else:
            print(f"❌ 원본 영상을 찾을 수 없습니다!")
            print(f"   -> 경로: {paths.mp4_path}")
            print("   -> 폴더명이나 파일명이 맞는지 확인해보세요.")

        # 저장 폴더 생성 확인 (get_paths 호출 시 생성되어야 함)
        if paths.save_dir.exists():
            print(f"✅ 결과 저장 폴더가 생성되었습니다.")
        else:
            print(f"❌ 결과 저장 폴더가 생성되지 않았습니다.")

    except Exception as e:
        print(f"❌ 경로 생성 중 에러 발생: {e}")

if __name__ == "__main__":
    check_system()