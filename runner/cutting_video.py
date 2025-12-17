from pathlib import Path
import subprocess

# -------------------------------------------------------
# 🎯 설정: 루트 경로 지정
# -------------------------------------------------------
# 원본 영상들이 있는 최상위 폴더
SOURCE_ROOT = Path("/workspace/nas203/ds_RehabilitationMedicineData/data/d02/Won_Kim_research_at_Bosanjin")

# 결과물이 저장될 최상위 폴더 (사용자 폴더)
OUTPUT_ROOT = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/Won_Kim_research_at_Bosanjin")

# 처리할 비디오 확장자 목록
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

print(f"🚀 일괄 처리 시작")
print(f"📂 원본 루트: {SOURCE_ROOT}")
print(f"📂 출력 루트: {OUTPUT_ROOT}")
print("-" * 60)

# -------------------------------------------------------
# ✂ FFmpeg 컷팅 함수 (재사용 및 개선)
# -------------------------------------------------------
def cut_video(input_video, start_frame, end_frame, output_video, fps=30):
    """
    특정 프레임 구간을 잘라내어 저장합니다.
    """
    # 시간 계산 (초 단위, 소수점 포함하여 정밀도 유지)
    start_seconds = start_frame / fps
    duration_seconds = (end_frame - start_frame) / fps

    cmd = [
        "ffmpeg",
        "-y",               # 덮어쓰기 허용
        "-ss", f"{start_seconds:.3f}", # 시작 시간 (초)
        "-i", str(input_video),        # 입력 파일 (-ss 뒤에 오면 더 정확하게 자름)
        "-t", f"{duration_seconds:.3f}", # 지속 시간 (초)
        "-c", "copy",       # 스트림 복사 (재인코딩 없음, 빠름)
        "-avoid_negative_ts", "make_zero", # 타임스탬프 보정
        str(output_video)
    ]

    # 로그가 너무 길어지지 않게 디버그 출력은 주석 처리하거나 필요시 켭니다.
    # print(f"[DEBUG] 명령어: {' '.join(cmd)}")
    
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"  ❌ [FFmpeg 실패] {output_video.name}")
        # 에러 로그가 너무 길면 핵심만 출력
        print(f"     Msg: {proc.stderr.splitlines()[-1] if proc.stderr else 'Unknown Error'}")
    else:
        if output_video.exists():
            size_kb = output_video.stat().st_size / 1024
            print(f"  ✅ 생성완료: {output_video.name} ({size_kb:.1f} KB)")
        else:
            print(f"  ⚠️ 생성실패 (파일없음): {output_video.name}")

# -------------------------------------------------------
# 🔄 폴더 순회 및 일괄 실행 로직
# -------------------------------------------------------

# SOURCE_ROOT 하위의 모든 파일을 뒤져서 비디오 파일만 찾습니다.
video_files = [p for p in SOURCE_ROOT.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]

if not video_files:
    print("❌ 처리할 비디오 파일을 찾지 못했습니다.")
    exit()

print(f"🔍 총 {len(video_files)}개의 비디오 파일을 발견했습니다.\n")

for idx, video_path in enumerate(video_files, 1):
    # 1. 경로 계산
    # 예: SOURCE_ROOT/M03/VideoName.mp4 -> relative_path는 "M03/VideoName.mp4"
    relative_path = video_path.relative_to(SOURCE_ROOT)
    
    # 출력 폴더 구조를 잡습니다.
    # 예: OUTPUT_ROOT/M03/VideoName/ (여기에 segment txt가 있어야 함)
    # video_path.stem은 확장자를 뺀 파일명입니다.
    target_dir_name = video_path.stem 
    current_output_dir = OUTPUT_ROOT / relative_path.parent / target_dir_name
    
    segment_txt = current_output_dir / "segment_frames.txt"

    print(f"[{idx}/{len(video_files)}] 처리 중: {video_path.name}")
    
    # 2. segment_frames.txt 존재 확인
    if not segment_txt.exists():
        print(f"  ⏭️ Skip: 설정 파일이 없습니다. ({segment_txt})")
        continue

    # 3. 출력 폴더 생성 (이미 txt가 있다면 폴더는 있겠지만 안전하게)
    current_output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 기존 mp4 삭제 (초기화)
    old_videos = list(current_output_dir.glob("*.mp4"))
    if old_videos:
        for v in old_videos:
            v.unlink()
        print(f"  🗑️ 기존 영상 {len(old_videos)}개 삭제 완료")

    # 5. Segment 파일 파싱
    segments = []
    try:
        with open(segment_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 파싱 포맷: label, start, end
                parts = line.split(",")
                if len(parts) >= 3:
                    label = parts[0].strip()
                    start = int(parts[1].strip())
                    end = int(parts[2].strip())
                    segments.append((start, end, label))
    except Exception as e:
        print(f"  ❌ 텍스트 파일 읽기 오류: {e}")
        continue

    if not segments:
        print("  ⚠️ 구간 정보가 비어있습니다.")
        continue

    # 6. 컷팅 실행
    for start, end, label in segments:
        output_path = current_output_dir / f"{label}.mp4"
        cut_video(video_path, start, end, output_path)

    print("-" * 40)

print("\n🎉 모든 작업이 완료되었습니다.")