import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# 1. 고정된 기본 경로 설정 (수정이 필요하면 여기만 고치세요)
# -----------------------------------------------------------------------------
# 사용자님의 경로 환경에 맞췄습니다.
DATA_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data")
CSV_PATH = DATA_DIR / "metadata.csv"

# 데이터프레임을 모듈 로딩 시 한 번만 읽어둡니다. (효율성)
try:
    _df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"⚠️ 경고: {CSV_PATH} 파일을 찾을 수 없습니다.")
    _df = pd.DataFrame() # 빈 데이터프레임으로 에러 방지

# -----------------------------------------------------------------------------
# 2. 반환될 경로 객체 구조 (자동완성 지원용)
# -----------------------------------------------------------------------------
@dataclass
class VideoPaths:
    """비디오 하나에 대한 경로 모음집"""
    target_idx: int      # 입력한 타겟 번호
    common_path: str     # 공통 경로 이름 (Key)
    n_frames: int        # 프레임 수
    
    # 디렉토리 경로
    frame_dir: Path      # 1_FRAME
    kpt_dir: Path        # 2_KEYPOINTS
    interp_dir: Path     # 4_INTERP_DATA
    save_dir: Path       # 결과 저장 폴더 (OUTPUT)
    
    # 파일 경로
    mp4_path: Path       # 3_MP4 원본 영상
    json_path: Path      # 저장될 JSON
    time_log_path: Path  # 시간 기록 텍스트

# -----------------------------------------------------------------------------
# 3. [핵심] 타겟 넘버로 경로 가져오는 함수
# -----------------------------------------------------------------------------
def get_paths(target_num: int) -> VideoPaths:
    """
    타겟 번호(Index)를 입력하면 해당 비디오의 모든 경로를 반환합니다.
    폴더가 없으면 자동으로 생성해줍니다.
    """
    if _df.empty:
        raise RuntimeError("메타데이터 CSV가 로드되지 않았습니다.")
        
    if target_num < 0 or target_num >= len(_df):
        raise IndexError(f"Target {target_num}은 존재하지 않습니다. (0 ~ {len(_df)-1})")

    # 1. CSV에서 정보 추출
    row = _df.iloc[target_num]
    common_path = str(row["common_path"])
    n_frames = int(row["n_frames"])

    # 2. 경로 조립 (사용자님의 규칙 적용)
    frame_dir  = DATA_DIR / "1_FRAME" / common_path
    kpt_dir    = DATA_DIR / "2_KEYPOINTS" / common_path
    mp4_path   = DATA_DIR / "3_MP4" / f"{common_path}.mp4"
    interp_dir = DATA_DIR / "4_INTERP_DATA" / common_path
    
    # 저장 경로 (test 폴더)
    save_dir = DATA_DIR / "test" / common_path
    
    # [편의 기능] 저장 폴더가 없으면 미리 만들어줍니다.
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. 결과 객체 반환
    return VideoPaths(
        target_idx=target_num,
        common_path=common_path,
        n_frames=n_frames,
        
        frame_dir=frame_dir,
        kpt_dir=kpt_dir,
        interp_dir=interp_dir,
        save_dir=save_dir,
        
        mp4_path=mp4_path,
        json_path=save_dir / "video_segments.json",
        time_log_path=save_dir / "elapsed_time.txt"
    )

# (선택 사항) 전체 데이터 개수 확인용
def get_total_count():
    return len(_df)