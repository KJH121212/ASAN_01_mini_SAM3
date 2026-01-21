import os  # 환경 변수 접근을 위해 os 모듈을 임포트합니다.
from huggingface_hub import login  # Hugging Face 로그인을 위한 함수를 가져옵니다.
from dotenv import load_dotenv  # .env 파일을 로드하기 위한 라이브러리를 가져옵니다.

def login_to_huggingface(env_path: str, token_key: str = "HUGGINGFACE_TOKEN") -> None:
    """
    지정된 .env 파일 경로에서 토큰을 불러와 Hugging Face에 로그인하는 함수입니다.
    """
    # 1. 환경 변수 로드
    if not os.path.exists(env_path): # 파일이 실제로 존재하는지 먼저 확인합니다.
        raise FileNotFoundError(f"지정된 경로에 .env 파일이 없습니다: {env_path}") # 파일이 없으면 명확한 에러를 발생시킵니다.

    load_dotenv(env_path)  # 지정된 경로의 .env 파일을 읽어 환경 변수로 로드합니다.
    hf_token = os.getenv(token_key)  # 로드된 환경 변수에서 지정한 키(예: HUGGINGFACE_TOKEN)의 값을 가져옵니다.

    # 2. 토큰 유효성 검사
    if hf_token is None:  # 토큰 값을 가져오지 못했는지 확인합니다.
        raise ValueError(f"{token_key}가 환경 변수 파일({env_path}) 내에 존재하지 않습니다.")  # 토큰이 없으면 에러를 발생시켜 실행을 중단합니다.

    # 3. 로그인 시도
    login(token=hf_token)  # 유효한 토큰을 사용하여 Hugging Face에 로그인을 시도합니다.
    print(f"Hugging Face 로그인 성공! (Token Key: {token_key})")  # 성공적으로 로그인되었음을 사용자에게 알립니다.

# --- 사용 예시 ---
if __name__ == "__main__":
    # 사용자의 특정 경로 변수 지정
    my_env_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/.env" # 사용자가 제공한 원본 경로를 변수에 담습니다.
    
    try:
        login_to_huggingface(my_env_path) # 함수를 호출하여 로그인을 수행합니다.
    except Exception as e:
        print(f"로그인 중 오류 발생: {e}") # 에러 발생 시 프로그램이 멈추지 않고 메시지를 출력하도록 예외 처리를 합니다.