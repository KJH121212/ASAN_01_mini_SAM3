# 🎥 Video Object Segmentation & Tracking with SAM3

[cite_start]이 프로젝트는 **SAM3(Segment Anything Model 3)** 및 **Sapiens**를 활용하여 비디오 내 객체를 정밀하게 세그멘테이션하고, Bounding Box(BBox), 포인트, 또는 텍스트 프롬프트를 기반으로 객체를 추적(Tracking)하는 도구 모음입니다[cite: 1].

## 📂 Project Structure

제공된 파일들을 기반으로 한 주요 구조는 다음과 같습니다.

* [cite_start]**`runner/`**: 실제 대규모 처리를 위한 실행 스크립트가 포함되어 있습니다[cite: 1].
    * [cite_start]`seg_by_samv3.py`: SAM3 모델을 사용하여 비디오 세그멘테이션을 수행하는 메인 로직입니다[cite: 1].
* [cite_start]**`func/`**: 마스크 변환, 데이터 로드, 텍스트 트래킹 등 핵심 기능을 담은 모듈입니다[cite: 1].
    * `text_tracking_v2.py`: 텍스트 프롬프트를 이용한 객체 검출 및 양방향 트래킹 기능을 제공합니다.
    * `mask_to_bbox.py`: 생성된 마스크에서 BBox를 추출하는 유틸리티입니다.
* [cite_start]**`test_code/`**: 다양한 시나리오(BBox 기반, 포인트 기반, 속도 측정 등)를 테스트하기 위한 Jupyter Notebook 및 스크립트들입니다[cite: 1].
* [cite_start]**`docker/`**: 일관된 실행 환경을 위한 Dockerfile 및 요구사항 정의서입니다[cite: 1].

---

## 🚀 Getting Started

### 1. Environment Setup
이 프로젝트는 GPU 환경(RTX 3090 이상 권장)에서 Docker를 통해 실행하는 것을 최우선으로 설계되었습니다.

```bash
# Docker 이미지 빌드 및 실행 (0_bbox_by_sam3.sh 참고)
docker build -t tojihoo/sam:v1.1 -f ./docker/Dockerfile .
```

### 2. Usage
비디오 세그멘테이션을 실행하려면 `0_bbox_by_sam3.sh` 스크립트를 사용하거나 직접 파이썬 파일을 실행합니다.

**BBox 기반 트래킹 실행 예시:**
```bash
python3 runner/bbox_by_sam3.py
```
* `metadata.csv`를 통해 타겟 비디오와 프레임 정보를 로드합니다.
* 초기 프레임의 Keypoint 정보를 바탕으로 BBox를 생성하고 비디오 전체로 전파(Propagate)합니다.
* 결과는 `video_segments.json` 형태로 저장됩니다.

**텍스트 프롬프트 기반 트래킹:**
`func/text_tracking_v2.py`를 활용하여 특정 텍스트(예: "person", "hand")를 입력하면 객체를 검출하고 양방향(Bidirectional)으로 추적합니다.

---

## 🧪 Key Experiments (Notebooks)
`test_code/` 폴더 내에서 다음 실험들을 수행할 수 있습니다:

| 파일명 | 설명 |
| :--- | :--- |
| **BBOX2SAM.ipynb** | Sapiens로 추출한 BBox를 입력값으로 하여 특정 인스턴스 추적 |
| **point2sam.ipynb** | 특정 좌표(Points)를 지정하여 해당 위치의 객체 추적 |
| **sam_plus_sapiens.ipynb** | SAM으로 추출한 영역을 Sapiens 모델의 입력값으로 연동 |
| **sam_speed.ipynb** | 비디오 해상도 변화에 따른 SAM3의 처리 속도 벤치마크 |

---

## 🛠 Main Features
* **Multi-Prompt Support**: BBox, Point, Text 등 다양한 입력 인터페이스 지원.
* **Bidirectional Tracking**: 기준 프레임을 중심으로 정방향 및 역방향 트래킹을 수행하여 정확도 향상.
* **Memory Optimization**: 긴 비디오 처리 시 VRAM 관리를 위해 과거 캐시를 삭제하는 로직 포함.
* **RLE Encoding**: 마스크 데이터를 효율적으로 저장하기 위해 Run-Length Encoding 포맷 지원.

---