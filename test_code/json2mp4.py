import os
import glob
import json
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# ⚙️ 1. 경로 및 설정
# ==============================================================================
FRAME_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/1_FRAME/AI_dataset/N01/N01_Treatment/diagonal__biceps_curl")
JSON_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/test/sam3/AI_dataset/N01/N01_Treatment/diagonal__biceps_curl_v4.0")
VIDEO_OUTPUT_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/test/sam3/sam3_Won2_v4.0.mp4")

FPS = 30.0
ALPHA = 0.5 

# ==============================================================================
# 🛠️ 2. 유틸리티 함수
# ==============================================================================
def rle_to_mask(rle, height, width):
    """
    [start, length, start, length...] 형태의 RLE 리스트를 마스크로 변환
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if not rle: 
        return mask.reshape((height, width))
    
    rle = np.array(rle)
    
    # [Start, Length] 쌍으로 분리
    starts = rle[0::2]
    lengths = rle[1::2]
    
    # 1-based index 보정 (-1)
    starts = starts - 1 
    ends = starts + lengths
    
    for lo, hi in zip(starts, ends):
        if lo < 0: lo = 0
        if hi > len(mask): hi = len(mask)
        mask[lo:hi] = 1
        
    return mask.reshape((height, width))

color_map = {}
def get_color(obj_id):
    if obj_id not in color_map:
        color_map[obj_id] = [
            random.randint(50, 255), 
            random.randint(50, 255), 
            random.randint(50, 255)
        ]
    return color_map[obj_id]

# ==============================================================================
# 🚀 3. 비디오 생성 로직
# ==============================================================================
frame_files = sorted(glob.glob(str(FRAME_DIR / "*.jpg")))
total_frames = len(frame_files)

if total_frames == 0:
    print("❌ 에러: 원본 프레임 이미지를 찾을 수 없습니다.")
    exit()

first_img = cv2.imread(frame_files[0])
height, width, layers = first_img.shape
print(f"🎬 영상 정보: {width}x{height}, 총 {total_frames} 프레임")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(VIDEO_OUTPUT_PATH), fourcc, FPS, (width, height))

print(f"⏳ 렌더링 시작... (저장 경로: {VIDEO_OUTPUT_PATH})")

for i, img_path in enumerate(tqdm(frame_files)):
    frame = cv2.imread(img_path)
    if frame is None: continue
    
    overlay = frame.copy()
    
    # 파일명 포맷 (6자리 숫자)
    json_filename = f"{i:06d}.json" 
    json_path = JSON_DIR / json_filename
    
    shapes_found = False
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # [수정된 파싱 로직]
            # data는 딕셔너리이고, 그 안에 "objects"라는 리스트가 있음.
            if "objects" in data:
                for obj in data["objects"]:
                    shapes_found = True
                    
                    obj_id = obj["id"]
                    
                    # [핵심 수정] segmentation 안에 counts가 들어있음!
                    # 구조: obj -> segmentation -> counts
                    rle_counts = obj["segmentation"]["counts"] 
                    
                    # 마스크 생성
                    mask = rle_to_mask(rle_counts, height, width)
                    
                    if mask.sum() > 0:
                        color = get_color(obj_id)
                        
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, color, 2) 
                        cv2.fillPoly(overlay, contours, color) 

                        y, x = np.where(mask)
                        if len(y) > 0:
                            cy, cx = int(np.mean(y)), int(np.mean(x))
                            label = f"ID {obj_id}"
                            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame, (cx, cy - h_text - 5), (cx + w_text, cy + 5), color, -1)
                            cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        except Exception as e:
            # 에러 발생 시 상세 내용 출력 (디버깅용)
            print(f"⚠️ 프레임 {i} 처리 중 에러: {e}")

    if shapes_found:
        cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)
    
    counter_text = f"Frame: {i} / {total_frames}"
    cv2.putText(frame, counter_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, counter_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)

    out.write(frame)

out.release()
print("\n🎉 변환 완료!")