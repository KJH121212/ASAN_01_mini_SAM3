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
FRAME_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/walking_data/FRAME/frontal__walking__1")
JSON_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/output_per_frame_json")
VIDEO_OUTPUT_PATH = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/ASAN_01_mini_SAM3/result_video_60fps_with_counter.mp4")

FPS = 60.0
ALPHA = 0.5

# ==============================================================================
# 🛠️ 2. 유틸리티 함수
# ==============================================================================
def rle_to_mask(rle, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)
    if not rle: return mask.reshape((height, width))
    rle = np.array(rle)
    starts = rle[0::2] - 1
    lengths = rle[1::2]
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        if lo < 0: lo = 0
        if hi > len(mask): hi = len(mask)
        mask[lo:hi] = 1
    return mask.reshape((height, width))

color_map = {}
def get_color(obj_id):
    if obj_id not in color_map:
        # 가시성 좋은 파스텔톤/밝은색 위주 랜덤 생성
        color_map[obj_id] = [
            random.randint(50, 255), 
            random.randint(100, 255), 
            random.randint(100, 255)
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
print(f"🎬 영상 크기: {width}x{height}, 총 프레임: {total_frames}장")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(VIDEO_OUTPUT_PATH), fourcc, FPS, (width, height))

print(f"⏳ 비디오 렌더링 시작... (Output: {VIDEO_OUTPUT_PATH})")

for i, img_path in enumerate(tqdm(frame_files)):
    frame = cv2.imread(img_path)
    if frame is None: continue
    
    overlay = frame.copy()
    json_path = JSON_DIR / f"{i:05d}.json"
    shapes_found = False
    
    # --- 마스크 그리기 ---
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if data:
                shapes_found = True
                for obj_id_str, info in data.items():
                    obj_id = int(obj_id_str)
                    mask = rle_to_mask(info['rle'], height, width)
                    
                    if mask.sum() > 0:
                        color = get_color(obj_id)
                        
                        # 윤곽선 및 내부 채우기
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, color, 2)
                        cv2.fillPoly(overlay, contours, color) 

                        # 객체 ID 라벨 (객체 중심)
                        y, x = np.where(mask)
                        if len(y) > 0:
                            cy, cx = int(np.mean(y)), int(np.mean(x))
                            label = f"ID {obj_id}"
                            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            # 라벨 배경
                            cv2.rectangle(frame, (cx, cy - h_text - 5), (cx + w_text, cy + 5), color, -1)
                            # 라벨 텍스트 (검은색 글씨로 대비 강조)
                            cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        except Exception as e:
            pass

    # 투명도 적용
    if shapes_found:
        cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)
    
    # ⭐ [추가됨] 좌측 상단 프레임 카운터 표시 ⭐
    counter_text = f"Frame: {i} / {total_frames}"
    
    # 텍스트 크기 계산
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(counter_text, font, font_scale, thickness)
    
    # 텍스트 위치 (좌측 상단, 여백 30px)
    x, y = 30, 60
    
    # 검은색 배경 박스 그리기 (가독성 확보)
    cv2.rectangle(frame, (x - 10, y - text_h - 10), (x + text_w + 10, y + baseline + 10), (0, 0, 0), -1)
    
    # 흰색 텍스트 쓰기
    cv2.putText(frame, counter_text, (x, y), font, font_scale, (255, 255, 255), thickness)

    # 비디오 쓰기
    out.write(frame)

out.release()
print("\n🎉 변환 완료!")
print(f"💾 저장된 파일: {VIDEO_OUTPUT_PATH}")