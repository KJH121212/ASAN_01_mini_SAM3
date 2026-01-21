import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt  # 시각화 저장을 위해 추가
from PIL import Image
from typing import List

# --- SAM3 라이브러리 임포트 ---
import sam3
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage

# sam3_root 경로 설정
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
sys.path.append(f"{sam3_root}/examples")
from sam3.visualization_utils import plot_results

# --- 설정 (Configuration) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
GLOBAL_COUNTER = 1

# --- 헬퍼 함수 (이전과 동일) ---
def create_empty_datapoint():
    return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

def add_text_prompt(datapoint, text_query):
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "이미지를 먼저 설정해주세요."
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1

# --- 메인 실행 로직 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 0. 결과 저장 디렉터리 생성
    output_dir = "./test"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 모델 로드
    print("모델 로드 중...")
    bpe_path = f"/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    model.to(device)

    # 2. Transform 정의
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # 3. 후처리기 정의
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )

    # Context Manager
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        
        # --- 이미지 및 프롬프트 설정 ---
        # [변경됨] 로컬 이미지 경로 사용
        img_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/1_FRAME/Won_Kim_research_at_Bosanjin/M01/M01_VISIT12/000000.jpg"
        
        if not os.path.exists(img_path):
            print(f"Error: 이미지를 찾을 수 없습니다: {img_path}")
            return

        print(f"이미지 로드 중: {img_path}")
        img1 = Image.open(img_path).convert("RGB") # PIL Image로 로드
        
        datapoint1 = create_empty_datapoint()
        set_image(datapoint1, img1)
        
        # [변경됨] 프롬프트 "person" 적용
        print("프롬프트 적용: 'person'")
        id1 = add_text_prompt(datapoint1, "person")
        
        datapoint1 = transform(datapoint1)

        # --- 배치 처리 및 추론 ---
        print("추론 시작...")
        batch = collate([datapoint1], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)
        
        output = model(batch)
        processed_results = postprocessor.process_results(output, batch.find_metadatas)


        def convert_tensor_to_serializable(obj, summarize_masks=True):
            """
            텐서를 JSON 저장이 가능한 형태(list, float, str)로 변환합니다.
            """
            if isinstance(obj, torch.Tensor):
                # 텐서를 CPU로 옮기고 리스트로 변환
                if obj.numel() > 1000 and summarize_masks:
                    # 요소가 너무 많으면(특히 마스크) 요약 정보만 반환
                    return f"<Tensor shape={obj.shape}, dtype={obj.dtype}, device={obj.device}>"
                else:
                    # 박스(boxes)나 점수(scores) 같은 작은 텐서는 값 전체 변환
                    # bfloat16은 json 호환을 위해 float32로 변환
                    return obj.detach().float().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensor_to_serializable(v, summarize_masks) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensor_to_serializable(v, summarize_masks) for v in obj]
            else:
                return obj

        # ---------------------------------------------------------
        # 사용 예시 (processed_results가 있다고 가정)
        # ---------------------------------------------------------
        # output_dir 설정 (위에서 만든 ./test 폴더 활용)
        output_path = "./test/processed_results.json"

        # 변환 수행 (summarize_masks=True로 하면 마스크는 shape만 보임)
        serializable_results = convert_tensor_to_serializable(processed_results, summarize_masks=False)

        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4)

        print(f"JSON 파일이 저장되었습니다: {output_path}")
        # # --- 결과 저장 ---
        # print(f"결과 저장 중... ({output_dir})")
        
        # # Matplotlib Figure 생성 및 저장
        # plt.figure(figsize=(10, 10))
        
        # # plot_results는 현재 활성화된 plt figure에 그림을 그립니다.
        # plot_results(img1, processed_results[id1]) 
        
        # save_path = os.path.join(output_dir, "result_person.png")
        # plt.axis('off') # 축 제거
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        # plt.close()
        
        # print(f"저장 완료: {save_path}")

if __name__ == "__main__":
    main()