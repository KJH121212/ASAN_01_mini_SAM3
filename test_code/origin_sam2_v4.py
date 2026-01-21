import os
import sys
import torch
import json
import numpy as np
import cv2
import random
import glob
from PIL import Image
from typing import List, Dict, Any
from collections import OrderedDict

# --- SAM3 라이브러리 ---
import sam3
from sam3 import build_sam3_image_model
from sam3.model_builder import build_sam3_video_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage

# sam3_root 경로 설정
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
sys.path.append(f"{sam3_root}/examples")

# --- 설정 ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
GLOBAL_COUNTER = 1

# ==============================================================================
# [Part 1] 이미지 내 객체 검출 (Detection) 관련 함수
# ==============================================================================

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

def detect_objects_in_first_frame(
    video_dir: str, 
    text_prompt: str, 
    model_checkpoint_path: str = None
) -> Dict[str, Any]:
    """
    1. 이미지 프레임 폴더와 프롬프트를 입력받아 첫 번째 프레임에서 객체를 찾습니다.
    Returns:
        detection_results (dict): 검출 결과 (scores, masks, boxes 등)
        image_path (str): 사용된 첫 번째 이미지의 경로
    """
    print("\n" + "="*60)
    print(f" [Step 1] 객체 검출 (Detection) 시작: '{text_prompt}' ")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 첫 번째 프레임 경로 찾기
    # 000000.jpg, 00000.jpg 등 다양한 자릿수 대응
    candidates = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if not candidates:
        print(f"[Error] 폴더 내에 jpg 이미지가 없습니다: {video_dir}")
        return None, None
    
    # 가장 첫 번째 파일 선택 (파일명 정렬 기준)
    img_path = candidates[0]
    print(f"[Info] 첫 번째 프레임 로드: {img_path}")

    # 2. 이미지 모델 로드
    print("[Model] SAM 3 Image Model 로드 중...")
    # bpe_path는 환경에 맞게 수정 필요 (여기서는 기본값 또는 인자값 사용)
    if model_checkpoint_path is None:
        # 기본 경로 예시 (사용자 환경에 맞게 수정 필요)
        model_checkpoint_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
    
    model = build_sam3_image_model(bpe_path=model_checkpoint_path)
    model.to(device)

    # 3. 전처리 및 데이터셋 구성
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False, # GPU 텐서 상태로 유지 (트래커로 바로 넘기기 위함)
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        img_pil = Image.open(img_path).convert("RGB")
        datapoint = create_empty_datapoint()
        set_image(datapoint, img_pil)
        
        print(f"[Prompt] 텍스트 프롬프트 적용: '{text_prompt}'")
        add_text_prompt(datapoint, text_prompt)
        
        datapoint = transform(datapoint)

        # 4. 추론 수행
        print("[Inference] 추론 시작...")
        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)
        
        output = model(batch)
        processed_results = postprocessor.process_results(output, batch.find_metadatas)

    # 5. 결과 추출 (첫 번째 이미지의 결과만 반환)
    if len(processed_results) > 0:
        first_result = list(processed_results.values())[0]
        num_obj = first_result["scores"].numel()
        print(f"[Result] 검출 완료: {num_obj}개의 객체 발견.")
        
        # 모델 메모리 해제
        del model
        del output
        del batch
        torch.cuda.empty_cache()
        
        return first_result, img_path
    else:
        print("[Result] 검출된 결과가 없습니다.")
        return None, img_path


# ==============================================================================
# [Part 2] 비디오 트래킹 (Tracking) 관련 함수 및 클래스
# ==============================================================================

class LazyVideoLoader:
    """이미지를 필요할 때만 디스크에서 읽어오는 로더 (메모리 절약)"""
    def __init__(self, video_path, image_size=1008):
        self.video_path = video_path
        self.image_size = image_size
        self.frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")) + 
                                  glob.glob(os.path.join(video_path, "*.jpeg")) +
                                  glob.glob(os.path.join(video_path, "*.png")))
        try:
            self.frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except:
            self.frame_paths.sort()
        print(f"[LazyLoader] 총 {len(self.frame_paths)}개의 프레임 준비됨.")

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img_path = self.frame_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"이미지 로드 실패: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # SAM3 전처리
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img -= np.array([0.485, 0.456, 0.406])
        img /= np.array([0.229, 0.224, 0.225])
        
        return torch.from_numpy(img).permute(2, 0, 1)

def init_state_lazy(predictor, video_path, offload_state_to_cpu=True):
    """LazyLoader를 사용하는 커스텀 init_state 함수"""
    inference_state = {}
    inference_state["offload_video_to_cpu"] = True
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    inference_state["device"] = predictor.device
    inference_state["storage_device"] = torch.device("cpu") if offload_state_to_cpu else torch.device("cuda")

    inference_state["images"] = LazyVideoLoader(video_path, image_size=predictor.image_size)
    inference_state["num_frames"] = len(inference_state["images"])
    
    # 원본 해상도 (첫 프레임)
    first_img = cv2.imread(inference_state["images"].frame_paths[0])
    inference_state["video_height"] = first_img.shape[0]
    inference_state["video_width"] = first_img.shape[1]
    
    # 기본 구조 초기화
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict"] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    inference_state["first_ann_frame_idx"] = None
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),
        "non_cond_frame_outputs": set(),
    }
    inference_state["tracking_has_started"] = False
    inference_state["frames_already_tracked"] = {}
    
    predictor.clear_all_points_in_video(inference_state)
    return inference_state

def track_objects_in_video(
    image_path: str, 
    detection_results: Dict[str, Any], 
    output_dir: str
):
    """
    2. 검출 결과를 받아 비디오 트래킹을 수행하는 함수
    """
    print("\n" + "="*60)
    print(" [Step 2] 비디오 트래킹 (Tracking) 시작 ")
    print("="*60)

    video_dir = os.path.dirname(image_path)
    print(f"[Info] 비디오 소스 경로: {video_dir}")

    # 마스크 키 확인 (segmentation vs masks)
    mask_key = "masks" if "masks" in detection_results else "segmentation"
    if mask_key not in detection_results:
        print(f"[Error] 검출 결과에 마스크 정보가 없습니다. Keys: {detection_results.keys()}")
        return

    num_detected = detection_results["scores"].numel()
    if num_detected == 0:
        print("[Error] 추적할 객체가 없습니다.")
        return
    
    print(f"[Info] 총 {num_detected}개의 객체 추적 시작.")

    # 1. SAM 3 Video Model 로드
    print("[Model] SAM 3 Video Model (Tracker) 로드 중...")
    sam3_model = build_sam3_video_model(apply_temporal_disambiguation=True, device="cuda")
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    print("[Model] 로드 완료.")

    # 2. Lazy 세션 초기화
    print("[Session] Lazy 추론 세션 초기화...")
    try:
        inference_state = init_state_lazy(
            predictor, 
            video_path=video_dir,
            offload_state_to_cpu=True # [중요] OOM 방지
        )
    except Exception as e:
        print(f"[Error] 세션 초기화 실패: {e}")
        return

    # 3. 마스크 등록
    print(f"[Tracking] {num_detected}개 객체 등록 중...")
    obj_colors = {}
    for i in range(num_detected):
        obj_colors[i+1] = [random.randint(50, 255) for _ in range(3)]

    for i in range(num_detected):
        mask = detection_results[mask_key][i]
        mask_input = mask.cuda().float()
        if mask_input.dim() == 2:
            mask_input = mask_input.unsqueeze(0)
        
        score = detection_results["scores"][i].item()
        obj_id = i + 1

        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            mask=mask_input
        )
        print(f"  - ID:{obj_id} 등록 (Score: {score:.4f})")

    # 4. 트래킹 루프
    print("[Tracking] 전파(Propagation) 시작...")
    vis_output_dir = os.path.join(output_dir, "tracking_results")
    os.makedirs(vis_output_dir, exist_ok=True)

    # 45,000 프레임 처리용 메모리 관리 루프
    for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state, 
        start_frame_idx=0, 
        max_frame_num_to_track=None, 
        reverse=False, 
        propagate_preflight=True 
    ):
        if frame_idx % 100 == 0:
            print(f"  > Frame {frame_idx}/{inference_state['num_frames']} 처리 중... (VRAM 정리)")
            
            # [메모리 관리] 100프레임마다 오래된 과거 데이터 삭제 (필수)
            # output_dict가 무한히 커지는 것을 방지
            cutoff = frame_idx - 50
            if cutoff > 0:
                # 안전하게 삭제 시도
                outputs = inference_state["output_dict"]
                for key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                    keys_to_remove = [k for k in outputs[key] if k < cutoff]
                    for k in keys_to_remove:
                        del outputs[key][k]
                
                # 객체별 딕셔너리도 정리
                for obj_dict in inference_state["output_dict_per_obj"].values():
                    for key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                        keys_to_remove = [k for k in obj_dict[key] if k < cutoff]
                        for k in keys_to_remove:
                            del obj_dict[key][k]

        # --- 시각화 ---
        frame_path = inference_state["images"].frame_paths[frame_idx]
        frame_img = cv2.imread(frame_path)
        if frame_img is None: continue

        if video_res_masks is not None and len(video_res_masks) > 0:
            for k, obj_id in enumerate(obj_ids):
                if isinstance(obj_id, torch.Tensor): obj_id = obj_id.item()
                mask_tensor = video_res_masks[k]
                if mask_tensor.dim() == 3: mask_tensor = mask_tensor.squeeze(0)
                
                mask_bool = mask_tensor.cpu().numpy() > 0.0
                color = obj_colors.get(obj_id, [0, 0, 255])
                
                ys, xs = np.where(mask_bool)
                if len(ys) > 0:
                    alpha = 0.5
                    roi = frame_img[ys, xs]
                    blended = (roi.astype(float) * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
                    frame_img[ys, xs] = blended

        cv2.imwrite(os.path.join(vis_output_dir, f"{frame_idx:06d}.jpg"), frame_img)

    print(f"[Success] 완료. 결과: {vis_output_dir}")
    
    # 정리
    del predictor
    del sam3_model
    del inference_state
    torch.cuda.empty_cache()


# ==============================================================================
# [Main] 메인 실행 함수
# ==============================================================================
def main():
    # 1. 설정
    video_dir = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/1_FRAME/Won_Kim_research_at_Bosanjin/M01/M01_VISIT12"
    prompt = "person"
    output_dir = "./test"
    
    # 2. [함수 1] 객체 검출 실행
    detection_results, first_img_path = detect_objects_in_first_frame(
        video_dir=video_dir,
        text_prompt=prompt
    )

    if detection_results is None:
        print("[Main] 검출 실패로 종료합니다.")
        return

    # 3. 중간 메모리 정리 (중요: 이미지 모델 해제)
    print("[Main] Detection 완료. VRAM 정리를 수행합니다.")
    torch.cuda.empty_cache()

    # 4. [함수 2] 비디오 트래킹 실행
    track_objects_in_video(
        image_path=first_img_path,
        detection_results=detection_results,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()