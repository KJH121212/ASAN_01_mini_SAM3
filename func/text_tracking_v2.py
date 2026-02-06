import os
import sys
import torch
import json
import numpy as np
import cv2
import glob
from collections import OrderedDict
from PIL import Image
from typing import Dict, Any

# --- SAM3 라이브러리 ---
import sam3
from sam3 import build_sam3_image_model
from sam3.model_builder import build_sam3_video_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
sys.path.append(f"{sam3_root}/examples")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
GLOBAL_COUNTER = 1

# ==============================================================================
# [Helper] 유틸리티
# ==============================================================================
def mask_to_rle(mask):
    """이진 마스크를 RLE 포맷으로 변환"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {"size": mask.shape, "counts": runs.tolist()}

def mask_to_bbox(mask):
    """
    이진 마스크로부터 Bounding Box 추출
    Returns: [x_min, y_min, x_max, y_max]
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None  # 객체가 없음

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # int형으로 변환하여 반환
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def create_empty_datapoint(): return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

def add_text_prompt(datapoint, text_query):
    global GLOBAL_COUNTER
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query, image_id=0, object_ids_output=[], is_exhaustive=True, query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER, original_image_id=GLOBAL_COUNTER, original_category_id=1,
                original_size=[w, h], object_id=0, frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1

# ==============================================================================
# [Part 1] 객체 검출 (Model 인자 지원)
# ==============================================================================
def detect_objects(frame_dir: str, text_prompt: str, target_frame_idx: int = 0, model=None) -> Dict[str, Any]:
    # print(f"--- [Step 1] 객체 검출 (Frame: {target_frame_idx}, Prompt: '{text_prompt}') ---")
    
    candidates = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")) + 
                        glob.glob(os.path.join(frame_dir, "*.png")))
    
    # 숫자 기준 정렬 (파일명 000000.jpg 등 고려)
    try: candidates.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except: candidates.sort()

    if not candidates:
        print(f"[Error] 이미지 없음: {frame_dir}")
        return None

    if target_frame_idx >= len(candidates):
        print(f"[Error] 요청한 프레임 인덱스({target_frame_idx})가 전체 프레임 수({len(candidates)})보다 큽니다.")
        return None

    img_path = candidates[target_frame_idx]
    # print(f"🎯 기준 이미지 로드: {os.path.basename(img_path)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⭐ 중요: 외부에서 모델이 들어오면 그것을 사용, 없으면 로드(Fallback)
    if model is None:
        print("⚠️ [Warning] Image Model이 전달되지 않아 새로 로드합니다.")
        checkpoint_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/sam3.pt"
        bpe_path = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/tojihoo/data/checkpoints/SAM3/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(checkpoint_path=checkpoint_path, bpe_path=bpe_path)
        model.to(device)

    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(
        max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True, use_original_sizes_mask=True,
        convert_mask_to_rle=False, detection_threshold=0.5, to_cpu=False
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        datapoint = create_empty_datapoint()
        set_image(datapoint, Image.open(img_path).convert("RGB"))
        add_text_prompt(datapoint, text_prompt)
        datapoint = transform(datapoint)
        
        batch = collate([datapoint], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)
        processed_results = postprocessor.process_results(model(batch), batch.find_metadatas)

    del batch # 배치는 삭제하되, 외부에서 받은 model은 여기서 삭제하지 않음
    
    if len(processed_results) > 0:
        result = list(processed_results.values())[0]
        # print(f"[Result] {result['scores'].numel()}개 객체 발견.")
        return result
    return None

# ==============================================================================
# [Part 2] 양방향 트래킹 및 저장 (Model 인자 지원)
# ==============================================================================
class LazyVideoLoader:
    def __init__(self, video_path, image_size=1008):
        self.frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")) + 
                                  glob.glob(os.path.join(video_path, "*.png")))
        try: self.frame_paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except: self.frame_paths.sort()
        self.image_size = image_size

    def __len__(self): return len(self.frame_paths)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.frame_paths[idx]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = (img.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(img).permute(2, 0, 1)

def init_state_lazy(predictor, video_path):
    state = {
        "offload_video_to_cpu": True, "offload_state_to_cpu": True,
        "device": predictor.device, "storage_device": torch.device("cpu"),
        "images": LazyVideoLoader(video_path, predictor.image_size),
        "point_inputs_per_obj": {}, "mask_inputs_per_obj": {}, "cached_features": {}, "constants": {},
        "obj_id_to_idx": OrderedDict(), "obj_idx_to_id": OrderedDict(), "obj_ids": [],
        "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        "tracking_has_started": False, "frames_already_tracked": {},
        "first_ann_frame_idx": None, "output_dict_per_obj": {}, "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}
    }
    state["num_frames"] = len(state["images"])
    first = cv2.imread(state["images"].frame_paths[0])
    state["video_height"], state["video_width"] = first.shape[:2]
    predictor.clear_all_points_in_video(state)
    return state

def save_frame_json(frame_idx, obj_ids, video_res_masks, state, json_output_dir):
    """JSON 저장 헬퍼 함수"""
    frame_data = {
        "frame_index": frame_idx,
        "file_name": os.path.basename(state["images"].frame_paths[frame_idx]),
        "objects": []
    }
    if video_res_masks is not None and len(video_res_masks) > 0:
        for k, obj_id in enumerate(obj_ids):
            if isinstance(obj_id, torch.Tensor): obj_id = obj_id.item()
            mask_tensor = video_res_masks[k]
            if mask_tensor.dim() == 3: mask_tensor = mask_tensor.squeeze(0)
            mask_np = (mask_tensor.cpu().numpy() > 0.0).astype(np.uint8)
            
            if np.any(mask_np):
                rle = mask_to_rle(mask_np)
                bbox = mask_to_bbox(mask_np)
                obj_info = {"id": obj_id, "segmentation": rle, "bbox": bbox}
                frame_data["objects"].append(obj_info)

    json_path = os.path.join(json_output_dir, f"{frame_idx:06d}.json")
    try:
        with open(json_path, 'w') as f:
            json.dump(frame_data, f)
    except Exception as e:
        print(f"Save Error: {e}")

def run_bidirectional_tracking(frame_dir: str, detection_results: Dict, json_output_dir: str, start_frame_idx: int, model=None):
    # print(f"--- [Step 2] 양방향 트래킹 시작 (기준 Frame: {start_frame_idx}) ---")
    os.makedirs(json_output_dir, exist_ok=True)
    mask_key = "masks" if "masks" in detection_results else "segmentation"
    num_objs = detection_results["scores"].numel()

    # ⭐ 중요: 외부에서 모델을 받아서 사용
    if model is None:
        print("⚠️ [Warning] Video Model이 전달되지 않아 새로 로드합니다.")
        sam3_model = build_sam3_video_model(apply_temporal_disambiguation=True, device="cuda")
    else:
        sam3_model = model

    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    
    state = init_state_lazy(predictor, frame_dir)
    
    # 1. 초기 마스크 등록
    for i in range(num_objs):
        mask = detection_results[mask_key][i].cuda().float()
        if mask.dim() == 3: mask = mask.squeeze(0)
        predictor.add_new_mask(inference_state=state, frame_idx=start_frame_idx, obj_id=i+1, mask=mask)

    # 2. 정방향 추적
    # print(f"➡️ 정방향(Forward) 추적 ({start_frame_idx} -> End)")
    for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
        state, start_frame_idx=start_frame_idx, max_frame_num_to_track=None, reverse=False, propagate_preflight=True
    ):
        save_frame_json(frame_idx, obj_ids, video_res_masks, state, json_output_dir)
        
        # VRAM 관리: 과거 데이터 삭제
        if frame_idx > start_frame_idx and frame_idx % 100 == 0:
            cutoff = frame_idx - 7
            if cutoff > start_frame_idx:
                outputs = state["output_dict"]["non_cond_frame_outputs"]
                keys_to_remove = [k for k in outputs.keys() if k < cutoff]
                for k in keys_to_remove: del outputs[k]
                for obj_dict in state["output_dict_per_obj"].values():
                    outputs_obj = obj_dict["non_cond_frame_outputs"]
                    keys_to_remove_obj = [k for k in outputs_obj.keys() if k < cutoff]
                    for k in keys_to_remove_obj: del outputs_obj[k]

    # 중간 메모리 정리
    outputs = state["output_dict"]["non_cond_frame_outputs"]
    keys_to_remove = [k for k in outputs.keys() if k > start_frame_idx]
    for k in keys_to_remove: del outputs[k]
    # torch.cuda.empty_cache() # 외부에서 관리하므로 여기서는 생략 가능

    # 3. 역방향 추적
    # print(f"⬅️ 역방향(Backward) 추적 ({start_frame_idx} -> 0)")
    for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
        state, start_frame_idx=start_frame_idx, max_frame_num_to_track=None, reverse=True, propagate_preflight=True
    ):
        save_frame_json(frame_idx, obj_ids, video_res_masks, state, json_output_dir)
        
        if frame_idx < start_frame_idx and frame_idx % 100 == 0:
            cutoff = frame_idx + 7
            if cutoff < start_frame_idx:
                outputs = state["output_dict"]["non_cond_frame_outputs"]
                keys_to_remove = [k for k in outputs.keys() if k > cutoff and k < start_frame_idx]
                for k in keys_to_remove: del outputs[k]
                for obj_dict in state["output_dict_per_obj"].values():
                    outputs_obj = obj_dict["non_cond_frame_outputs"]
                    keys_to_remove_obj = [k for k in outputs_obj.keys() if k > cutoff and k < start_frame_idx]
                    for k in keys_to_remove_obj: del outputs_obj[k]

    # model은 외부에서 관리하므로 삭제하지 않음. state만 정리.
    del state