# ... (앞부분 import 및 설정 코드 동일) ...

# ==========================================
# 2. 10개 비디오 처리 루프 (0 ~ 9)
# ==========================================
TARGET_INDICES = range(10)

for target in TARGET_INDICES:
    print(f"\n{'='*50}")
    print(f"🎬 Target {target} 처리 시작")
    print(f"{'='*50}")

    # --- [A. 경로 설정 및 BBox 로드] ---
    # (기존 코드와 동일)
    COMMON_PATH = df.loc[target, "common_path"]
    VIDEO_PTH = Path(df.loc[target, "video_path"])
    
    KPT_DIR = DATA_DIR / "2_KEYPOINTS" / COMMON_PATH
    JSON_PATH = KPT_DIR / "000000.json" 

    SAVE_ROOT = DATA_DIR / "test" / COMMON_PATH
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    
    OUTPUT_JSON = SAVE_ROOT / "tracking_results.json"
    OUTPUT_MP4 = SAVE_ROOT / f"{COMMON_PATH}_result.mp4"
    TIME_LOG = SAVE_ROOT / "time_log.txt"
    
    # --------------------------------------
    # B. 비디오 정보 읽기 및 BBox 정규화
    # --------------------------------------
    cap = cv2.VideoCapture(str(VIDEO_PTH))
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다: {VIDEO_PTH}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    TGT_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    TGT_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # BBox 로드 및 정규화 (SAM 입력용 0.0~1.0 비율)
    norm_box_input = None
    if JSON_PATH.exists():
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
        
        raw_bbox = data["instance_info"][0]["bbox"][0]
        # [핵심] 정규화 (0.0~1.0 비율로 변환)
        norm_box = [
            raw_bbox[0] / SRC_W,
            raw_bbox[1] / SRC_H,
            raw_bbox[2] / SRC_W,
            raw_bbox[3] / SRC_H
        ]
        # SAM 입력용 TENSOR 형태 (Batch 차원 추가)
        norm_box_input = torch.tensor(norm_box, dtype=torch.float32).unsqueeze(0).to(device)
        
        print(f"📦 BBox 정규화 완료 (0~1): {norm_box}")
    else:
        print(f"❌ JSON 파일이 없습니다: {JSON_PATH}")
        cap.release()
        continue 
    
    # --------------------------------------
    # C. 모델 초기화 및 프롬프트 주입 (Chunking 루프 제거)
    # --------------------------------------
    start_time = time.time()
    
    # 1. 모델 상태 초기화 (비디오 전체 로드, CPU Offload 사용)
    predictor.image_size = 1024 # 안전한 1024 해상도 복구
    inference_state = predictor.init_state(
        video_path=str(VIDEO_PTH),
        offload_video_to_cpu=True,   # 긴 영상에 필수
        offload_state_to_cpu=True,   # 긴 영상에 필수
        async_loading_frames=True
    )

    # 2. [핵심] 첫 프레임에 BBox 프롬프트 주입
    # 이 과정은 전체 추적 전에 단 한 번만 수행되어야 합니다.
    _, _, _, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1, # 객체 ID 1
        box=norm_box_input, 
        points=None,
        labels=None
    )
    print("✅ 초기 프롬프트(BBox) 주입 완료.")

    # --------------------------------------
    # D. 추적 및 저장 루프
    # --------------------------------------
    json_results = {}
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, fps, (TGT_W, TGT_H))
    
    # 1. 추적 시작 (프레임 0부터 끝까지)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0, # 0번부터 시작
        reverse=False,
        propagate_preflight=True
    ):
        
        # 2. JSON 데이터 수집
        json_results[out_frame_idx] = {
            "obj_ids": [int(id) for id in out_obj_ids],
            # 마스크의 바운딩 박스 정보 등을 여기에 추가할 수 있습니다.
        }

        # 3. 영상 처리 및 저장
        cap.set(cv2.CAP_PROP_POS_FRAMES, out_frame_idx) # cap 위치를 현재 프레임으로 이동
        ret, frame = cap.read()
        
        if not ret: 
            print(f"⚠️ 프레임 {out_frame_idx}를 읽을 수 없습니다. 중단.")
            break
        
        if len(out_mask_logits) > 0:
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8).squeeze()
            
            # 마스크 리사이즈 (SAM 출력을 원본 해상도에 맞춤)
            if mask.shape[:2] != (TGT_H, TGT_W):
                mask = cv2.resize(mask, (TGT_W, TGT_H), interpolation=cv2.INTER_NEAREST)
            
            # 마스크 합성 (녹색 오버레이)
            colored_mask = np.zeros_like(frame)
            colored_mask[mask == 1] = MASK_COLOR
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

        out_writer.write(frame)
    
    # --------------------------------------
    # E. 마무리 및 메모리 정리
    # --------------------------------------
    cap.release()
    out_writer.release()
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_results, f, indent=4)

    elapsed_time = time.time() - start_time
    
    with open(TIME_LOG, 'w') as f:
        f.write(f"Target: {target}\nProcessing Time: {elapsed_time:.2f}s\n")

    print(f"✅ Target {target} 완료! (시간: {elapsed_time:.2f}초)")
    
    # GPU 메모리 정리
    del inference_state
    gc.collect()
    torch.cuda.empty_cache()

print("\n🎉 모든 작업 종료!")