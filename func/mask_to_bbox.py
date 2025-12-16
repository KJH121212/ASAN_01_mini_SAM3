# -----------------------------------------------------------------------------
# 1. 함수 정의 (Helper Functions)
# -----------------------------------------------------------------------------
def mask_to_bbox(mask):
    """Convert a binary mask to bounding box [x_min, y_min, x_max, y_max]."""
    # 💡 [수정] 차원이 3차원(1, H, W)이면 2차원(H, W)으로 압축
    if mask.ndim == 3:
        mask = mask.squeeze()
        
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]  # No mask found
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    # JSON 저장을 위해 numpy int를 python int로 변환
    return [int(x_min), int(y_min), int(x_max), int(y_max)]