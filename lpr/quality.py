"""Đánh giá chất lượng ảnh vùng biển số: biển quá nhỏ hoặc quá mờ thì kết quả OCR kém tin cậy.

Đo trên 98 biển có đáp án (data/ocr_labels), PaddleOCR bản chính xác:
    biển cao >= 60 px: đọc đúng 67/70 (95.7%)   |  cao < 60 px: 20/28 (71%)
    độ nét  >= 300   : đọc đúng 69/73 (94.5%)   |  độ nét < 300: 18/25 (72%)
    biển đọc sai: cao trung vị 40 px, độ nét 129; biển đọc đúng: 132 px, độ nét 1720.
Độ nét = phương sai Laplacian của ảnh xám sau khi đưa về cao 100 px (so được giữa các cỡ biển).
"""
import cv2

MIN_HEIGHT = 60       # px trên ảnh gốc
MIN_SHARPNESS = 300


def plate_quality(crop):
    """Trả về dict: cao (px), do_net, kem (True = nên nghi ngờ kết quả), ly_do (chuỗi tiếng Việt)."""
    if crop is None or crop.size == 0:
        return {"cao": 0, "do_net": 0.0, "kem": True, "ly_do": "không có ảnh biển"}
    h = crop.shape[0]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    s = 100 / max(h, 1)
    gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if h < MIN_HEIGHT:
        reasons.append(f"biển nhỏ ({h} px, nên ≥ {MIN_HEIGHT} px)")
    if sharp < MIN_SHARPNESS:
        reasons.append("ảnh biển mờ")
    return {"cao": int(h), "do_net": round(sharp, 1), "kem": bool(reasons), "ly_do": ", ".join(reasons)}
