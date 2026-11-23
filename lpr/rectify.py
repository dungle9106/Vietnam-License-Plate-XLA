"""Nắn thẳng biển số bị chụp xiên / nghiêng trước khi OCR (biến đổi phối cảnh).

Vùng YOLO cắt ra là hình chữ nhật song song trục ảnh, nhưng biển chụp xiên thì thành hình
thang/bình hành -> ký tự bị nghiêng, méo, OCR dễ đọc sai. Các bước:
    1. Tìm vùng nền biển: thành phần sáng lớn nhất sau ngưỡng Otsu (biển nền tối thì đảo ảnh).
    2. Xấp xỉ đường bao thành tứ giác (approxPolyDP; không ra 4 đỉnh thì dùng minAreaRect).
    3. Biến đổi phối cảnh tứ giác đó về hình chữ nhật đúng tỉ lệ, rồi phóng to cho đủ nét.
Không tìm được tứ giác hợp lý -> trả về None, nơi gọi dùng ảnh gốc.
"""
import cv2
import numpy as np


def order_corners(pts):
    """4 điểm -> thứ tự (trên-trái, trên-phải, dưới-phải, dưới-trái)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def find_plate_quad(crop):
    """Tìm tứ giác nền biển trong vùng cắt. Trả về 4 góc (đã sắp xếp) hoặc None."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    if h < 12 or w < 24:
        return None
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Nền biển chiếm phần giữa vùng cắt; nếu giữa vùng cắt phần lớn là tối -> biển nền tối
    core = bw[h // 4:3 * h // 4, w // 4:3 * w // 4]
    if core.size and (core > 0).mean() < 0.5:
        bw = 255 - bw
    # Đóng hình thái để chữ (lỗ tối trong nền sáng) không cắt rời nền biển
    k = max(3, (min(h, w) // 8) | 1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.3 * h * w:  # quá nhỏ -> không phải nền biển
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
    quad = order_corners(quad)
    if cv2.contourArea(quad) < 0.3 * h * w:
        return None
    return quad


def rectify_plate(crop, min_h=100):
    """Nắn biển về hình chữ nhật thẳng. Trả về ảnh mới, hoặc None nếu không nắn được
    hoặc biển vốn đã gần như thẳng (khỏi tốn thời gian OCR lại)."""
    if crop is None or crop.size == 0:
        return None
    quad = find_plate_quad(crop)
    if quad is None:
        return None
    tl, tr, br, bl = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if width < 10 or height < 10 or not 0.8 <= width / height <= 6.5:
        return None
    # Độ lệch của tứ giác so với khung chữ nhật song song trục: nhỏ thì không cần nắn
    h, w = crop.shape[:2]
    box = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    tilt = max(abs(tl[1] - tr[1]), abs(bl[1] - br[1]), abs(tl[0] - bl[0]), abs(tr[0] - br[0]))
    margin = np.abs(quad - box).max()
    if tilt < 0.03 * max(width, height) and margin < 0.08 * max(w, h):
        return None
    scale = max(1.0, min_h / height)
    W, H = int(round(width * scale)), int(round(height * scale))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(crop, M, (W, H), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
