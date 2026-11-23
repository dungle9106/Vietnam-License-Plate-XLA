"""Sinh ảnh biển số giả lập (vẽ bằng OpenCV) để kiểm thử mà không cần dataset."""
import cv2
import numpy as np


# Màu nền / màu chữ (BGR) theo loại biển
PLATE_COLORS = {
    "trắng": ((235, 235, 235), (20, 20, 20)),
    "vàng": ((40, 200, 240), (20, 20, 20)),
    "xanh": ((160, 80, 20), (245, 245, 245)),
    "đỏ": ((40, 30, 200), (245, 245, 245)),
}


def make_plate(lines, height=110, color="trắng"):
    """Biển số giả lập, viền cùng màu chữ. 1 dòng: tỉ lệ ~4.5; 2 dòng: tỉ lệ ~1.4."""
    bg, fg = PLATE_COLORS[color]
    w = int(height * (4.5 if len(lines) == 1 else 1.4))
    img = np.full((height, w, 3), bg, np.uint8)
    cv2.rectangle(img, (2, 2), (w - 3, height - 3), fg, max(2, height // 40))
    row_h = height / len(lines)
    for i, text in enumerate(lines):
        scale, thick = row_h / 45, max(2, int(row_h / 18))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        if tw > 0.9 * w:  # thu nhỏ cho vừa chiều rộng
            scale *= 0.9 * w / tw
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        y = int(row_h * i + (row_h + th) / 2)
        cv2.putText(img, text, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    fg, thick, cv2.LINE_AA)
    return img
