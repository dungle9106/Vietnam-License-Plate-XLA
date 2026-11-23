"""Đọc/ghi ảnh an toàn với đường dẫn có dấu tiếng Việt trên Windows.

cv2.imread/cv2.imwrite không đọc được đường dẫn Unicode trên Windows,
nên dùng np.fromfile + cv2.imdecode thay thế.
"""
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


def is_image(path):
    return Path(path).suffix.lower() in IMG_EXTS


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


# Mô hình YOLOv8s phát hiện biển số (tự huấn luyện, notebooks/train_colab.ipynb)
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"
