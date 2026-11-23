"""Thư viện nhận diện biển số xe Việt Nam: YOLOv8s (phát hiện biển) + PaddleOCR (đọc ký tự)."""
import sys

# Trên Windows, nạp PaddlePaddle TRƯỚC PyTorch thì PyTorch báo lỗi DLL (WinError 127, torch\lib\shm.dll);
# nạp PyTorch trước thì cả hai chạy bình thường -> luôn nạp torch ngay khi dùng thư viện lpr.
try:
    import torch  # noqa: F401
except ImportError:
    pass

# Console Windows mặc định dùng cp1252, không in được tiếng Việt
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
