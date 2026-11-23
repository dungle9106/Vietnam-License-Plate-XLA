"""Giai đoạn 1: phát hiện vùng biển số bằng YOLOv8.

Kích thước ảnh đưa vào YOLO (imgsz): YOLO thu nhỏ mọi ảnh về imgsz trước khi tìm, nên biển trong
ảnh lớn bị co lại quá nhỏ. Đo trên 74 ảnh gốc data/lp/images/test (99 biển, ngưỡng 0.4):
    imgsz 640 : tìm thấy 92/99 (92.9%)  ~0.15 s/ảnh CPU   <- video / webcam (FAST_IMGSZ)
    imgsz 960 : tìm thấy 96/99 (97.0%)  ~0.33 s/ảnh       <- ảnh tĩnh (mặc định)
    imgsz 1280: tìm thấy 97/99 (98.0%)  ~0.59 s/ảnh, số vùng ngoài nhãn tăng gấp đôi
Biển bị sót ở 640 mà 960 tìm được: biển cao ~46-48 px trong ảnh 1536-2048 px (co còn ~15-20 px).
"""
import copy

from ultralytics import YOLO

DEFAULT_IMGSZ = 960
FAST_IMGSZ = 640


class PlateDetector:
    def __init__(self, weights, conf=0.4, iou=0.5, imgsz=DEFAULT_IMGSZ, device=None):
        self.model = YOLO(str(weights))
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device

    def detect(self, img):
        """Trả về list (x1, y1, x2, y2, conf) theo toạ độ pixel."""
        res = self.model.predict(img, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        return [(*map(float, b), float(c)) for b, c in zip(xyxy, confs)]

    def with_imgsz(self, imgsz):
        """Bản sao dùng CHUNG mô hình đã tải, chỉ khác kích thước ảnh (vd video dùng 640 cho nhanh)."""
        other = copy.copy(self)
        other.imgsz = imgsz
        return other
