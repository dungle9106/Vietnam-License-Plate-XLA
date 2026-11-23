"""Pipeline hoàn chỉnh: ảnh -> phát hiện biển số -> cắt vùng -> OCR -> kết quả."""
import time
from dataclasses import dataclass, field

import cv2

from .detector import DEFAULT_IMGSZ, PlateDetector
from .ocr import make_ocr
from .plate_format import is_valid
from .tracker import IoUTracker


@dataclass
class PlateResult:
    box: tuple              # (x1, y1, x2, y2)
    det_conf: float
    text: str               # chuỗi chuẩn hoá, vd "51F12345"
    display: str            # chuỗi hiển thị, vd "51F-123.45"
    ocr_conf: float
    valid: bool = False
    track_id: int = None
    color: str = ""         # màu nền biển
    kind: str = ""          # loại biển
    province: str = ""      # tỉnh/thành đăng ký
    raw: str = ""           # chuỗi OCR trước hậu xử lý
    crop: object = field(default=None, repr=False)  # ảnh vùng biển (BGR), dùng khi xuất Excel

    def to_dict(self):
        return {"bien_so": self.display, "chuoi": self.text, "hop_le": self.valid,
                "loai": self.kind, "mau": self.color, "tinh": self.province, "ocr_tho": self.raw,
                "conf_phat_hien": round(float(self.det_conf), 4),
                "conf_ocr": round(float(self.ocr_conf), 4),
                "box": [int(round(v)) for v in self.box]}


def crop_plate(img, box, pad=0.12):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return img[y1:y2, x1:x2]


def draw_results(img, results, fps=None):
    """Vẽ khung (xanh lá: đúng định dạng, cam: chưa chắc chắn) kèm chuỗi biển số."""
    out = img.copy()
    thick = max(2, round(min(out.shape[:2]) / 400))
    for r in results:
        x1, y1, x2, y2 = map(int, r.box)
        color = (0, 200, 0) if r.valid else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = r.display or "?"
        scale = thick * 0.4
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 10
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return out


class LPRPipeline:
    """YOLOv8s (weights) phát hiện biển số + PaddleOCR đọc ký tự.

    ocr_engine: 'paddleocr' (chính xác, ảnh tĩnh) | 'paddleocr_fast' (nhanh, video).
    Truyền `ocr` có sẵn để khỏi tải lại mô hình OCR.
    """

    def __init__(self, weights, det_conf=0.4, imgsz=None, device=None, ocr=None, ocr_engine="paddleocr"):
        self.detector = PlateDetector(weights, conf=det_conf, imgsz=imgsz or DEFAULT_IMGSZ, device=device)
        self.ocr = ocr or make_ocr(ocr_engine)

    @classmethod
    def from_parts(cls, detector, ocr):
        """Ghép bộ phát hiện + OCR đã tải sẵn (web app cache model giữa các request)."""
        self = cls.__new__(cls)
        self.detector, self.ocr = detector, ocr
        return self

    def process_image(self, img, multi=True, full=None):
        """Trả về danh sách biển số trong ảnh (trái -> phải, trên -> dưới).

        multi=False: chỉ giữ 1 biển tốt nhất (đúng định dạng trước, rồi tới điểm phát hiện).
        full: ảnh gốc độ phân giải cao mà `img` được thu nhỏ từ đó. Phát hiện chạy trên `img`
        (nhanh), còn vùng biển để OCR được cắt từ `full` cho rõ nét - ảnh điện thoại 4000 px
        thu về 1920 px thì biển số chỉ còn ~130 px, OCR dễ đọc nhầm 3->2, 1->4.
        Toạ độ box trả về vẫn theo `img`.
        """
        s = full.shape[1] / img.shape[1] if full is not None else 1.0
        results = []
        for det in self.detector.detect(img):
            crop = crop_plate(full, [v * s for v in det[:4]]) if s > 1 else crop_plate(img, det)
            o = self.ocr.read(crop)
            results.append(PlateResult(det[:4], det[4], o.text, o.display, o.conf, o.valid,
                                       color=o.color, kind=o.kind, province=o.province,
                                       raw=o.raw, crop=crop))
        if not multi and results:
            results = [max(results, key=lambda r: (r.valid, r.det_conf * max(r.ocr_conf, 0.1)))]
        results.sort(key=lambda r: (r.box[0], r.box[1]))  # trái -> phải
        return results


class VideoProcessor:
    """Xử lý từng frame video/webcam với tracking + bỏ phiếu OCR.

    OCR chậm hơn detect nhiều, nên mỗi biển số chỉ OCR mỗi `ocr_every` frame
    và dừng khi đã đọc đủ `max_reads` lần.
    """

    def __init__(self, pipeline, ocr_every=3, max_reads=8):
        self.p, self.ocr_every, self.max_reads = pipeline, ocr_every, max_reads
        self.tracker = IoUTracker()
        self._t = None
        self.fps = 0.0

    def process_frame(self, frame):
        dets = self.p.detector.detect(frame)
        ids = self.tracker.update([d[:4] for d in dets])
        results = []
        for det, tid in zip(dets, ids):
            tr = self.tracker.tracks[tid]
            if tr.ocr_count < self.max_reads and (tr.hits - 1) % self.ocr_every == 0:
                tr.add_vote(self.p.ocr.read(crop_plate(frame, det)))
            text, display, share = tr.best()
            o = tr.info.get(text)
            results.append(PlateResult(det[:4], det[4], text, display, share,
                                       o.valid if o else is_valid(text), tid,
                                       color=o.color if o else "", kind=o.kind if o else "",
                                       province=o.province if o else "", raw=o.raw if o else ""))
        now = time.perf_counter()
        if self._t is not None:
            inst = 1.0 / max(now - self._t, 1e-6)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._t = now
        return results
