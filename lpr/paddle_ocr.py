"""Nhận dạng ký tự biển số bằng PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR).

Mỗi biển được đọc theo 2 cách rồi chọn kết quả đáng tin hơn (_trust: đúng định dạng > đọc trực
tiếp chữ seri > ít ký tự phải sửa > độ tin cậy):

  1. Tách dòng bằng code (_read_by_split): tách dòng theo histogram -> (cắt sát ký tự) -> nhận dạng
     từng dòng; thử thêm ảnh đã nắn thẳng (lpr/rectify.py) và đọc cả khung như 1 dòng.
  2. Tìm dòng chữ bằng mô hình phát hiện chữ của PaddleOCR (_det_lines): mỗi dòng được nắn thẳng
     theo đa giác mô hình tìm ra rồi mới nhận dạng -> không phụ thuộc tỉ lệ khung cắt
     (biển 1 dòng chụp nghiêng không còn bị tách nhầm thành 2 dòng).

Đo trên 98 biển có đáp án (data/ocr_labels, gán nhãn tay ở /label):
    chỉ cách 1: 73/98   chỉ cách 2: 72/98   chọn tốt hơn giữa 1 và 2: 83/98
    unclip_ratio 2.0 (nới khung dòng nhiều hơn) chỉ cách 2: 60/98; thêm viền trắng + phóng to: 52/98
    -> giữ unclip_ratio = 1.5 (mặc định). Luôn chạy cả 2 cách: kể cả khi cách 1 rất tự tin
    (độ tin cậy >= 0.9) vẫn có biển sai mà cách 2 sửa được.

Cỡ mô hình (mobile = nhỏ, nhanh; server = lớn ~10 lần, chính xác hơn, chậm trên CPU), cùng 98 biển:
    nhận dạng mobile + phát hiện dòng mobile : 83/98   0.4 s/biển   -> "paddleocr_fast" (video, webcam)
    nhận dạng server + phát hiện dòng mobile : 86/98   5.7 s/biển   (quá chậm)
    nhận dạng mobile + phát hiện dòng server : 87/98   1.9 s/biển   -> "paddleocr" (ảnh tĩnh)

Cài đặt (đã cài trong .venv):
    pip install paddlepaddle paddleocr
    PaddleX đòi gói opencv-contrib-python -> cài ĐÚNG phiên bản OpenCV đang dùng
    (vd opencv-contrib-python==5.0.0.93), đừng để pip kéo bản 4.10 về đè lên cv2.

Trọng số (nhận dạng mobile ~8 MB, phát hiện dòng mobile ~5 MB / server ~85 MB) tự tải về
~/.paddlex lần đầu chạy.
"""
import os
from pathlib import Path

import cv2
import numpy as np

from .ocr import OCRResult, _is_subsequence, _trust, build_result, preprocess_steps, split_lines, trim_to_chars
from .plate_type import plate_color

# Mô hình tiếng Anh: chỉ chữ Latin + số, khớp với ký tự trên biển số VN
DEFAULT_MODEL = "en_PP-OCRv5_mobile_rec"
# Mô hình nhận dạng đã tinh chỉnh trên biển số VN (notebooks/finetune_paddleocr_colab.ipynb): có thư mục
# này (inference.json + inference.pdiparams + inference.yml) thì tự dùng thay cho mô hình gốc.
FINETUNED_DIR = Path(__file__).resolve().parent.parent / "weights" / "paddle_rec"
DET_MODEL = "PP-OCRv5_server_det"        # ảnh tĩnh: chính xác hơn
FAST_DET_MODEL = "PP-OCRv5_mobile_det"   # video / webcam: nhanh


def finetuned_available():
    return (FINETUNED_DIR / "inference.pdiparams").exists()


def _warp_quad(img, poly):
    """Nắn vùng chữ hình tứ giác (4 đỉnh theo chiều kim đồng hồ từ góc trên trái) thành chữ nhật."""
    p = np.asarray(poly, dtype=np.float32)
    w = int(max(np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[2] - p[3])))
    h = int(max(np.linalg.norm(p[0] - p[3]), np.linalg.norm(p[1] - p[2])))
    if w < 2 or h < 2:
        return None
    M = cv2.getPerspectiveTransform(p, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class PaddlePlateOCR:
    name = "paddleocr"
    _result = staticmethod(build_result)

    def __init__(self, model=DEFAULT_MODEL, line_det=True, det_model=DET_MODEL, share_rec=None):
        """share_rec: PaddlePlateOCR khác để dùng chung mô hình nhận dạng (đỡ tải 2 lần)."""
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")  # khỏi dò mạng mỗi lần
        from paddleocr import TextDetection, TextRecognition
        if share_rec is not None:
            self.rec = share_rec.rec
        elif model == DEFAULT_MODEL and finetuned_available():
            self.rec = TextRecognition(model_name=DEFAULT_MODEL, model_dir=str(FINETUNED_DIR))
        else:
            self.rec = TextRecognition(model_name=model)
        # enable_mkldnn=False: Paddle 3.3 trên CPU chạy mô hình det với oneDNN bị lỗi
        # "ConvertPirAttribute2RuntimeAttribute not support" -> tắt oneDNN riêng cho mô hình này.
        self.det = TextDetection(model_name=det_model, enable_mkldnn=False) if line_det else None

    def _predict(self, img):
        """Nhận dạng ảnh chứa đúng 1 dòng chữ. Trả về (chuỗi in hoa, độ tin cậy 0-1)."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for r in self.rec.predict(img):
            return str(r["rec_text"]).upper(), float(r["rec_score"])
        return "", 0.0

    # ---------- Cách 1: tách dòng bằng code ----------
    def _read_lines(self, crop, trim, single=False):
        """trim: False | True (cắt sát ký tự) | "ext" (cắt sát + nới cứu ký tự ở mép).
        single=True: đọc cả khung như 1 dòng, không tách dòng."""
        gray = preprocess_steps(crop)["Lọc bilateral"]
        lines, confs = [], []
        for part in ([gray] if single else split_lines(gray)):
            text, conf = self._predict(trim_to_chars(part, extend=trim == "ext") if trim else part)
            if text.strip():
                lines.append(text)
                confs.append(conf)
        return lines, (float(np.mean(confs)) if confs else 0.0)

    def _rescue_edge_chars(self, res, img, single, correct, color):
        """Cắt sát ký tự đôi khi bỏ mất ký tự đầu/cuối dính vào viền biển, mà chuỗi thiếu vẫn đúng
        định dạng (30E-931.05 -> 30E-9310, kiểu biển cũ 4 số). Đọc lại với khung nới ra 2 bên; chỉ
        nhận khi chuỗi mới hợp lệ, DÀI HƠN và chứa đủ mọi ký tự cũ theo đúng thứ tự (chỉ thêm, không
        đổi ký tự) - nới khung cho mọi biển thì hay đọc thừa / đọc sai ký tự."""
        ext = self._result(*self._read_lines(img, "ext", single), correct, color)
        if ext.valid and len(ext.text) > len(res.text) and _is_subsequence(res.text, ext.text):
            return ext
        return res

    def _read_by_split(self, crop, correct=True):
        """Thử lần lượt (dừng sớm khi đã chắc chắn, mỗi lần đọc ~0.05-0.2 s trên CPU):
            ảnh gốc cắt sát ký tự -> ảnh gốc giữ nguyên dòng -> ảnh nắn thẳng (lpr/rectify.py)."""
        from .rectify import rectify_plate
        color = plate_color(crop)
        flat = rectify_plate(crop)
        attempts = [(crop, True, False), (crop, False, False)]
        # Biển 1 dòng chụp nghiêng / gần có khung cắt khá vuông (rộng/cao ~1.8-2.5) nên split_lines
        # tưởng là biển 2 dòng và cắt ngang qua chữ (vd 51F-388.88 -> "51F38888" + "207750000").
        # Đọc cả khung như 1 dòng TRƯỚC: nửa trên + nửa dưới chữ đôi khi ghép tình cờ thành chuỗi
        # "đúng định dạng" xe máy (30E-931.05 -> 30E9-310.05, 30F93-3105), nên không thể đợi
        # tách dòng thất bại rồi mới thử. Biển 2 dòng thật mà đọc như 1 dòng thì ra rác, sai định dạng.
        if 1.8 <= crop.shape[1] / max(crop.shape[0], 1) < 2.5:
            attempts[:0] = [(crop, True, True), (crop, False, True)]  # cắt sát ký tự trước: bỏ viền biển
        if flat is not None:
            attempts += [(flat, False, False), (flat, True, False)]
        best = None
        for img, trim, single in attempts:
            res = self._result(*self._read_lines(img, trim, single), correct, color)
            if trim is True and res.valid:
                res = self._rescue_edge_chars(res, img, single, correct, color)
            if single and _trust(res)[:2] == (True, True) and res.conf >= 0.8:
                return res  # đọc 1 dòng đã ra biển hợp lệ -> không để kết quả tách dòng ghi đè
            if best is None or _trust(res) > _trust(best):
                best = res
            t = _trust(best)
            if t[0] and t[1] and t[2] >= -1 and best.conf >= 0.8:  # đủ chắc chắn -> dừng
                break
        return best

    # ---------- Cách 2: mô hình phát hiện dòng chữ ----------
    @staticmethod
    def _same_line(p, group):
        """Vùng chữ p có cùng dòng với nhóm vùng chữ `group` không.

        - Tâm lệch dọc < nửa chiều cao vùng THẤP hơn. (Dùng vùng cao hơn thì biển 2 dòng nghiêng
          mạnh, khung mỗi dòng cao, bị gộp 2 dòng làm 1 rồi xếp trái -> phải: 19A / 388.84 đọc thành
          38B84-194 - 6/65 góc xoay/xiên thử nghiệm bị gộp sai.)
        - Hoặc: mảnh nhỏ (mô hình tách rời 1 ký tự, vd "2" của 299.68) có tâm nằm trong chiều cao của
          vùng lớn và KHÔNG chồng theo chiều ngang với nó. Hai dòng của biển 2 dòng thì chồng ngang
          (dòng này nằm trên dòng kia) nên không bị gộp.
        """
        cy, h = p[:, 1].mean(), np.ptp(p[:, 1])
        x1, x2 = p[:, 0].min(), p[:, 0].max()
        for q in group:
            qy, qh = q[:, 1].mean(), np.ptp(q[:, 1])
            if abs(cy - qy) < 0.5 * min(h, qh):
                return True
            small_cy, big = (cy, q) if h < qh else (qy, p)
            sx1, sx2 = (x1, x2) if h < qh else (q[:, 0].min(), q[:, 0].max())
            bx1, bx2 = big[:, 0].min(), big[:, 0].max()
            overlap = max(0.0, min(sx2, bx2) - max(sx1, bx1))
            if (big[:, 1].min() <= small_cy <= big[:, 1].max()
                    and overlap < 0.2 * (sx2 - sx1)):
                return True
        return False

    def line_regions(self, crop):
        """Các dòng chữ trên biển (trên -> dưới); mỗi dòng là list vùng chữ (đa giác 4 đỉnh) trái -> phải.
        Dùng cho cách đọc 2 và để cắt ảnh từng dòng làm dữ liệu tinh chỉnh (scripts/export_paddle_rec.py)."""
        H = crop.shape[0]
        polys = []
        for r in self.det.predict(crop, unclip_ratio=1.5):
            for poly in r["dt_polys"]:
                poly = np.asarray(poly)
                if np.ptp(poly[:, 1]) >= 0.15 * H:  # bỏ chấm, ốc vít, nhiễu nhỏ
                    polys.append(poly)
        polys.sort(key=lambda p: p[:, 1].mean())
        groups = []  # gom các vùng chữ cùng 1 dòng (trên -> dưới)
        for p in polys:
            if groups and self._same_line(p, groups[-1]):
                groups[-1].append(p)
            else:
                groups.append([p])
        return [sorted(ps, key=lambda p: p[:, 0].min()) for ps in groups]

    def _det_lines(self, crop):
        """Mô hình phát hiện chữ tìm từng dòng -> nắn thẳng -> nhận dạng. Trả về (dòng, tin cậy)."""
        lines, confs = [], []
        for ps in self.line_regions(crop):
            for p in ps:
                img = _warp_quad(crop, p)
                if img is None:
                    continue
                text, conf = self._predict(img)
                if text.strip():
                    lines.append(text)
                    confs.append(conf)
        return lines, (float(np.mean(confs)) if confs else 0.0)

    def read(self, crop, correct=True):
        if crop is None or crop.size == 0:
            return OCRResult()
        best = self._read_by_split(crop, correct)  # cách 1
        if self.det is None:
            return best
        res = self._result(*self._det_lines(crop), correct, plate_color(crop))  # cách 2
        # Cả 2 hợp lệ mà 1 chuỗi chỉ là chuỗi kia THIẾU ký tự (vùng chữ bị cắt mất ký tự ở mép:
        # 59U1-299.68 -> 59U1-9968, vẫn đúng dạng biển 4 số) -> lấy chuỗi đầy đủ hơn.
        if res.valid and best.valid and len(res.text) != len(best.text):
            short, long_ = sorted((res, best), key=lambda r: len(r.text))
            if _is_subsequence(short.text, long_.text):
                return long_
        return res if _trust(res) > _trust(best) else best  # hoà điểm -> giữ cách 1


def make_fast(accurate=None):
    """Bản nhanh cho video/webcam: phát hiện dòng bản mobile. Truyền bản chính xác để dùng chung
    mô hình nhận dạng."""
    return PaddlePlateOCR(det_model=FAST_DET_MODEL, share_rec=accurate)
