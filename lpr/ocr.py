"""Giai đoạn 2: đọc ký tự trên vùng biển số - phần dùng chung.

    preprocess_steps / split_lines / trim_to_chars : tiền xử lý OpenCV, tách dòng, cắt sát ký tự
    build_result                                   : hậu xử lý theo định dạng biển số Việt Nam
    make_ocr                                       : tạo bộ đọc PaddleOCR (lpr/paddle_ocr.py)
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .plate_format import MILITARY_FIRST, TWO_LETTER_SERIES, correct_plate, format_plate, is_valid, normalize
from .plate_type import classify, province


@dataclass
class OCRResult:
    text: str = ""          # chuỗi chuẩn hoá đã hậu xử lý, vd "51F12345"
    display: str = ""       # chuỗi hiển thị, vd "51F-123.45"
    raw: str = ""           # chuỗi OCR đọc được trước hậu xử lý
    conf: float = 0.0
    valid: bool = False     # đúng định dạng biển số VN
    lines: list = field(default_factory=list)
    color: str = ""         # màu nền biển: trắng / vàng / xanh / đỏ
    kind: str = ""          # loại biển, vd "Xe máy - cá nhân / doanh nghiệp"
    province: str = ""      # tỉnh/thành theo mã 2 số đầu


def preprocess_steps(crop):
    """Ảnh xám -> phóng to -> cân bằng sáng CLAHE -> lọc nhiễu giữ biên -> nhị phân Otsu.

    Trả về dict {tên bước: ảnh} theo thứ tự (dùng cho cả OCR và minh hoạ).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    steps = {"Ảnh xám": gray}
    if not dark_text(gray):  # biển nền xanh/đỏ chữ trắng -> đảo để luôn là chữ tối nền sáng
        gray = 255 - gray
        steps["Đảo cực (chữ sáng → tối)"] = gray
    h, w = gray.shape[:2]
    # Biển 2 dòng (gần vuông) cần cao hơn để mỗi dòng đủ rõ
    target_h = 160 if w / max(h, 1) < 2.5 else 80
    scale = target_h / max(h, 1)
    gray = cv2.resize(gray, (max(1, int(w * scale)), target_h),
                      interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    steps["Chuẩn hoá kích thước"] = gray
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    steps["CLAHE"] = gray
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    steps["Lọc bilateral"] = gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps["Nhị phân Otsu"] = binary
    return steps


def dark_text(gray):
    """True nếu chữ tối trên nền sáng. Chữ chỉ chiếm ~20-35% vùng giữa biển, nên nếu lớp tối
    theo Otsu chiếm quá nửa thì nền là màu tối (biển xanh, đỏ)."""
    h, w = gray.shape[:2]
    core = gray[int(h * 0.15):max(int(h * 0.85), 1), int(w * 0.1):max(int(w * 0.9), 1)]
    if core.size == 0:
        return True
    thr, _ = cv2.threshold(core, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (core <= thr).mean() < 0.55


def split_lines(gray):
    """Tách biển 2 dòng thành 2 ảnh theo hàng có ít điểm chữ nhất (histogram chiếu ngang).

    Biển dài (1 dòng) được giữ nguyên. Chỉ tìm đường cắt ở khoảng 30-70% chiều cao.
    """
    h, w = gray.shape[:2]
    if w / max(h, 1) >= 2.5:
        return [gray]
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Bỏ viền trái/phải để viền biển không làm nhiễu histogram
    profile = inv[:, int(w * 0.08):int(w * 0.92)].sum(axis=1).astype(float)
    profile = np.convolve(profile, np.ones(5) / 5, mode="same")
    lo, hi = int(h * 0.3), int(h * 0.7)
    cut = lo + int(np.argmin(profile[lo:hi]))
    return [gray[:cut], gray[cut:]]


def _extend_to_edge_chars(inv, x1, x2, y1, y2, cw):
    """Ký tự đầu/cuối dòng chạm viền biển (hay vùng tối ngoài biển) dính liền thành 1 khối lớn nên
    bị trim_to_chars loại như viền, mất ký tự (vd dòng '931.05' còn '931.0' -> đọc ra 30E-931.0).
    Quét tiếp ra hai bên: cột có mực MỘT PHẦN (nét chữ) thì nới khung; gặp cột tối gần kín
    (viền / nền ngoài biển) thì dừng; cho phép khe trống <= 0.5 ký tự giữa 2 ký tự; nới tối đa 1.5 ký tự."""
    w = inv.shape[1]
    frac = (inv[y1:y2] > 0).mean(axis=0)  # tỉ lệ điểm mực của từng cột trong dải chiều cao ký tự

    def scan(start, step):
        best, gap, x = start, 0, start
        limit = int(1.5 * cw)
        while 0 <= x < w and abs(x - start) <= limit:
            f = frac[x]
            if f > 0.85:      # viền biển / nền tối -> dừng
                break
            if f >= 0.1:      # nét chữ
                best, gap = x, 0
            else:
                gap += 1
                if gap > 0.5 * cw:
                    break
            x += step
        return best

    right = scan(min(x2, w - 1), 1)
    left = scan(max(x1 - 1, 0), -1)
    # Chỉ nới khi thật sự có nét chữ mới (không nới vài cột lẻ do nhiễu)
    if right - x2 >= 0.3 * cw:
        x2 = right + 1
    if x1 - left >= 0.3 * cw:
        x1 = left
    return x1, x2


def trim_to_chars(gray, extend=False):
    """Cắt sát vùng ký tự để viền biển, ốc vít không bị đọc thành ký tự thừa.

    extend=True: nới khung ra 2 bên để cứu ký tự đầu/cuối dính vào viền (_extend_to_edge_chars).
    Chỉ dùng làm lần đọc THÊM: nới cho mọi biển thì hay đọc thừa ký tự (51H30440 -> 51H300440).

    Ký tự = thành phần liên thông có chiều cao 30-95% và chiều rộng < 40% ảnh dòng,
    không chạm mép trái/phải (dải tối ở mép là viền biển/khe ốp biển khi chụp xiên,
    nếu giữ lại sẽ bị đọc thành '1').
    """
    h, w = gray.shape[:2]
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    chars = [s for s in stats[1:]
             if 0.3 * h <= s[cv2.CC_STAT_HEIGHT] <= 0.95 * h and s[cv2.CC_STAT_WIDTH] < 0.4 * w
             and s[cv2.CC_STAT_LEFT] > 0 and s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] < w]
    if len(chars) < 2:
        return gray
    x1 = min(s[0] for s in chars)
    y1 = min(s[1] for s in chars)
    x2 = max(s[0] + s[2] for s in chars)
    y2 = max(s[1] + s[3] for s in chars)
    if extend:
        x1, x2 = _extend_to_edge_chars(inv, x1, x2, y1, y2, int(np.median([s[2] for s in chars])))
    m = max(2, int(0.08 * (y2 - y1)))
    return gray[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]


def build_result(lines, conf, correct=True, color="trắng"):
    """Các dòng chữ OCR đọc được -> OCRResult đã hậu xử lý theo định dạng biển VN
    (sửa nhầm chữ/số theo vị trí, nhận dạng biển quân đội, loại biển, tỉnh)."""
    raw = normalize("".join(lines))
    if not correct:
        return OCRResult(raw, raw, raw, conf, is_valid(raw), lines, color)
    military = color == "đỏ"
    series, number = correct_plate(lines, military=military)
    if military and not is_valid(series + number, military=True):
        military = False  # nền đỏ nhưng không ra dạng quân đội -> thử như biển dân sự
        series, number = correct_plate(lines)
    if not military and not is_valid(series + number) and conf >= 0.8:
        # Nhận màu nền có thể sai (nền đỏ sẫm, ảnh tối) -> biển quân đội bị sửa như biển dân sự
        # (BH5262 -> 8HS262, sai định dạng). Chuỗi khớp dạng quân đội thì nhận là biển quân đội.
        # Không có bằng chứng màu đỏ nên đòi thêm: đọc rõ (tin cậy >= 0.8) và chữ đầu đúng khối
        # đơn vị quân đội - nếu không, chuỗi rác từ biển mờ thành "quân đội" (vd CA-21-15).
        m_series, m_number = correct_plate(lines, military=True)
        if is_valid(m_series + m_number, military=True) and m_series[:1] in MILITARY_FIRST:
            series, number, military, color = m_series, m_number, True, "đỏ"
    text = series + number
    valid = is_valid(text, military=military)
    kind = classify(series, number, len(lines), color) if valid else ""
    return OCRResult(text, format_plate(series, number), raw, conf, valid, lines, color,
                     kind, province(text) if valid else "")


def _trust(r):
    """Khoá so sánh độ tin cậy của 1 kết quả OCR (lớn hơn = tốt hơn):
    đúng định dạng > chữ seri đọc trực tiếp > ít ký tự phải sửa > độ tin cậy.

    Hậu xử lý có thể "đoán" chữ seri từ số (0->D, 4->A, 8->B...). Kết quả đọc ra ĐÚNG chữ
    ở vị trí seri đáng tin hơn kết quả phải đoán (vd biển 51M-999.88 bị đọc M -> 0 -> D).
    """
    raw, text = r.raw, r.text
    guessed_series = len(raw) > 2 and len(text) > 2 and raw[2].isdigit() and text[2].isalpha()
    # Biển 1 dòng có seri 2 chữ cái thì chỉ có các seri đặc biệt (LD, KT, NG...); "30GC-039.16" là
    # OCR đọc thừa ký tự -> coi như sai định dạng (biển 2 dòng xe máy 50cc vẫn có seri AA, AB...).
    plausible = not (len(r.lines) == 1 and len(text) > 4 and text[:2].isdigit()
                     and text[2:4].isalpha() and text[4].isdigit() and text[2:4] not in TWO_LETTER_SERIES)
    if len(raw) == len(text):
        fixes = sum(a != b for a, b in zip(raw, text))
    else:
        fixes = abs(len(raw) - len(text)) + 2
    return (r.valid and plausible, not guessed_series, -fixes, r.conf)


def _is_subsequence(short, long):
    """True nếu `long` chứa mọi ký tự của `short` theo đúng thứ tự (chỉ thêm ký tự, không đổi)."""
    it = iter(long)
    return all(ch in it for ch in short)


OCR_ENGINES = {"paddleocr": "PaddleOCR", "paddleocr_fast": "PaddleOCR (nhanh)"}


def make_ocr(engine="paddleocr"):
    """paddleocr: bản chính xác cho ảnh tĩnh; paddleocr_fast: bản nhanh cho video / webcam."""
    engine = (engine or "paddleocr").lower()
    if engine == "paddleocr":
        from .paddle_ocr import PaddlePlateOCR
        return PaddlePlateOCR()
    if engine == "paddleocr_fast":
        from .paddle_ocr import make_fast
        return make_fast()
    raise ValueError(f"Engine OCR không hỗ trợ: {engine} (chọn {', '.join(OCR_ENGINES)})")
