"""Hậu xử lý chuỗi ký tự biển số Việt Nam.

Cấu trúc biển số: [2 số mã tỉnh][1 chữ cái seri][0-1 chữ/số][4-5 số thứ tự]
    Ô tô 1 dòng : 51F-123.45, 30A-1234, 51LD-123.45
    Xe máy 2 dòng: 59-X1 / 123.45, 29-B1 / 1234
    Quân đội     : TM-12-34 (2 chữ + 4 số, nền đỏ)
    Ngoại giao   : 80-NG-123.45, dạng cũ 41-291-NN-01
OCR hay nhầm chữ <-> số (O/0, B/8, S/5...), nên ta sửa theo vị trí:
vị trí mã tỉnh và dãy số cuối phải là số, vị trí seri đầu tiên phải là chữ.
"""
import re

# Chữ cái dùng trên biển dân sự VN: không có I, J, O, Q, W (dễ nhầm với số 1, 0 / không có trong
# tiếng Việt). Seri chứa các chữ này là OCR đọc nhầm (vd vết bẩn cạnh "19A" -> "19AW").
_L = "A-HK-NPR-VX-Z"
# Seri: 1 chữ (51F), chữ + số/chữ (59X1, 51LD), hoặc xe máy điện MĐ + số (41MD1)
PLATE_RE = re.compile(r"^\d{2}(MD\d|[" + _L + r"][" + _L + r"0-9]?)\d{4,5}$")
DIPLO_OLD_RE = re.compile(r"^\d{5}(NG|QT|CV|NN)\d{2,3}$")
MILITARY_RE = re.compile(r"^[A-Z]{2}\d{4}$")
# Chữ đầu biển quân đội = khối đơn vị: A (quân đoàn), B (bộ tư lệnh, binh chủng), D (đơn vị kinh tế),
# H (học viện), K (quân khu), P (cơ quan đặc biệt), Q (quân chủng), T (tổng cục), V (viện, bệnh viện)
MILITARY_FIRST = set("ABDHKPQTV")
DIPLO_CODES = ("NG", "QT", "CV", "NN")

TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "J": "1",
            "T": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}
TO_LETTER = {"0": "D", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}


def normalize(text):
    """Chỉ giữ chữ in hoa và số (bỏ '-', '.', khoảng trắng)."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid(text, military=False):
    """Đúng định dạng biển VN. Dạng quân đội (2 chữ + 4 số) dễ trùng chuỗi rác, nên chỉ
    chấp nhận khi biết trước là biển quân đội (nền đỏ)."""
    t = normalize(text)
    if military:
        return bool(MILITARY_RE.match(t))
    return bool(PLATE_RE.match(t) or DIPLO_OLD_RE.match(t))


def _fix_series(s):
    c = list(s)
    for i in (0, 1):
        if i < len(c):
            c[i] = TO_DIGIT.get(c[i], c[i])
    if len(c) > 2:
        c[2] = TO_LETTER.get(c[2], c[2])
    # Xe máy điện 41-MĐ1: OCR hay đọc M -> N, Đ -> 0. Không có seri "ND"/"N0"/"M0" + số.
    if len(c) == 5 and c[2] in "MN" and c[3] in "D0" and c[4].isdigit():
        c[2:4] = ["M", "D"]
    return "".join(c)


def _fix_number(s):
    return "".join(TO_DIGIT.get(ch, ch) for ch in s)


def _to_letters(s):
    return "".join(TO_LETTER.get(ch, ch) for ch in s)


def correct_plate(lines, military=False):
    """Nhận danh sách các dòng OCR (trên -> dưới), trả về (seri, dãy số).

    military=True (biển nền đỏ): 2 ký tự đầu là chữ ký hiệu đơn vị, còn lại là số.
    """
    lines = [normalize(l) for l in lines]
    lines = [l for l in lines if l]
    if not lines:
        return "", ""
    joined = "".join(lines)
    if military:
        return _to_letters(joined[:2]), _fix_number(joined[2:])
    # Biển ngoại giao / người nước ngoài dạng cũ: 41-291-NN-01
    if 9 <= len(joined) <= 10 and joined[5:7] in DIPLO_CODES:
        return _fix_number(joined[:5]) + joined[5:7], _fix_number(joined[7:])
    if len(lines) >= 2:  # biển 2 dòng: dòng 1 là seri, phần còn lại là dãy số
        return _fix_series(lines[0]), _fix_number("".join(lines[1:]))
    t = lines[0]
    if len(t) < 7:
        return _fix_series(t[:3]), t[3:]
    # Biển 1 dòng: 3 ký tự đầu là seri, ký tự thứ 4 (nếu thuộc seri) có thể là
    # chữ hoặc số nên giữ nguyên, phần còn lại là dãy số.
    split = _series_len(t)
    return _fix_series(t[:3]) + t[3:split], _fix_number(t[split:])


# Seri 2 chữ cái của ô tô (liên doanh, kinh tế, ngoại giao...)
TWO_LETTER_SERIES = {"LD", "KT", "NG", "NN", "DA", "MD", "TD", "HC"}


def _series_len(t):
    n = len(t)
    if n >= 9 and _fix_series(t[:5])[2:4] == "MD" and t[4].isdigit():
        return 5  # xe máy điện 1 dòng: 41MD1 + 4-5 số
    if n >= 9:
        return n - 5
    if n == 8:
        # 51LD1234 (seri 2 chữ + 4 số) hay 51F12345 (seri 1 chữ + 5 số)?
        # Ký tự thứ 4 là chữ có thể do OCR đọc nhầm số (l->1), nên chỉ coi là
        # seri 2 chữ khi khớp seri có thật hoặc là chữ không thể nhầm với số.
        if _fix_series(t[:3])[2:3] + t[3] in TWO_LETTER_SERIES:
            return 4
        return 4 if t[3].isalpha() and t[3] not in TO_DIGIT else 3
    return 3


def format_plate(series, number):
    """Định dạng hiển thị: '51F-123.45', '59X1-123.45', '30A-1234'."""
    if not series and not number:
        return ""
    if len(series) == 2 and series.isalpha() and len(number) == 4:      # quân đội
        return f"{series}-{number[:2]}-{number[2:]}"
    if len(series) == 7 and series[5:] in DIPLO_CODES:                 # ngoại giao cũ
        return f"{series[:2]}-{series[2:5]}-{series[5:]}-{number}"
    if len(number) == 5:
        number = f"{number[:3]}.{number[3:]}"
    return f"{series}-{number}" if number else series
