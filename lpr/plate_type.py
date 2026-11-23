"""Phân loại biển số xe Việt Nam: màu nền, loại phương tiện, tỉnh/thành đăng ký.

Màu nền (Thông tư 24/2023/TT-BCA):
    trắng chữ đen  : xe cá nhân, doanh nghiệp
    vàng chữ đen   : xe kinh doanh vận tải
    xanh chữ trắng : xe cơ quan nhà nước, đơn vị sự nghiệp công lập
    đỏ chữ trắng   : xe quân đội (Bộ Quốc phòng)
Dạng chuỗi:
    51F-123.45        ô tô (seri 1 chữ)
    51LD-123.45       ô tô seri đặc biệt (LD liên doanh, KT quân đội làm kinh tế, ...)
    59-X1 123.45      xe máy (seri chữ + số)
    14-AA 178.91      xe máy dưới 50 cm3 (seri 2 chữ)
    29-MĐ1 123.45     xe máy điện
    80-NG-123-45      xe cơ quan ngoại giao / tổ chức quốc tế (NG, QT, CV, NN)
    41-291-NN-01      dạng cũ của biển ngoại giao / người nước ngoài
    TM-12-34          biển quân đội (2 chữ ký hiệu đơn vị + 4 số, nền đỏ)
"""
import re

import cv2
import numpy as np

# Mã tỉnh/thành trên biển số (theo đơn vị hành chính khi cấp biển; biển cũ vẫn giữ mã sau sáp nhập 2025)
PROVINCES = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh", "15": "Hải Phòng", "16": "Hải Phòng",
    "17": "Thái Bình", "18": "Nam Định", "19": "Phú Thọ", "20": "Thái Nguyên", "21": "Yên Bái",
    "22": "Tuyên Quang", "23": "Hà Giang", "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La",
    "27": "Điện Biên", "28": "Hòa Bình", "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội",
    "32": "Hà Nội", "33": "Hà Nội", "34": "Hải Dương", "35": "Ninh Bình", "36": "Thanh Hóa",
    "37": "Nghệ An", "38": "Hà Tĩnh", "40": "Hà Nội", "41": "TP. Hồ Chí Minh", "43": "Đà Nẵng",
    "47": "Đắk Lắk", "48": "Đắk Nông", "49": "Lâm Đồng", "50": "TP. Hồ Chí Minh",
    "51": "TP. Hồ Chí Minh", "52": "TP. Hồ Chí Minh", "53": "TP. Hồ Chí Minh",
    "54": "TP. Hồ Chí Minh", "55": "TP. Hồ Chí Minh", "56": "TP. Hồ Chí Minh",
    "57": "TP. Hồ Chí Minh", "58": "TP. Hồ Chí Minh", "59": "TP. Hồ Chí Minh",
    "60": "Đồng Nai", "61": "Bình Dương", "62": "Long An", "63": "Tiền Giang", "64": "Vĩnh Long",
    "65": "Cần Thơ", "66": "Đồng Tháp", "67": "An Giang", "68": "Kiên Giang", "69": "Cà Mau",
    "70": "Tây Ninh", "71": "Bến Tre", "72": "Bà Rịa - Vũng Tàu", "73": "Quảng Bình",
    "74": "Quảng Trị", "75": "Thừa Thiên Huế", "76": "Quảng Ngãi", "77": "Bình Định",
    "78": "Phú Yên", "79": "Khánh Hòa", "80": "Cơ quan Trung ương", "81": "Gia Lai",
    "82": "Kon Tum", "83": "Sóc Trăng", "84": "Trà Vinh", "85": "Ninh Thuận", "86": "Bình Thuận",
    "88": "Vĩnh Phúc", "89": "Hưng Yên", "90": "Hà Nam", "92": "Quảng Nam", "93": "Bình Phước",
    "94": "Bạc Liêu", "95": "Hậu Giang", "97": "Bắc Kạn", "98": "Bắc Giang", "99": "Bắc Ninh",
}

# Seri đặc biệt (2 chữ cái) của ô tô / xe máy
SPECIAL_SERIES = {
    "LD": "Ô tô doanh nghiệp có vốn nước ngoài, thuê của nước ngoài",
    "KT": "Ô tô quân đội làm kinh tế",
    "DA": "Ô tô của các Ban quản lý dự án nước ngoài",
    "TD": "Xe thí điểm / sản xuất lắp ráp trong nước",
    "HC": "Ô tô hạn chế phạm vi hoạt động",
    "MD": "Xe máy điện",
}
DIPLOMATIC = {
    "NG": "Xe cơ quan đại diện ngoại giao",
    "QT": "Xe cơ quan đại diện tổ chức quốc tế",
    "CV": "Xe nhân viên hành chính kỹ thuật cơ quan ngoại giao",
    "NN": "Xe tổ chức, cá nhân người nước ngoài",
}
COLOR_USE = {
    "trắng": "cá nhân / doanh nghiệp",
    "vàng": "kinh doanh vận tải",
    "xanh": "cơ quan nhà nước",
    "đỏ": "quân đội",
}

MILITARY_RE = re.compile(r"^[A-Z]{2}\d{4}$")
DIPLO_NEW_RE = re.compile(r"^\d{2}(NG|QT|CV|NN)\d{3,5}$")
DIPLO_OLD_RE = re.compile(r"^\d{5}(NG|QT|CV|NN)\d{2,3}$")


def plate_color(crop):
    """Ước lượng màu nền biển số: 'trắng' | 'vàng' | 'xanh' | 'đỏ'.

    Nền chiếm phần lớn diện tích biển (chữ chỉ ~20-35%), nên lấy trung vị HSV của các pixel
    thuộc lớp Otsu có diện tích lớn hơn trong vùng giữa biển (bỏ viền 10%).
    """
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return "trắng"
    h, w = crop.shape[:2]
    roi = crop[int(h * 0.1):max(int(h * 0.9), 1), int(w * 0.08):max(int(w * 0.92), 1)]
    if roi.size == 0:
        roi = crop
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = bw > 0 if (bw > 0).mean() >= 0.5 else bw == 0
    pix = hsv[mask]
    if len(pix) < 10:
        pix = hsv.reshape(-1, 3)
    hue, sat, val = (float(np.median(pix[:, i])) for i in range(3))
    # Biển trắng hay bị ám màu: đèn đường vàng ban đêm, thân xe đỏ/cam phía sau, ảnh tối.
    # Đo trên 122 biển data/ocr_labels: biển trắng bị ám màu có bão hoà nền tới 125, biển vàng /
    # xanh thật ~181-201 -> ngưỡng 150 (ngưỡng cũ 70: 13/15 biển bị gọi là có màu thực ra là trắng).
    # Hệ quả: biển ĐỎ nền sẫm (quân đội BH-52-62: bão hoà ~85) bị gọi là trắng. Không hạ ngưỡng
    # riêng cho đỏ được (biển trắng 86C-038.64 chữ đậm trên xe xanh lam đo ra gần y hệt: ~88) ->
    # biển quân đội được nhận theo DẠNG CHỮ (2 chữ cái + số) trong lpr/ocr.py build_result thay vì theo màu.
    if sat < 150 or val < 40:
        return "trắng"
    if hue < 10 or hue >= 160:
        return "đỏ"
    if 12 <= hue <= 38:
        return "vàng"
    if 90 <= hue <= 135:
        return "xanh"
    return "trắng"


def province(text):
    """Tên tỉnh/thành theo 2 số đầu; '' nếu không xác định (vd biển quân đội)."""
    return PROVINCES.get(text[:2], "") if len(text) >= 2 and text[:2].isdigit() else ""


def classify(series, number="", n_lines=1, color="trắng"):
    """Mô tả loại biển số từ phần seri (vd '59X1') + dãy số (vd '12345'), số dòng và màu nền.

    Cần tách seri vì '51F12345' (ô tô, seri 51F) và '29B11234' (xe máy, seri 29B1) dài bằng nhau.
    """
    text = series + number
    if not text:
        return ""
    if MILITARY_RE.match(text):
        return "Biển quân đội"
    m = DIPLO_NEW_RE.match(text) or DIPLO_OLD_RE.match(text)
    if m:
        return DIPLOMATIC[m.group(1)]
    code = series[2:]                            # phần seri sau mã tỉnh: 'F', 'X1', 'AA', 'LD'
    if code[:2] in SPECIAL_SERIES:
        return SPECIAL_SERIES[code[:2]]
    if len(code) == 2 and code[0].isalpha() and code[1].isdigit():
        kind = "Xe máy"                          # 59X1-123.45, 29B1-1234
    elif len(code) == 2 and code.isalpha() and n_lines >= 2:
        kind = "Xe máy dưới 50 cm³"             # 14AA-178.91 (biển 2 dòng)
    else:
        kind = "Ô tô" + (" (biển 2 dòng)" if n_lines >= 2 else "")
    use = COLOR_USE.get(color, "") if color != "đỏ" else ""  # nền đỏ mà chuỗi dân sự: không suy ra
    return f"{kind} - {use}" if use else kind
