"""Đọc biển số hàng loạt (cả thư mục ảnh) và xuất kết quả ra Excel.

    recs = process_paths(pipe, list_images("thu_muc_anh"), multi=True)
    export_excel(recs, "ket_qua.xlsx")

Mỗi ảnh cho ra một mảng biển số (ảnh có nhiều xe -> nhiều biển). File Excel gồm 3 sheet:
    "Biển số"  : mỗi dòng 1 biển (kèm ảnh cắt vùng biển, loại, màu, tỉnh, độ tin cậy)
    "Theo ảnh" : mỗi dòng 1 ảnh, cột danh sách biển số dạng mảng JSON
    "Tổng hợp" : số ảnh, số biển, tỉ lệ đúng định dạng, thống kê theo loại biển / tỉnh
"""
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2

from .io import IMG_EXTS, imread


@dataclass
class ImageRecord:
    path: str                               # đường dẫn (hoặc tên file khi tải lên web)
    plates: list = field(default_factory=list)   # list PlateResult hoặc dict (PlateResult.to_dict + 'crop_jpg')
    ms: float = 0.0
    error: str = ""

    @property
    def name(self):
        return Path(self.path).name

    def plate_dicts(self):
        return [p if isinstance(p, dict) else p.to_dict() for p in self.plates]

    def plate_list(self):
        """Mảng chuỗi biển số của ảnh, vd ['51F-123.45', '59X1-678.90']."""
        return [d["bien_so"] or "?" for d in self.plate_dicts()]


def list_images(folder, recursive=True):
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


def process_image_record(pipe, img, path, multi=True):
    t0 = time.perf_counter()
    try:
        plates = pipe.process_image(img, multi=multi)
        return ImageRecord(str(path), plates, (time.perf_counter() - t0) * 1000)
    except Exception as e:  # noqa: BLE001 - một ảnh lỗi không làm dừng cả lô
        return ImageRecord(str(path), [], (time.perf_counter() - t0) * 1000, str(e))


def process_paths(pipe, paths, multi=True, on_progress=None, should_stop=None):
    """Nhận diện lần lượt từng ảnh. on_progress(i, n, record) được gọi sau mỗi ảnh."""
    records, n = [], len(paths)
    for i, p in enumerate(paths, 1):
        if should_stop and should_stop():
            break
        img = imread(p)
        rec = (ImageRecord(str(p), error="Không đọc được ảnh") if img is None
               else process_image_record(pipe, img, p, multi))
        records.append(rec)
        if on_progress:
            on_progress(i, n, rec)
    return records


def to_json(records):
    return [{"anh": r.name, "duong_dan": r.path, "so_bien": len(r.plates),
             "bien_so": r.plate_list(), "chi_tiet": [_strip(d) for d in r.plate_dicts()],
             "thoi_gian_ms": round(r.ms, 1), "loi": r.error} for r in records]


def _strip(d):
    return {k: v for k, v in d.items() if k != "crop_jpg"}


def _crop_png(plate, max_h=48, max_w=200):
    """Ảnh vùng biển thu nhỏ (PNG bytes) để chèn vào Excel, hoặc None."""
    if isinstance(plate, dict):
        data = plate.get("crop_jpg")
        if not data:
            return None
        import numpy as np
        crop = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    else:
        crop = plate.crop
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    s = min(max_h / h, max_w / w, 1.0)
    crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", crop)
    return buf.tobytes() if ok else None


def export_excel(records, path=None, with_images=True, meta=None):
    """Ghi file .xlsx. path=None -> trả về bytes (dùng cho web). meta: dict thông tin cấu hình."""
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="1F4E78")
    warn_fill = PatternFill("solid", fgColor="FCE4D6")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(vertical="center", wrap_text=True)

    def header(ws, cols):
        for c, (title, width) in enumerate(cols, 1):
            cell = ws.cell(1, c, title)
            cell.font, cell.fill, cell.alignment, cell.border = head_font, head_fill, center, border
            ws.column_dimensions[get_column_letter(c)].width = width
        ws.freeze_panes = "A2"
        ws.row_dimensions[1].height = 30

    def finish(ws, ncols):
        for row in ws.iter_rows(min_row=2, max_col=ncols):
            for cell in row:
                cell.border = border
                if cell.alignment.horizontal is None:
                    cell.alignment = left
        if ws.max_row > 1:
            ws.auto_filter.ref = f"A1:{get_column_letter(ncols)}{ws.max_row}"

    wb = Workbook()

    # --- Sheet 1: mỗi dòng 1 biển số ---
    ws = wb.active
    ws.title = "Biển số"
    cols = [("STT", 6), ("Tên ảnh", 28), ("Biển thứ", 8), ("Biển số", 16), ("Chuỗi chuẩn hoá", 14),
            ("Loại biển", 34), ("Màu nền", 9), ("Tỉnh/TP", 18), ("Đúng định dạng", 10),
            ("Tin cậy phát hiện", 10), ("Tin cậy OCR", 10), ("Vị trí (x1,y1,x2,y2)", 20),
            ("Ảnh biển số", 30), ("Đường dẫn", 50)]
    header(ws, cols)
    row = 2
    for rec in records:
        for k, plate in enumerate(rec.plates, 1):
            d = plate if isinstance(plate, dict) else plate.to_dict()
            vals = [row - 1, rec.name, k, d["bien_so"] or "?", d["chuoi"], d["loai"], d["mau"],
                    d["tinh"], "Có" if d["hop_le"] else "Không", d["conf_phat_hien"],
                    d["conf_ocr"], ", ".join(map(str, d["box"])), None, rec.path]
            for c, v in enumerate(vals, 1):
                ws.cell(row, c, v)
            for c in (1, 3, 7, 9, 10, 11):
                ws.cell(row, c).alignment = center
            ws.cell(row, 4).font = Font(bold=True, size=12)
            ws.cell(row, 10).number_format = ws.cell(row, 11).number_format = "0.00"
            if not d["hop_le"]:
                for c in range(1, len(cols) + 1):
                    ws.cell(row, c).fill = warn_fill
            png = _crop_png(plate) if with_images else None
            if png:
                img = XLImage(BytesIO(png))
                ws.add_image(img, f"M{row}")
                ws.row_dimensions[row].height = max(20, img.height * 0.75 + 6)
            row += 1
    finish(ws, len(cols))

    # --- Sheet 2: mỗi dòng 1 ảnh, danh sách biển dạng mảng ---
    ws2 = wb.create_sheet("Theo ảnh")
    cols2 = [("STT", 6), ("Tên ảnh", 28), ("Số biển", 8), ("Danh sách biển số (mảng)", 40),
             ("Thời gian xử lý (ms)", 12), ("Ghi chú", 26), ("Đường dẫn", 50)]
    header(ws2, cols2)
    for i, rec in enumerate(records, 1):
        note = rec.error or ("Không tìm thấy biển số" if not rec.plates else "")
        vals = [i, rec.name, len(rec.plates), json.dumps(rec.plate_list(), ensure_ascii=False),
                round(rec.ms, 1), note, rec.path]
        for c, v in enumerate(vals, 1):
            ws2.cell(i + 1, c, v)
        for c in (1, 3, 5):
            ws2.cell(i + 1, c).alignment = center
        if note:
            for c in range(1, len(cols2) + 1):
                ws2.cell(i + 1, c).fill = warn_fill
    finish(ws2, len(cols2))

    # --- Sheet 3: tổng hợp ---
    ws3 = wb.create_sheet("Tổng hợp")
    plates = [p if isinstance(p, dict) else p.to_dict() for r in records for p in r.plates]
    n_img, n_plate = len(records), len(plates)
    n_valid = sum(d["hop_le"] for d in plates)
    rows = [("Thời điểm xuất", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            *((k, v) for k, v in (meta or {}).items()),
            ("Số ảnh", n_img),
            ("Số ảnh có biển số", sum(1 for r in records if r.plates)),
            ("Số ảnh lỗi", sum(1 for r in records if r.error)),
            ("Tổng số biển số", n_plate),
            ("Số biển đúng định dạng", n_valid),
            ("Tỉ lệ đúng định dạng", f"{n_valid / n_plate:.1%}" if n_plate else "-"),
            ("Thời gian TB / ảnh (ms)", round(sum(r.ms for r in records) / n_img, 1) if n_img else 0)]
    ws3.column_dimensions["A"].width, ws3.column_dimensions["B"].width = 40, 30
    r = 1
    for k, v in rows:
        ws3.cell(r, 1, k).font = Font(bold=True)
        ws3.cell(r, 2, v)
        r += 1
    for title, counter in (("Theo loại biển", Counter(d["loai"] or "(chưa xác định)" for d in plates)),
                           ("Theo tỉnh/thành", Counter(d["tinh"] or "(chưa xác định)" for d in plates)),
                           ("Theo màu nền", Counter(d["mau"] or "(chưa xác định)" for d in plates))):
        r += 1
        for c, t in enumerate((title, "Số biển"), 1):
            cell = ws3.cell(r, c, t)
            cell.font, cell.fill = head_font, head_fill
        r += 1
        for k, v in counter.most_common():
            ws3.cell(r, 1, k)
            ws3.cell(r, 2, v)
            r += 1

    if path is None:
        buf = BytesIO()
        wb.save(buf)
        return buf.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
