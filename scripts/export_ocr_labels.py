"""Tạo / bổ sung bộ dữ liệu ảnh biển số có đáp án (nhãn ký tự) để đo độ chính xác OCR
và sau này tinh chỉnh (fine-tune) PaddleOCR.

Mỗi biển số được cắt ra 1 file ảnh trong data/ocr_labels/crops/, PaddleOCR đọc sẵn 1 gợi ý.
Sau đó mở trang gán nhãn  http://127.0.0.1:5000/label  (chạy python web/server.py)
để xem từng biển, sửa chữ cho đúng rồi bấm Enter.

Nguồn ảnh (dùng được nhiều nguồn trong 1 lần chạy, chạy lại nhiều lần để bổ sung dần):
    python scripts/export_ocr_labels.py --split test                 # data/lp_full/images/test, box theo nhãn YOLO
    python scripts/export_ocr_labels.py --images "D:/anh_bien_so"    # thư mục ảnh bất kỳ (quét cả thư mục con)
    python scripts/export_ocr_labels.py --video xe.mp4 --every 0.5   # cắt biển từ video, 0.5 s lấy 1 khung

Nguồn --images / --video không có nhãn box nên dùng YOLOv8s (weights/best.pt) để tìm biển.
Chạy lại KHÔNG ghi đè nhãn đã sửa: biển đã có trong labels.csv được giữ nguyên.
Bỏ qua ảnh tăng cường *_augN (cùng biển với ảnh gốc, đo vào sẽ bị tính trùng).

Cột của labels.csv:
    crop     đường dẫn ảnh biển đã cắt          image    ảnh / video gốc (video: kèm @giây)
    box_idx  thứ tự biển trong ảnh               pred     gợi ý của PaddleOCR lúc tạo
    text     ĐÁP ÁN (sửa tay; '?' = không đọc được, bỏ qua khi đo)
    checked  1 = đã có người kiểm tra (chỉ các dòng này được dùng để đo / huấn luyện)
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lpr.io import IMG_EXTS, imread, imwrite  # noqa: E402
from lpr.pipeline import crop_plate  # noqa: E402

FIELDS = ["crop", "image", "box_idx", "pred", "text", "checked"]
AUG_RE = re.compile(r"_aug\d+$")


def load_gt_boxes(img_path, img_shape):
    """Đọc nhãn YOLO của ảnh (images/<split>/x.jpg -> labels/<split>/x.txt) ra pixel."""
    lbl = img_path.parents[1].parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
    h, w = img_shape[:2]
    boxes = []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) == 5:
                xc, yc, bw, bh = (float(x) for x in v[1:])
                boxes.append(((xc - bw / 2) * w, (yc - bh / 2) * h,
                              (xc + bw / 2) * w, (yc + bh / 2) * h))
    return boxes


def read_labels(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("checked", "")
        r["checked"] = r["checked"] or ""
    return rows


def write_labels(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)  # ghi file tạm rồi đổi tên: không hỏng file nhãn nếu bị ngắt giữa chừng


def safe_name(s):
    return re.sub(r"[^\w.-]+", "_", s)[-80:]


class Exporter:
    def __init__(self, out, ocr, detector=None):
        self.out, self.ocr, self.det = Path(out), ocr, detector
        self.csv = self.out / "labels.csv"
        self.rows = read_labels(self.csv)
        self.known = {r["crop"] for r in self.rows}
        self.added = 0

    def add(self, img, box, source, idx, stem):
        crop = crop_plate(img, box)
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return
        crop_path = (self.out / "crops" / f"{safe_name(stem)}_{idx}.jpg")
        key = crop_path.relative_to(ROOT).as_posix() if crop_path.is_absolute() else crop_path.as_posix()
        if key in self.known:
            return  # đã có (có thể đã được sửa tay) -> giữ nguyên
        imwrite(crop_path, crop)
        pred = self.ocr.read(crop).display if self.ocr else ""
        self.rows.append({"crop": key, "image": source, "box_idx": idx, "pred": pred,
                          "text": pred, "checked": ""})
        self.known.add(key)
        self.added += 1
        if self.added % 25 == 0:
            write_labels(self.csv, self.rows)  # lưu dần
            print(f"  ... {self.added} biển mới")

    def detect(self, img):
        return [d[:4] for d in self.det.detect(img)]

    def save(self):
        write_labels(self.csv, self.rows)


def iter_images(folder, recursive=True):
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.suffix.lower() in IMG_EXTS and "ocr_labels" not in p.parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/lp_full")
    ap.add_argument("--split", default=None, help="Lấy ảnh data/lp_full/images/<split> với box theo nhãn YOLO")
    ap.add_argument("--images", action="append", default=[], help="Thư mục ảnh bất kỳ (lặp lại được)")
    ap.add_argument("--video", action="append", default=[], help="File video (lặp lại được)")
    ap.add_argument("--every", type=float, default=0.5, help="Video: cách bao nhiêu giây lấy 1 khung")
    ap.add_argument("--conf", type=float, default=0.5, help="Ngưỡng tin cậy YOLO khi tự tìm biển")
    ap.add_argument("--out", default="data/ocr_labels")
    ap.add_argument("--no-ocr", action="store_true", help="Không điền sẵn gợi ý (nhanh hơn)")
    args = ap.parse_args()
    if not (args.split or args.images or args.video):
        ap.error("Chọn ít nhất 1 nguồn: --split test, --images <thư mục> hoặc --video <file>")

    ocr = None
    if not args.no_ocr:
        from lpr.paddle_ocr import PaddlePlateOCR
        ocr = PaddlePlateOCR()
    det = None
    if args.images or args.video:
        from lpr.detector import PlateDetector
        det = PlateDetector(ROOT / "weights" / "best.pt", conf=args.conf)
    ex = Exporter(ROOT / args.out, ocr, det)
    before = len(ex.rows)

    if args.split:
        folder = Path(args.data_dir) / "images" / args.split
        imgs = [p for p in iter_images(folder, recursive=False) if not AUG_RE.search(p.stem)]
        print(f"[{folder}] {len(imgs)} ảnh gốc (bỏ ảnh *_aug)")
        for p in imgs:
            img = imread(p)
            if img is not None:
                for bi, box in enumerate(load_gt_boxes(p, img.shape)):
                    ex.add(img, box, p.as_posix(), bi, f"{args.split}_{p.stem}")

    for folder in args.images:
        imgs = [p for p in iter_images(folder) if not AUG_RE.search(p.stem)]
        print(f"[{folder}] {len(imgs)} ảnh")
        for p in imgs:
            img = imread(p)
            if img is not None:
                for bi, box in enumerate(ex.detect(img)):
                    ex.add(img, box, p.as_posix(), bi, f"{Path(folder).name}_{p.stem}")

    for video in args.video:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"Không mở được video: {video}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps * args.every))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(f"[{video}] {n} khung, lấy 1 khung mỗi {args.every} s")
        f = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if f % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    t = f / fps
                    for bi, box in enumerate(ex.detect(frame)):
                        ex.add(frame, box, f"{Path(video).as_posix()}@{t:.1f}s", bi,
                               f"{Path(video).stem}_{int(t * 10):05d}")
            f += 1
        cap.release()

    ex.save()
    done = sum(r["checked"] == "1" for r in ex.rows)
    print(f"\nThêm {len(ex.rows) - before} biển mới. Tổng {len(ex.rows)} biển, đã kiểm tra {done}.")
    print(f"File nhãn: {ex.csv}")
    print("=> Chạy  python web/server.py  rồi mở  http://127.0.0.1:5000/label  để sửa đáp án.")


if __name__ == "__main__":
    main()
