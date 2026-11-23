"""Đánh giá OCR trên vùng biển số đã gán nhãn tay (data/ocr_labels): Exact-match Accuracy và CER.

So sánh 2 cấu hình để thấy tác dụng của hậu xử lý theo định dạng biển số VN:
  - raw      : chuỗi PaddleOCR đọc trực tiếp
  - corrected: sau khi sửa nhầm chữ/số theo vị trí

    python scripts/eval_ocr.py                       # PaddleOCR bản chính xác, tập test
    python scripts/eval_ocr.py --ocr paddleocr_fast  # bản nhanh (video / webcam)
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lpr.io import imread  # noqa: E402
from lpr.metrics import levenshtein  # noqa: E402
from lpr.ocr import make_ocr  # noqa: E402
from lpr.plate_format import normalize  # noqa: E402


def label_split(row):
    """Biển thuộc tập nào: 'train' / 'val' (cắt từ data/lp_full/images/train|val - dùng để tinh chỉnh
    PaddleOCR) hay 'test' (data/lp_full/images/test, Bo_anh_test_..., video/ảnh riêng - chỉ dùng để đo)."""
    img = row["image"].replace("\\", "/")
    for s in ("train", "val"):
        if f"/images/{s}/" in img:
            return s
    return "test"


def load_labels(path, checked_only=True, subset="test"):
    """Đọc labels.csv. Có cột `checked` (trang /label) thì mặc định chỉ lấy biển đã được người
    kiểm tra - gợi ý OCR chưa ai xem lại không phải đáp án. Bỏ dòng trống hoặc '?'."""
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if checked_only and rows and "checked" in rows[0]:
        rows = [r for r in rows if r.get("checked") == "1"]
    for r in rows:  # đường dẫn tương đối tính từ thư mục dự án (chạy script ở đâu cũng được)
        if not Path(r["crop"]).is_absolute():
            r["crop"] = str(ROOT / r["crop"])
    if subset != "all":  # mặc định chỉ đo trên tập test: đo cả biển đã dùng để huấn luyện là gian lận
        rows = [r for r in rows if label_split(r) == subset]
    return [r for r in rows if normalize(r["text"]) and r["text"].strip() != "?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/ocr_labels/labels.csv")
    ap.add_argument("--errors", default="runs/ocr_errors.csv")
    ap.add_argument("--ocr", choices=["paddleocr", "paddleocr_fast"], default="paddleocr",
                    help="paddleocr: bản chính xác (ảnh tĩnh); paddleocr_fast: bản nhanh (video)")
    ap.add_argument("--subset", default="test", choices=["test", "val", "train", "all"],
                    help="Đo trên tập nào (mặc định test - KHÔNG dùng biển train/val đã dùng để tinh chỉnh)")
    args = ap.parse_args()

    rows = load_labels(args.labels, subset=args.subset)
    print(f"Tập đo: {args.subset} ({len(rows)} biển)")
    if not rows:
        sys.exit("Chưa có biển nào được kiểm tra (checked=1). Mở http://127.0.0.1:5000/label để gán nhãn.")
    ocr = make_ocr(args.ocr)
    print(f"Engine OCR: {args.ocr}")
    stats = {k: {"exact": 0, "edits": 0} for k in ("raw", "corrected")}
    total_chars, errors = 0, []
    for r in rows:
        gt = normalize(r["text"])
        crop = imread(r["crop"])
        res = ocr.read(crop)
        preds = {"raw": res.raw, "corrected": res.text}
        total_chars += len(gt)
        for k, pred in preds.items():
            stats[k]["exact"] += pred == gt
            stats[k]["edits"] += levenshtein(pred, gt)
        if res.text != gt:
            errors.append({"crop": r["crop"], "gt": gt, "pred": res.text, "raw": res.raw})

    n = len(rows)
    print(f"\n=== KẾT QUẢ OCR ({n} biển số, {total_chars} ký tự) ===")
    print(f"{'Cấu hình':12s} {'Exact-match':>12s} {'CER':>8s}")
    for k, s in stats.items():
        print(f"{k:12s} {s['exact'] / max(n, 1):12.2%} {s['edits'] / max(total_chars, 1):8.2%}")

    Path(args.errors).parent.mkdir(parents=True, exist_ok=True)
    with open(args.errors, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "gt", "pred", "raw"])
        w.writeheader()
        w.writerows(errors)
    print(f"Các trường hợp đọc sai: {args.errors}")


if __name__ == "__main__":
    main()
