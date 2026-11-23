"""Xuất hình minh hoạ (Matplotlib) cho báo cáo, lưu vào runs/figures/.

    python scripts/visualize.py steps --image anh.jpg                  # các bước xử lý 1 ảnh
    python scripts/visualize.py steps --image anh.jpg --weights weights/best.pt
    python scripts/visualize.py dataset --data-dir data/lp            # thống kê dữ liệu
    python scripts/visualize.py ocr-errors --errors runs/ocr_errors.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lpr.io import imread  # noqa: E402
from lpr.visualize import (dataset_stats_figure, detection_figure,  # noqa: E402
                           ocr_errors_figure, ocr_steps_figure)

OUT = ROOT / "runs" / "figures"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=150)
    print(f"Đã lưu {path}")


def cmd_steps(args):
    from lpr.pipeline import LPRPipeline, crop_plate, draw_results

    img = imread(args.image)
    if img is None:
        raise SystemExit(f"Không đọc được ảnh {args.image}")
    pipe = LPRPipeline(args.weights)
    results = pipe.process_image(img)
    stem = Path(args.image).stem
    save(detection_figure(img, draw_results(img, results), [crop_plate(img, r.box) for r in results]),
         f"{stem}_detect.png")
    for i, r in enumerate(results):
        crop = crop_plate(img, r.box)
        save(ocr_steps_figure(crop, pipe.ocr.read(crop)), f"{stem}_ocr_{i}.png")
    if not results:
        print("Không phát hiện được biển số nào.")


def cmd_dataset(args):
    save(dataset_stats_figure(args.data_dir), "dataset_stats.png")


def cmd_ocr_errors(args):
    with open(args.errors, encoding="utf-8-sig") as f:
        save(ocr_errors_figure(list(csv.DictReader(f))), "ocr_errors.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("steps", help="Các bước phát hiện + tiền xử lý OCR trên 1 ảnh")
    p.add_argument("--image", required=True)
    p.add_argument("--weights", default="weights/best.pt")
    p.set_defaults(func=cmd_steps)
    p = sub.add_parser("dataset", help="Thống kê tập dữ liệu đã chia")
    p.add_argument("--data-dir", default="data/lp")
    p.set_defaults(func=cmd_dataset)
    p = sub.add_parser("ocr-errors", help="Phân tích lỗi OCR từ eval_ocr.py")
    p.add_argument("--errors", default="runs/ocr_errors.csv")
    p.set_defaults(func=cmd_ocr_errors)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
