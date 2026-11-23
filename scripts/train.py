"""Huấn luyện YOLOv8 phát hiện biển số -> weights/best.pt.

Ví dụ (nên chạy trên GPU Colab/Kaggle, xem notebooks/train_colab.ipynb):
    python scripts/train.py --data data/lp_full/data.yaml --model yolov8s.pt --epochs 100
"""
import argparse
import sys
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lpr  # noqa: E402,F401  (cấu hình in UTF-8 trên Windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lp_full/data.yaml")
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8n/s/m.pt (n nhanh nhất, m chính xác hơn)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 cho GPU, cpu cho CPU")
    ap.add_argument("--name", default="lp_yolov8")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    data, imgsz, name, out_name = args.data, args.imgsz, args.name, "best.pt"
    # Tăng cường dữ liệu theo đề cương: xoay nhẹ, sáng/tương phản, lật ngang.
    # Nhiễu/blur được Ultralytics tự thêm khi đã cài albumentations.
    aug = dict(degrees=5.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, fliplr=0.5,
               translate=0.1, scale=0.5, mosaic=1.0)

    model = YOLO(args.model)
    model.train(
        data=data, epochs=args.epochs, imgsz=imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=str(ROOT / "runs"), name=name,
        patience=20, exist_ok=True, seed=42, **aug,
    )

    best = ROOT / "runs" / name / "weights" / "best.pt"
    if best.exists():
        (ROOT / "weights").mkdir(exist_ok=True)
        shutil.copy2(best, ROOT / "weights" / out_name)
        print(f"Đã copy mô hình tốt nhất -> {ROOT / 'weights' / out_name}")


if __name__ == "__main__":
    main()
