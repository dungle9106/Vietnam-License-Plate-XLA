"""Đánh giá mô hình phát hiện trên tập test: Precision, Recall, mAP@0.5, mAP@0.5:0.95.

    python scripts/eval_detector.py --weights weights/best.pt --data data/lp/data.yaml
"""
import argparse
import sys
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lpr  # noqa: E402,F401  (cấu hình in UTF-8 trên Windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--data", default="data/lp/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    m = YOLO(args.weights).val(data=args.data, split=args.split, imgsz=args.imgsz,
                               device=args.device, project=str(ROOT / "runs"),
                               name=f"eval_{args.split}", exist_ok=True)
    res = {"precision": float(m.box.mp), "recall": float(m.box.mr),
           "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}
    print("\n=== KẾT QUẢ PHÁT HIỆN BIỂN SỐ (tập %s) ===" % args.split)
    for k, v in res.items():
        print(f"{k:10s}: {v:.4f}")
    out = ROOT / "runs" / f"eval_{args.split}" / "detector_metrics.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Đã lưu {out} (kèm biểu đồ PR curve, confusion matrix trong cùng thư mục)")


if __name__ == "__main__":
    main()
