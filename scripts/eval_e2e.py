"""Đánh giá end-to-end: tỉ lệ ảnh mà hệ thống vừa phát hiện đúng vị trí (IoU >= 0.5)
vừa đọc đúng toàn bộ chuỗi ký tự của MỌI biển số đã gán nhãn trong ảnh.

    python scripts/eval_e2e.py
    python scripts/eval_e2e.py --imgsz 640 --ocr paddleocr_fast   # cấu hình video / webcam
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lpr.io import imread  # noqa: E402
from lpr.metrics import match_boxes  # noqa: E402
from lpr.pipeline import LPRPipeline  # noqa: E402
from lpr.plate_format import normalize  # noqa: E402
from scripts.eval_ocr import load_labels  # noqa: E402
from scripts.export_ocr_labels import load_gt_boxes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--labels", default="data/ocr_labels/labels.csv")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--ocr", choices=["paddleocr", "paddleocr_fast"], default="paddleocr")
    ap.add_argument("--imgsz", type=int, default=None, help="Kích thước ảnh YOLO (mặc định 960)")
    args = ap.parse_args()

    by_image = defaultdict(dict)
    for r in load_labels(args.labels):
        # Chỉ ảnh có nhãn box YOLO (data/lp/images/<split>) mới đo end-to-end được;
        # biển lấy từ --images / --video thì box do chính YOLO tìm -> chỉ dùng cho eval_ocr.py
        if "/images/" not in r["image"].replace("\\", "/") or "@" in r["image"]:
            continue
        by_image[r["image"]][int(r["box_idx"])] = normalize(r["text"])

    pipe = LPRPipeline(args.weights, ocr_engine=args.ocr, imgsz=args.imgsz)
    n_det_ok = n_all_ok = n_plates = n_plates_ok = 0
    t0 = time.perf_counter()
    for img_path, texts in by_image.items():
        img = imread(img_path)
        gts = load_gt_boxes(Path(img_path), img.shape)
        results = pipe.process_image(img)
        matched = match_boxes([r.box for r in results], gts, args.iou)
        det_ok = all(bi in matched for bi in texts)
        plates_ok = [bi in matched and results[matched[bi]].text == gt for bi, gt in texts.items()]
        n_det_ok += det_ok
        n_all_ok += all(plates_ok)
        n_plates += len(plates_ok)
        n_plates_ok += sum(plates_ok)
    dt = time.perf_counter() - t0

    n = len(by_image)
    name = f"YOLOv8s + {args.ocr}"
    print(f"\n=== ĐÁNH GIÁ END-TO-END - {name} ({n} ảnh, {n_plates} biển số) ===")
    print(f"Phát hiện đúng mọi biển trong ảnh     : {n_det_ok / max(n, 1):.2%}")
    print(f"Phát hiện + đọc đúng toàn bộ (theo ảnh): {n_all_ok / max(n, 1):.2%}")
    print(f"Phát hiện + đọc đúng (theo biển số)    : {n_plates_ok / max(n_plates, 1):.2%}")
    print(f"Thời gian trung bình                  : {dt / max(n, 1) * 1000:.0f} ms/ảnh")


if __name__ == "__main__":
    main()
