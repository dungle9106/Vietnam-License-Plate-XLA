"""Chạy nhận diện biển số từ dòng lệnh.

    python predict.py --source anh.jpg                 # 1 ảnh
    python predict.py --source thu_muc_anh/            # cả thư mục
    python predict.py --source video.mp4 --show        # video
    python predict.py --source 0 --show                # webcam
    python predict.py --source thu_muc_anh/ --excel ket_qua.xlsx
    python predict.py --source thu_muc_anh/ --single --json ket_qua.json   # 1 biển / ảnh
Kết quả lưu trong runs/predict/. YOLOv8s (weights/best.pt) + PaddleOCR: ảnh dùng bản chính xác,
video / webcam dùng bản nhanh (YOLO 640 px + PaddleOCR phát hiện dòng mobile).
"""
import argparse
import json
from pathlib import Path

import cv2

from lpr.batch import export_excel, list_images, process_paths, to_json
from lpr.io import DEFAULT_WEIGHTS, imread, imwrite, is_video
from lpr.pipeline import LPRPipeline, VideoProcessor, draw_results

OUT = Path(__file__).resolve().parent / "runs" / "predict"


def run_images(pipe, paths, multi=True, excel=None, json_path=None, meta=None, root=None):
    def progress(i, n, rec):
        texts = json.dumps(rec.plate_list(), ensure_ascii=False) if rec.plates else "[]"
        note = f"  ! {rec.error}" if rec.error else ""
        print(f"[{i}/{n}] {rec.name}: {texts}  ({rec.ms:.0f} ms){note}")
        if rec.plates:
            img = imread(rec.path)
            rel = Path(rec.path).relative_to(root) if root else Path(rec.name)
            imwrite(OUT / rel, draw_results(img, rec.plates))

    records = process_paths(pipe, paths, multi=multi, on_progress=progress)
    n_plates = sum(len(r.plates) for r in records)
    print(f"\nĐã xử lý {len(records)} ảnh, đọc được {n_plates} biển số. Ảnh kết quả: {OUT}")
    if excel:
        print(f"Đã xuất Excel: {export_excel(records, excel, meta=meta)}")
    if json_path:
        Path(json_path).write_text(json.dumps(to_json(records), ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"Đã xuất JSON: {json_path}")


def run_video(pipe, source, show):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được nguồn video: {source}")
    vp = VideoProcessor(pipe)
    writer, seen = None, {}
    out_path = OUT / (f"{Path(source).stem}_out.mp4" if not source.isdigit() else "webcam_out.mp4")
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = vp.process_frame(frame)
        for r in results:
            if r.valid:
                seen[r.track_id] = r.display
        vis = draw_results(frame, results, vp.fps)
        if writer is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (vis.shape[1], vis.shape[0]))
        writer.write(vis)
        if show:
            cv2.imshow("LPR - nhan Q de thoat", vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Các biển số đã nhận diện:", sorted(set(seen.values())))
    print(f"Video kết quả: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="File trọng số YOLOv8 (.pt)")
    ap.add_argument("--source", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--ocr", choices=["paddleocr", "paddleocr_fast"], default=None,
                    help="Mặc định: paddleocr (chính xác) cho ảnh, paddleocr_fast (nhanh) cho video/webcam")
    ap.add_argument("--single", action="store_true",
                    help="Mỗi ảnh chỉ lấy 1 biển số tốt nhất (mặc định: đọc mọi biển trong ảnh)")
    ap.add_argument("--excel", help="Xuất kết quả ảnh/thư mục ra file .xlsx")
    ap.add_argument("--json", help="Xuất kết quả ra file .json (mỗi ảnh một mảng biển số)")
    ap.add_argument("--no-recursive", action="store_true", help="Không quét thư mục con")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Không thấy mô hình YOLOv8: {args.weights}")
    src = Path(args.source)
    video = args.source.isdigit() or is_video(src)
    if video:
        from lpr.detector import FAST_IMGSZ
        pipe = LPRPipeline(args.weights, det_conf=args.conf, imgsz=FAST_IMGSZ,
                           ocr_engine=args.ocr or "paddleocr_fast")
    else:
        pipe = LPRPipeline(args.weights, det_conf=args.conf, ocr_engine=args.ocr or "paddleocr")
    meta = {"Bộ phát hiện": f"YOLOv8s {args.weights}", "Engine OCR": args.ocr or "paddleocr",
            "Chế độ": "1 biển / ảnh" if args.single else "Nhiều biển / ảnh",
            "Nguồn": str(src.resolve())}
    opts = dict(multi=not args.single, excel=args.excel, json_path=args.json, meta=meta)
    if src.is_dir():
        paths = list_images(src, recursive=not args.no_recursive)
        print(f"Tìm thấy {len(paths)} ảnh trong {src}")
        run_images(pipe, paths, root=src, **opts)
    elif video:
        run_video(pipe, args.source, args.show)
    else:
        run_images(pipe, [src], **opts)


if __name__ == "__main__":
    main()
