"""Web app nhận diện biển số xe (Flask).

    python web/server.py                 # mở http://127.0.0.1:5000
    python web/server.py --host 0.0.0.0  # cho máy khác trong mạng LAN truy cập

Cấu hình cố định: YOLOv8s (weights/best.pt) phát hiện biển số + PaddleOCR đọc ký tự.

API:
    GET  /api/config     cấu hình đang dùng (YOLOv8s + PaddleOCR)
    POST /api/recognize  multipart: files=<ảnh...>, multi=1|0, conf=0.4 -> JSON: mỗi ảnh một mảng biển số
    POST /api/export     JSON {"images": [...kết quả /api/recognize...]} -> file Excel .xlsx
    POST /api/stream/frame  multipart: sid=<mã phiên>, frame=<jpg>, conf
                         -> biển số trong khung hình (video/webcam), có track_id + bỏ phiếu OCR qua nhiều khung
    POST /api/stream/stop   sid=<mã phiên> -> giải phóng phiên
"""
import argparse
import base64
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lpr.batch import ImageRecord, export_excel  # noqa: E402
from lpr.ocr import OCR_ENGINES, make_ocr  # noqa: E402
from lpr.quality import plate_quality  # noqa: E402
from lpr.pipeline import LPRPipeline, VideoProcessor, crop_plate, draw_results  # noqa: E402

WEIGHTS = ROOT / "weights" / "best.pt"   # YOLOv8s tự huấn luyện
OCR_ENGINE = "paddleocr"                  # chỉ dùng PaddleOCR
MAX_SIDE = 1920  # ảnh lớn hơn được thu nhỏ trước khi nhận diện cho nhanh

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.json.ensure_ascii = False

_lock = threading.Lock()      # mô hình không an toàn đa luồng -> xử lý tuần tự
_detectors, _ocrs = {}, {}
_streams = {}                 # mã phiên video/webcam -> {"vp": VideoProcessor, "key", "t", "sent"}
STREAM_TTL = 300              # phiên không gửi khung hình quá 5 phút thì bị xoá


@app.errorhandler(Exception)
def json_error(e):
    """Mọi lỗi trả về JSON (không phải trang HTML) để giao diện hiện đúng nguyên nhân.
    Hay gặp: ổ đĩa chứa thư mục tạm bị đầy -> Flask không lưu tạm được ảnh tải lên (> 500 KB)."""
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return jsonify({"error": f"{e.code} {e.name}: {e.description}"}), e.code
    traceback.print_exc()
    msg = f"{type(e).__name__}: {e}"
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        msg = "Ổ đĩa chứa thư mục tạm (thường là C:) đã đầy - hãy giải phóng dung lượng. " + msg
    return jsonify({"error": msg}), 500


def get_detector(conf, fast=False):
    """Ảnh tĩnh: YOLO imgsz 960 (tìm được biển nhỏ trong ảnh lớn); video/webcam (fast): imgsz 640
    cho nhanh. Hai bản dùng chung 1 mô hình đã tải."""
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Không thấy mô hình YOLOv8s: {WEIGHTS}")
    if "yolo" not in _detectors:
        from lpr.detector import FAST_IMGSZ, PlateDetector
        _detectors["yolo"] = PlateDetector(WEIGHTS)
        _detectors["yolo_fast"] = _detectors["yolo"].with_imgsz(FAST_IMGSZ)
    det = _detectors["yolo_fast" if fast else "yolo"]
    det.conf = conf
    return det


def get_ocr(engine=OCR_ENGINE):
    """Ảnh tĩnh: PaddleOCR bản chính xác (phát hiện dòng bản server, ~2 s/biển trên CPU)."""
    if engine not in _ocrs:
        _ocrs[engine] = make_ocr(engine)
    return _ocrs[engine]


def get_fast_ocr():
    """Video / webcam: PaddleOCR bản nhanh (phát hiện dòng bản mobile, ~0.4 s/biển), dùng chung
    mô hình nhận dạng với bản chính xác. Bản server làm video chỉ còn ~0.5 khung/s."""
    if "fast" not in _ocrs:
        from lpr.paddle_ocr import make_fast
        _ocrs["fast"] = make_fast(get_ocr())
    return _ocrs["fast"]


def b64jpg(img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


@app.get("/")
def index():
    return render_template("index.html")


# ================= Gán nhãn bộ dữ liệu OCR (data/ocr_labels/labels.csv) =================
# Tạo dữ liệu: python scripts/export_ocr_labels.py --split test / --images <thư mục> / --video <file>
LABELS_DIR = ROOT / "data" / "ocr_labels"
LABELS_CSV = LABELS_DIR / "labels.csv"
_labels_lock = threading.Lock()


@app.get("/label")
def label_page():
    return render_template("label.html")


@app.get("/api/labels")
def labels_list():
    from scripts.export_ocr_labels import read_labels
    with _labels_lock:
        rows = read_labels(LABELS_CSV)
    from scripts.eval_ocr import label_split
    return jsonify({"rows": [{"i": i, "crop": r["crop"], "image": r["image"], "pred": r["pred"],
                              "text": r["text"], "checked": r["checked"] == "1", "split": label_split(r)}
                             for i, r in enumerate(rows)],
                    "csv": str(LABELS_CSV.relative_to(ROOT))})


@app.get("/labels/crop/<path:name>")
def labels_crop(name):
    p = (LABELS_DIR / "crops" / name).resolve()
    if not p.is_relative_to((LABELS_DIR / "crops").resolve()) or not p.is_file():
        return jsonify({"error": "Không có ảnh"}), 404
    return send_file(p)


@app.post("/api/labels/<int:i>")
def labels_save(i):
    from lpr.plate_format import is_valid, normalize
    from scripts.export_ocr_labels import read_labels, write_labels
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip().upper()
    with _labels_lock:
        rows = read_labels(LABELS_CSV)
        if not 0 <= i < len(rows):
            return jsonify({"error": "Không có dòng này"}), 404
        rows[i]["text"] = text
        rows[i]["checked"] = "1" if data.get("checked", True) else ""
        write_labels(LABELS_CSV, rows)
    norm = normalize(text)
    return jsonify({"ok": True, "text": text, "chuoi": norm,
                    "hop_le": text != "?" and bool(norm) and is_valid(norm)})


@app.get("/api/config")
def config():
    return jsonify({
        "detector": "YOLOv8s", "ocr": OCR_ENGINES[OCR_ENGINE],
        "weights": str(WEIGHTS.relative_to(ROOT)) if WEIGHTS.exists() else None,
    })


@app.post("/api/recognize")
def recognize():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Chưa gửi ảnh nào (trường 'files')."}), 400
    multi = request.form.get("multi", "1") not in ("0", "false", "")
    conf = float(request.form.get("conf", 0.4))
    want_image = request.form.get("image", "1") not in ("0", "false")

    out = []
    with _lock:
        try:
            pipe = LPRPipeline.from_parts(get_detector(conf), get_ocr())
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"Không tải được mô hình: {e}"}), 500
        for f in files:
            name = f.filename or "anh.jpg"
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                out.append({"anh": name, "so_bien": 0, "bien_so": [], "chi_tiet": [],
                            "loi": "Không đọc được ảnh"})
                continue
            h, w = img.shape[:2]
            full = img  # giữ ảnh gốc: vùng biển để OCR cắt từ ảnh gốc cho rõ nét
            if max(h, w) > MAX_SIDE:
                s = MAX_SIDE / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            try:  # 1 ảnh lỗi không làm hỏng cả lượt; lỗi được ghi vào "loi" của ảnh đó
                plates = pipe.process_image(img, multi=multi, full=full if full is not img else None)
                ms = (time.perf_counter() - t0) * 1000
                details = []
                for p in plates:
                    d = p.to_dict()
                    d["crop"] = b64jpg(p.crop) if p.crop is not None and p.crop.size else ""
                    d["chat_luong"] = plate_quality(p.crop)  # biển nhỏ / mờ -> cảnh báo kết quả
                    details.append(d)
                item = {"anh": name, "so_bien": len(plates),
                        "bien_so": [p.display or "?" for p in plates],
                        "chi_tiet": details, "thoi_gian_ms": round(ms, 1), "loi": "",
                        "kich_thuoc": [w, h],
                        # toạ độ box tính theo ảnh sau khi thu nhỏ -> giao diện dùng để tô sáng vùng biển
                        "kich_thuoc_xu_ly": [img.shape[1], img.shape[0]]}
                if want_image:
                    vis = draw_results(img, plates)
                    if vis.shape[1] > 1280:  # ảnh trả về trình duyệt chỉ để xem -> thu nhỏ
                        s = 1280 / vis.shape[1]
                        vis = cv2.resize(vis, (1280, int(vis.shape[0] * s)), interpolation=cv2.INTER_AREA)
                    item["anh_ket_qua"] = b64jpg(vis, 80)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                item = {"anh": name, "so_bien": 0, "bien_so": [], "chi_tiet": [],
                        "thoi_gian_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "loi": f"{type(e).__name__}: {e}", "kich_thuoc": [w, h]}
            out.append(item)
    return jsonify({"images": out, "detector": "YOLOv8s", "ocr": OCR_ENGINE, "multi": multi})


@app.post("/api/stream/frame")
def stream_frame():
    """Nhận 1 khung hình từ video/webcam trên trình duyệt. Mỗi phiên (sid) giữ 1 VideoProcessor:
    tracker IoU gán ID cho từng biển qua các khung, OCR nhiều lần rồi bỏ phiếu chọn chuỗi tốt nhất."""
    f = request.files.get("frame")
    sid = request.form.get("sid", "")
    if f is None or not sid:
        return jsonify({"error": "Thiếu khung hình (frame) hoặc mã phiên (sid)."}), 400
    conf = float(request.form.get("conf", 0.4))
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Không giải mã được khung hình."}), 400

    with _lock:
        now = time.time()
        for k in [k for k, s in _streams.items() if now - s["t"] > STREAM_TTL]:
            del _streams[k]
        try:
            det, ocr = get_detector(conf, fast=True), get_fast_ocr()
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": f"Không tải được mô hình: {e}"}), 500
        key = (id(det), OCR_ENGINE)
        s = _streams.get(sid)
        if s is None or s["key"] != key:
            s = _streams[sid] = {"vp": VideoProcessor(LPRPipeline.from_parts(det, ocr), ocr_every=2, max_reads=6),
                                 "key": key, "sent": {}}
        s["t"] = now
        t0 = time.perf_counter()
        results = s["vp"].process_frame(img)
        ms = (time.perf_counter() - t0) * 1000

    plates = []
    for r in results:
        d = r.to_dict()
        d["track_id"] = r.track_id
        crop = crop_plate(img, r.box)
        d["chat_luong"] = plate_quality(crop)
        # Ảnh vùng biển chỉ gửi khi chuỗi đọc được của track thay đổi (đỡ tốn băng thông)
        if r.text and s["sent"].get(r.track_id) != r.text:
            d["crop"] = b64jpg(crop) if crop.size else ""
            s["sent"][r.track_id] = r.text
        plates.append(d)
    return jsonify({"plates": plates, "ms": round(ms, 1), "fps": round(s["vp"].fps, 2),
                    "detector": "YOLOv8s"})


@app.post("/api/stream/stop")
def stream_stop():
    sid = request.form.get("sid") or (request.get_json(silent=True) or {}).get("sid", "")
    with _lock:
        _streams.pop(sid, None)
    return jsonify({"ok": True})


@app.post("/api/export")
def export():
    data = request.get_json(silent=True) or {}
    images = data.get("images") or []
    if not images:
        return jsonify({"error": "Không có kết quả để xuất."}), 400
    records = []
    for it in images:
        plates = []
        for d in it.get("chi_tiet", []):
            d = dict(d)
            crop = d.pop("crop", "")
            d["crop_jpg"] = base64.b64decode(crop) if crop else None
            plates.append(d)
        records.append(ImageRecord(it.get("anh", ""), plates, float(it.get("thoi_gian_ms", 0)),
                                   it.get("loi", "")))
    meta = {"Nguồn": "Web app", "Bộ phát hiện": "YOLOv8s (weights/best.pt)",
            "Engine OCR": OCR_ENGINES[OCR_ENGINE],
            "Chế độ": "Nhiều biển / ảnh" if data.get("multi", True) else "1 biển / ảnh"}
    xlsx = export_excel(records, None, meta=meta)
    from io import BytesIO
    return send_file(BytesIO(xlsx), as_attachment=True,
                     download_name=f"bien_so_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-preload", action="store_true", help="Không tải sẵn mô hình khi khởi động")
    args = ap.parse_args()
    if not WEIGHTS.exists():
        sys.exit(f"Không thấy mô hình YOLOv8s: {WEIGHTS}")
    if not args.no_preload:
        print("Đang tải YOLOv8s + PaddleOCR...")
        get_detector(0.4)
        get_ocr()
        get_fast_ocr()
    print(f"Bộ phát hiện: YOLOv8s {WEIGHTS} · OCR: {OCR_ENGINES[OCR_ENGINE]}")
    print(f"Mở trình duyệt: http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
