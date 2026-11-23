"""Ứng dụng desktop nhận diện biển số xe (Tkinter).

    python app.py
Chức năng: mở ảnh / cả thư mục ảnh / video / webcam, hiển thị khung bao biển số kèm chuỗi ký tự,
liệt kê kết quả (loại biển, tỉnh), lưu ảnh kết quả, xem các bước xử lý ảnh,
xuất kết quả ra Excel (.xlsx) hoặc CSV.
Cấu hình cố định: YOLOv8s (weights/best.pt, tự huấn luyện) phát hiện biển số + PaddleOCR đọc ký tự.
"""
import csv
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from lpr.io import DEFAULT_WEIGHTS, imread, imwrite

DETECTOR_NAME, OCR_NAME = "YOLOv8s", "PaddleOCR"
APP_TITLE = "Nhận diện biển số xe - Nhóm 08"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.loading = False
        self.geometry("1360x780")
        self.minsize(1000, 600)

        self.pipeline = None
        self.fast_ocr = None         # PaddleOCR bản nhanh cho video (tải khi cần)
        self.current = None          # ảnh kết quả đang hiển thị (BGR)
        self.last = None             # (đường dẫn, ảnh gốc, kết quả) của lần nhận diện ảnh gần nhất
        self.batch = []              # list ImageRecord khi đọc cả thư mục
        self.stop_event = threading.Event()
        self.worker = None
        self.frames = queue.Queue(maxsize=2)
        self.events = queue.Queue()  # luồng nền -> giao diện (Tkinter chỉ an toàn ở luồng chính)
        self.seen = {}               # track_id -> (PlateResult, thời điểm thấy lần đầu)

        self._build_ui()
        self.after(30, self._poll)
        self._load_model(DEFAULT_WEIGHTS)

    # ---------- Giao diện ----------
    def _build_ui(self):
        bar = ttk.Frame(self, padding=(6, 6, 6, 0))
        bar.pack(side=tk.TOP, fill=tk.X)
        for text, cmd in [("Chọn mô hình", self.choose_model), ("Mở ảnh", self.open_image),
                          ("Mở thư mục", self.open_folder), ("Mở video", self.open_video),
                          ("Webcam", self.open_webcam), ("Dừng", self.stop), ("Lưu ảnh", self.save),
                          ("Xuất Excel", self.export_excel), ("Xuất CSV", self.export_csv),
                          ("Các bước xử lý", self.show_steps)]:
            ttk.Button(bar, text=text, command=cmd).pack(side=tk.LEFT, padx=2)

        opts = ttk.Frame(self, padding=6)
        opts.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(opts, text=f"Mô hình: {DETECTOR_NAME} + {OCR_NAME}").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Label(opts, text="Ngưỡng tin cậy:").pack(side=tk.LEFT, padx=(16, 3))
        self.conf_var = tk.DoubleVar(value=0.4)
        ttk.Spinbox(opts, from_=0.1, to=0.9, increment=0.05, width=5,
                    textvariable=self.conf_var).pack(side=tk.LEFT)
        self.multi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Đọc nhiều biển số trong 1 ảnh",
                        variable=self.multi_var).pack(side=tk.LEFT, padx=(16, 3))
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Quét cả thư mục con",
                        variable=self.recursive_var).pack(side=tk.LEFT, padx=(8, 3))
        self.progress = ttk.Progressbar(opts, length=180, mode="determinate")
        self.progress.pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Label(body, bg="#202020")
        body.add(self.canvas, weight=3)

        side = ttk.Frame(body, padding=6)
        body.add(side, weight=2)
        ttk.Label(side, text="Kết quả nhận diện", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        cols = [("file", "Ảnh", 140), ("plate", "Biển số", 110), ("kind", "Loại biển", 170),
                ("prov", "Tỉnh/TP", 100), ("det", "Phát hiện", 62), ("ocr", "OCR", 50)]
        frame = ttk.Frame(side)
        frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.tree = ttk.Treeview(frame, columns=[c[0] for c in cols], show="headings")
        for col, title, w in cols:
            self.tree.heading(col, text=title)
            self.tree.column(col, width=w, anchor="w" if col in ("file", "kind", "prov") else "center")
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.tag_configure("bad", foreground="#d07000")
        self.tree.tag_configure("none", foreground="#888888")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.summary = ttk.Label(side, text="", anchor="w")
        self.summary.pack(fill=tk.X)

        self.status = ttk.Label(self, relief=tk.SUNKEN, anchor="w", padding=4)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _status(self, text):
        self.status.config(text=text)

    def _set_loading(self, on):
        """Hiện trạng thái tải ở thanh tiêu đề (dòng trạng thái dưới đáy có thể bị taskbar che)."""
        self.loading = on
        self.title(APP_TITLE + (" - Đang tải mô hình, vui lòng chờ..." if on else ""))
        self.config(cursor="watch" if on else "")

    def _show(self, bgr):
        self.current = bgr
        cw, ch = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)
        h, w = bgr.shape[:2]
        s = min(cw / w, ch / h)
        rgb = cv2.cvtColor(cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s)))), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.config(image=self._photo)

    def _row(self, iid, name, r):
        """Thêm 1 dòng vào bảng. r=None -> ảnh không có biển số."""
        if r is None:
            self.tree.insert("", tk.END, iid=iid, values=(name, "(không thấy)", "", "", "", ""),
                             tags=("none",))
            return
        self.tree.insert("", tk.END, iid=iid,
                         values=(name, r.display or "?", r.kind, r.province,
                                 f"{r.det_conf:.2f}", f"{r.ocr_conf:.2f}"),
                         tags=() if r.valid else ("bad",))

    def _clear_table(self):
        self.tree.delete(*self.tree.get_children())
        self.summary.config(text="")

    # ---------- Mô hình ----------
    def choose_model(self):
        path = filedialog.askopenfilename(title="Chọn file trọng số YOLOv8",
                                          filetypes=[("YOLOv8 weights", "*.pt")])
        if path:
            self._load_model(Path(path))

    def _load_model(self, path):
        if not Path(path).exists():
            self._status(f"Không thấy mô hình {path} - bấm 'Chọn mô hình' để chọn file best.pt")
            return
        self._status(f"Đang tải {DETECTOR_NAME} ({Path(path).name}) và {OCR_NAME} (lần đầu sẽ tải mô hình OCR)...")
        self._set_loading(True)
        conf = float(self.conf_var.get())
        old_ocr = self.pipeline.ocr if self.pipeline is not None else None

        def job():
            try:
                from lpr.pipeline import LPRPipeline
                self.events.put(("loaded", LPRPipeline(path, det_conf=conf, ocr=old_ocr), path))
            except Exception as e:  # noqa: BLE001
                self.events.put(("error", "Lỗi tải mô hình", str(e)))
        threading.Thread(target=job, daemon=True).start()

    def _on_loaded(self, pipe, path):
        self._set_loading(False)
        self.pipeline = pipe
        self._status(f"Sẵn sàng - {DETECTOR_NAME}: {path} + {OCR_NAME}")

    def _ready(self):
        if self.pipeline is None:
            if self.loading:
                messagebox.showinfo("Đang tải mô hình",
                                    "Mô hình YOLO và OCR đang được tải (khoảng 10-30 giây, lần đầu lâu hơn).\n"
                                    "Khi thanh tiêu đề hết dòng 'Đang tải mô hình' thì bấm lại.")
            else:
                messagebox.showwarning("Chưa có mô hình",
                                       "Chưa tải được mô hình. Bấm 'Chọn mô hình' để chọn file best.pt.")
            return False
        if self.worker and self.worker.is_alive():
            self.stop()
        self.pipeline.detector.conf = float(self.conf_var.get())
        return True

    # ---------- Ảnh ----------
    def open_image(self):
        if not self._ready():
            return
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        img = imread(path)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh.")
            return
        self._status("Đang nhận diện...")
        self.update_idletasks()
        t0 = time.perf_counter()
        from lpr.pipeline import draw_results
        results = self.pipeline.process_image(img, multi=self.multi_var.get())
        dt = (time.perf_counter() - t0) * 1000
        self.last, self.batch, self.seen = (path, img, results), [], {}
        self._show(draw_results(img, results))
        self._clear_table()
        name = Path(path).name
        for i, r in enumerate(results):
            self._row(f"0:{i}", name, r)
        if not results:
            self._row("0:-", name, None)
        plates = [r.display or "?" for r in results]
        self.summary.config(text=f"Mảng biển số: {plates}")
        self._status(f"{name}: {len(results)} biển số - {dt:.0f} ms "
                     f"({DETECTOR_NAME} + {OCR_NAME})")

    # ---------- Thư mục ảnh ----------
    def open_folder(self):
        if not self._ready():
            return
        folder = filedialog.askdirectory(title="Chọn thư mục ảnh")
        if not folder:
            return
        from lpr.batch import list_images
        paths = list_images(folder, recursive=self.recursive_var.get())
        if not paths:
            messagebox.showinfo("Mở thư mục", "Thư mục không có ảnh (.jpg, .png, .bmp, .webp).")
            return
        self.stop_event.clear()
        self.batch, self.last, self.seen = [], None, {}
        self._batch_root = Path(folder)
        self._clear_table()
        self.progress.config(maximum=len(paths), value=0)
        multi = self.multi_var.get()

        def job():
            from lpr.batch import process_paths
            from lpr.pipeline import draw_results
            t0 = time.perf_counter()

            def progress(i, n, rec):
                img = imread(rec.path)
                vis = draw_results(img, rec.plates) if img is not None else None
                self.events.put(("batch_item", i, n, rec, vis))
            recs = process_paths(self.pipeline, paths, multi=multi, on_progress=progress,
                                 should_stop=self.stop_event.is_set)
            self.events.put(("batch_done", len(recs), len(paths), time.perf_counter() - t0))
        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()
        self._status(f"Đang đọc {len(paths)} ảnh trong {folder}... bấm 'Dừng' để kết thúc")

    def _on_batch_item(self, i, n, rec, vis):
        self.batch.append(rec)
        k = len(self.batch) - 1
        try:
            name = str(Path(rec.path).relative_to(self._batch_root))
        except ValueError:
            name = rec.name
        for j, r in enumerate(rec.plates):
            self._row(f"{k}:{j}", name, r)
        if not rec.plates:
            self._row(f"{k}:-", name, None)
        self.tree.yview_moveto(1.0)
        if vis is not None:
            self._show(vis)
        self.progress.config(value=i)
        n_plates = sum(len(r.plates) for r in self.batch)
        self.summary.config(text=f"{len(self.batch)}/{n} ảnh - {n_plates} biển số")
        self._status(f"[{i}/{n}] {rec.name}: {rec.plate_list()} ({rec.ms:.0f} ms)")

    def _on_batch_done(self, done, total, secs):
        n_plates = sum(len(r.plates) for r in self.batch)
        n_valid = sum(p.valid for r in self.batch for p in r.plates)
        msg = (f"Xong {done}/{total} ảnh trong {secs:.1f} s - {n_plates} biển số "
               f"({n_valid} đúng định dạng). Bấm 'Xuất Excel' để lưu kết quả.")
        self.summary.config(text=msg)
        self._status(msg)

    def _on_select(self, _event=None):
        """Bấm vào 1 dòng kết quả thư mục -> hiện lại ảnh đó với khung biển số."""
        sel = self.tree.selection()
        if not sel or not self.batch or (self.worker and self.worker.is_alive()):
            return
        k = int(sel[0].split(":")[0])
        if k >= len(self.batch):
            return
        rec = self.batch[k]
        img = imread(rec.path)
        if img is not None:
            from lpr.pipeline import draw_results
            self._show(draw_results(img, rec.plates))
            self.last = (rec.path, img, rec.plates)
            self._status(f"{rec.name}: {rec.plate_list()}")

    def show_steps(self):
        """Cửa sổ minh hoạ các bước xử lý (Matplotlib) cho ảnh vừa nhận diện."""
        if self.last is None:
            messagebox.showinfo("Các bước xử lý", "Hãy mở một ảnh (hoặc chọn 1 dòng kết quả) trước.")
            return
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from lpr.pipeline import crop_plate, draw_results
        from lpr.visualize import detection_figure, ocr_steps_figure

        _, img, results = self.last
        win = tk.Toplevel(self)
        win.title("Các bước xử lý ảnh")
        win.geometry("1300x700")
        tabs = ttk.Notebook(win)
        tabs.pack(fill=tk.BOTH, expand=True)
        figs = [("Phát hiện (YOLOv8s)", detection_figure(img, draw_results(img, results),
                                                        [crop_plate(img, r.box) for r in results]))]
        for i, r in enumerate(results, 1):
            crop = crop_plate(img, r.box)
            figs.append((f"Biển {i}: {r.display or '?'}", ocr_steps_figure(crop, self.pipeline.ocr.read(crop))))
        for title, fig in figs:
            frame = ttk.Frame(tabs)
            tabs.add(frame, text=title)
            FigureCanvasTkAgg(fig, master=frame).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- Video / webcam ----------
    def open_video(self):
        if not self._ready():
            return
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if path:
            self._start_stream(path)

    def open_webcam(self):
        if self._ready():
            self._start_stream(0)

    def _start_stream(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", f"Không mở được nguồn video: {source}")
            return
        self.stop_event.clear()
        self.seen, self.last, self.batch = {}, None, []
        self.worker = threading.Thread(target=self._stream_loop, args=(cap,), daemon=True)
        self.worker.start()
        self._status("Đang xử lý video... bấm 'Dừng' để kết thúc")

    def _stream_loop(self, cap):
        from lpr.pipeline import LPRPipeline, VideoProcessor, draw_results
        # Video/webcam: PaddleOCR bản nhanh (phát hiện dòng mobile) - bản chính xác ~2 s/biển làm video giật
        if self.fast_ocr is None:
            from lpr.paddle_ocr import make_fast
            self.fast_ocr = make_fast(self.pipeline.ocr)
        from lpr.detector import FAST_IMGSZ
        det = self.pipeline.detector
        det = det.with_imgsz(FAST_IMGSZ) if hasattr(det, "with_imgsz") else det  # YOLO 640 cho video
        vp = VideoProcessor(LPRPipeline.from_parts(det, self.fast_ocr))
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            results = vp.process_frame(frame)
            vis = draw_results(frame, results, vp.fps)
            for r in results:
                if r.text:
                    first = self.seen.get(r.track_id, (None, datetime.now()))[1]
                    self.seen[r.track_id] = (r, first)
            try:
                self.frames.get_nowait()  # bỏ frame cũ, luôn hiển thị frame mới nhất
            except queue.Empty:
                pass
            self.frames.put(vis)
        cap.release()
        self.events.put(("status", f"Đã dừng video - {len(self.seen)} biển số."))

    def _poll(self):
        while not self.events.empty():
            kind, *data = self.events.get_nowait()
            if kind == "loaded":
                self._on_loaded(*data)
            elif kind == "batch_item":
                self._on_batch_item(*data)
            elif kind == "batch_done":
                self._on_batch_done(*data)
            elif kind == "error":
                self._set_loading(False)
                self._status(data[0])
                messagebox.showerror(*data)
            else:
                self._status(data[0])
        try:
            frame = self.frames.get_nowait()
            self._show(frame)
            self._clear_table()
            for tid, (r, _) in sorted(self.seen.items()):
                self._row(f"v:{tid}", f"Xe #{tid}", r)
        except queue.Empty:
            pass
        self.after(30, self._poll)

    def stop(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=5)

    def save(self):
        if self.current is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if path and imwrite(path, self.current):
            self._status(f"Đã lưu {path}")

    # ---------- Xuất kết quả ----------
    def _records(self):
        """Kết quả hiện có dưới dạng list ImageRecord (thư mục, 1 ảnh, hoặc video)."""
        from lpr.batch import ImageRecord
        if self.batch:
            return self.batch
        if self.seen:
            return [ImageRecord(f"video - xe #{tid} ({t:%H:%M:%S})", [r])
                    for tid, (r, t) in sorted(self.seen.items())]
        if self.last:
            return [ImageRecord(str(self.last[0]), list(self.last[2]))]
        return []

    def export_excel(self):
        recs = self._records()
        if not recs:
            messagebox.showinfo("Xuất Excel", "Chưa có kết quả nhận diện.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            initialfile=f"bien_so_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        from lpr.batch import export_excel
        meta = {"Bộ phát hiện": DETECTOR_NAME, "Engine OCR": OCR_NAME,
                "Chế độ": "Nhiều biển / ảnh" if self.multi_var.get() else "1 biển / ảnh"}
        try:
            export_excel(recs, path, meta=meta)
        except PermissionError:
            messagebox.showerror("Xuất Excel", "Không ghi được file - file đang mở trong Excel?")
            return
        n = sum(len(r.plates) for r in recs)
        self._status(f"Đã xuất {len(recs)} ảnh / {n} biển số -> {path}")

    def export_csv(self):
        """Xuất danh sách biển số (vd nhật ký xe ra vào) ra CSV, mở được bằng Excel."""
        recs = self._records()
        if not recs:
            messagebox.showinfo("Xuất CSV", "Chưa có kết quả nhận diện.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["anh", "bien_so", "loai", "tinh", "mau", "conf_phat_hien", "conf_ocr",
                        "dung_dinh_dang", "thoi_diem"])
            for rec in recs:
                for r in rec.plates:
                    w.writerow([rec.name, r.display, r.kind, r.province, r.color,
                                f"{r.det_conf:.3f}", f"{r.ocr_conf:.3f}", r.valid, now])
        self._status(f"Đã xuất CSV -> {path}")

    def destroy(self):
        self.stop()
        super().destroy()


if __name__ == "__main__":
    App().mainloop()
