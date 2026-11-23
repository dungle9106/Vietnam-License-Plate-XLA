"""Minh hoạ các bước xử lý ảnh và thống kê dữ liệu bằng Matplotlib (cho báo cáo và ứng dụng).

Các hàm trả về matplotlib.figure.Figure (không dùng pyplot) nên vừa lưu được ra file,
vừa nhúng được vào cửa sổ Tkinter.
"""
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from .io import IMG_EXTS
from .ocr import preprocess_steps, split_lines, trim_to_chars


def _grid(images, ncols, title=None):
    nrows = math.ceil(len(images) / ncols)
    fig = Figure(figsize=(3.4 * ncols, 2.4 * nrows), layout="constrained")
    for i, (name, img) in enumerate(images.items(), 1):
        ax = fig.add_subplot(nrows, ncols, i)
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    return fig


def detection_figure(img, result_img, crops=()):
    """Kết quả phát hiện của YOLOv8s: ảnh gốc, ảnh có khung biển số và các vùng biển đã cắt."""
    steps = {"Ảnh gốc": img, "YOLOv8s phát hiện biển số": result_img}
    for i, c in enumerate(crops, 1):
        steps[f"Vùng biển {i}"] = c
    return _grid(steps, min(4, len(steps)), "Giai đoạn 1: phát hiện vùng biển số (YOLOv8s)")


def ocr_steps_figure(crop, ocr_result=None):
    """Các bước tiền xử lý trước OCR: xám -> phóng to -> CLAHE -> bilateral -> Otsu -> tách dòng."""
    steps = {"Vùng biển số": crop, **preprocess_steps(crop)}
    for i, part in enumerate(split_lines(steps["Lọc bilateral"]), 1):
        steps[f"Dòng {i} (cắt sát ký tự)"] = trim_to_chars(part)
    title = None
    if ocr_result is not None:
        title = f"OCR: '{ocr_result.raw}'  →  hậu xử lý: '{ocr_result.display}'"
    return _grid(steps, 4, title)


def _read_boxes(lbl):
    if not lbl.exists():
        return []
    rows = [line.split() for line in lbl.read_text(encoding="utf-8").splitlines()]
    return [tuple(float(x) for x in r[1:5]) for r in rows if len(r) == 5]


def dataset_stats_figure(data_dir):
    """Thống kê tập dữ liệu YOLO: số ảnh/biển theo tập, số biển mỗi ảnh,
    tỉ lệ khung (phân biệt biển 1 dòng / 2 dòng), kích thước tương đối, vị trí tâm."""
    from PIL import Image  # chỉ đọc header ảnh -> nhanh hơn cv2 nhiều

    data_dir = Path(data_dir)
    counts, per_img, aspects, sizes, centers = {}, Counter(), [], [], []
    for split in ("train", "val", "test"):
        img_dir = data_dir / "images" / split
        if not img_dir.exists():
            continue
        n_img = n_box = 0
        for p in img_dir.iterdir():
            if p.suffix.lower() not in IMG_EXTS:
                continue
            boxes = _read_boxes(data_dir / "labels" / split / f"{p.stem}.txt")
            n_img += 1
            n_box += len(boxes)
            per_img[min(len(boxes), 5)] += 1
            if not boxes:
                continue
            with Image.open(p) as im:
                W, H = im.size
            for xc, yc, w, h in boxes:
                aspects.append(w * W / max(h * H, 1e-6))
                sizes.append(math.sqrt(w * h))
                centers.append((xc, yc))
        counts[split] = (n_img, n_box)

    fig = Figure(figsize=(14, 8), layout="constrained")
    fig.suptitle(f"Thống kê dữ liệu: {data_dir.name}")
    ax = fig.add_subplot(2, 3, 1)
    x = np.arange(len(counts))
    ax.bar(x - 0.2, [c[0] for c in counts.values()], 0.4, label="Ảnh")
    ax.bar(x + 0.2, [c[1] for c in counts.values()], 0.4, label="Biển số")
    ax.set_xticks(x, list(counts))
    ax.set_title("Số lượng theo tập")
    ax.legend()

    ax = fig.add_subplot(2, 3, 2)
    ks = sorted(per_img)
    ax.bar([str(k) if k < 5 else "5+" for k in ks], [per_img[k] for k in ks])
    ax.set_title("Số biển số trong mỗi ảnh")

    ax = fig.add_subplot(2, 3, 3)
    ax.hist(np.clip(aspects, 0, 8), bins=50)
    ax.axvline(2.5, color="red", ls="--", label="ngưỡng 1 dòng / 2 dòng")
    ax.set_title("Tỉ lệ rộng/cao của biển số")
    ax.legend()

    ax = fig.add_subplot(2, 3, 4)
    ax.hist(sizes, bins=50)
    ax.set_title("Kích thước biển (√(w·h), tỉ lệ so với ảnh)")

    ax = fig.add_subplot(2, 3, 5)
    if centers:
        c = np.array(centers)
        ax.hist2d(c[:, 0], c[:, 1], bins=30, range=[[0, 1], [0, 1]], cmap="viridis")
    ax.invert_yaxis()
    ax.set_title("Phân bố vị trí tâm biển số")

    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    n_1 = sum(a >= 2.5 for a in aspects)
    lines = [f"{s}: {n} ảnh, {b} biển số" for s, (n, b) in counts.items()]
    lines += ["", f"Biển 1 dòng (tỉ lệ ≥ 2.5): {n_1}", f"Biển 2 dòng (tỉ lệ < 2.5): {len(aspects) - n_1}"]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=11)
    return fig


def ocr_errors_figure(errors):
    """Phân tích lỗi OCR từ runs/ocr_errors.csv (các dòng có 'gt', 'pred').

    Lỗi cùng độ dài -> đếm từng cặp ký tự bị nhầm; khác độ dài -> thừa/thiếu ký tự.
    """
    pairs, kinds = Counter(), Counter()
    for r in errors:
        gt, pred = r["gt"], r["pred"]
        if not pred:
            kinds["Không đọc được"] += 1
        elif len(pred) != len(gt):
            kinds["Thừa ký tự" if len(pred) > len(gt) else "Thiếu ký tự"] += 1
        else:
            kinds["Nhầm ký tự"] += 1
            pairs.update(f"{g}→{p}" for g, p in zip(gt, pred) if g != p)

    fig = Figure(figsize=(12, 4.5), layout="constrained")
    ax = fig.add_subplot(1, 2, 1)
    ax.bar(list(kinds), list(kinds.values()), color="tab:orange")
    ax.set_title(f"Loại lỗi OCR ({len(errors)} biển đọc sai)")
    ax = fig.add_subplot(1, 2, 2)
    top = pairs.most_common(15)[::-1]
    ax.barh([k for k, _ in top], [v for _, v in top])
    ax.set_title("Các cặp ký tự hay bị nhầm (đúng → đọc thành)")
    return fig
