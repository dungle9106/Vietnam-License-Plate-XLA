"""Gộp các bộ dữ liệu (Kaggle + Roboflow, định dạng YOLO) và chia train/val/test = 70/20/10.

- Tự tìm file nhãn .txt tương ứng với mỗi ảnh (thư mục labels/ song song hoặc cùng thư mục).
- Nhãn polygon (segmentation) được chuyển thành bounding box.
- Mọi lớp được gộp thành 1 lớp duy nhất: license_plate.
- Chống rò rỉ dữ liệu: loại ảnh trùng (MD5) và giữ các bản augment của Roboflow
  (tên dạng 'abc_jpg.rf.<hash>.jpg') của cùng 1 ảnh gốc trong cùng 1 tập.

Ví dụ:
    python scripts/prepare_dataset.py --src raw/kaggle raw/roboflow --out data/lp_full
"""
import argparse
import hashlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lpr.io import IMG_EXTS  # noqa: E402


def find_label(img_path):
    parts = list(img_path.parts)
    candidates = [img_path.with_suffix(".txt")]
    for i in range(len(parts) - 1, -1, -1):  # thay thư mục 'images' gần nhất bằng 'labels'
        if parts[i].lower() == "images":
            candidates.insert(0, Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
            break
    return next((c for c in candidates if c.exists()), None)


def parse_label(path):
    """Đọc nhãn YOLO (bbox hoặc polygon) -> list (cls, xc, yc, w, h) chuẩn hoá."""
    boxes = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        cls, nums = int(float(v[0])), [float(x) for x in v[1:]]
        if len(nums) == 4:
            xc, yc, w, h = nums
        else:
            xs, ys = nums[0::2], nums[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        if w > 0 and h > 0:
            boxes.append((cls, xc, yc, w, h))
    return boxes


def group_key(img_path):
    """Các bản augment Roboflow của cùng ảnh gốc có chung tiền tố trước '.rf.'."""
    return re.split(r"\.rf\.", img_path.stem)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="Thư mục dataset nguồn (YOLO format)")
    ap.add_argument("--out", default="data/lp_full")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--classes", nargs="*", type=int, default=None,
                    help="Chỉ giữ các class id này (mặc định giữ tất cả, gộp thành 1 lớp)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    groups, seen, cls_count = defaultdict(list), set(), Counter()
    n_nolabel = n_dup = 0
    for si, src in enumerate(args.src):
        for img in sorted(Path(src).rglob("*")):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = find_label(img)
            if lbl is None:
                n_nolabel += 1
                continue
            digest = hashlib.md5(img.read_bytes()).hexdigest()
            if digest in seen:
                n_dup += 1
                continue
            seen.add(digest)
            boxes = parse_label(lbl)
            cls_count.update(b[0] for b in boxes)
            if args.classes is not None:
                boxes = [b for b in boxes if b[0] in args.classes]
            groups[f"{si}_{group_key(img)}"].append((img, boxes))

    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    n = len(keys)
    n_train, n_val = int(n * args.split[0]), int(n * args.split[1])
    split_of = {k: "train" for k in keys[:n_train]}
    split_of.update({k: "val" for k in keys[n_train:n_train + n_val]})
    split_of.update({k: "test" for k in keys[n_train + n_val:]})

    stats = Counter()
    for k, items in groups.items():
        sp = split_of[k]
        for img, boxes in items:
            name = f"{k}_{img.stem}"[:150]
            (out / "images" / sp).mkdir(parents=True, exist_ok=True)
            (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / "images" / sp / f"{name}{img.suffix.lower()}")
            (out / "labels" / sp / f"{name}.txt").write_text(
                "".join(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for _, xc, yc, w, h in boxes),
                encoding="utf-8")
            stats[sp] += 1
            stats[f"{sp}_boxes"] += len(boxes)

    # Không ghi 'path' (đường dẫn tuyệt đối của máy): Ultralytics tự lấy thư mục chứa data.yaml làm gốc,
    # nên thư mục dữ liệu chép sang máy khác / đưa lên GitHub vẫn dùng được.
    yaml = "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: license_plate\n"
    (out / "data.yaml").write_text(yaml, encoding="utf-8")

    print(f"Class id gốc trong nhãn: {dict(cls_count)}")
    if len(cls_count) > 1 and args.classes is None:
        print("  ! Có nhiều class id. Nếu có class không phải biển số (vd ký tự), dùng --classes.")
    print(f"Bỏ qua: {n_nolabel} ảnh không có nhãn, {n_dup} ảnh trùng lặp")
    for sp in ("train", "val", "test"):
        print(f"{sp:5s}: {stats[sp]:6d} ảnh, {stats[sp + '_boxes']:6d} biển số")
    print(f"Đã ghi {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
