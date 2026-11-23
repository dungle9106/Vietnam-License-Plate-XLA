"""Xuất dữ liệu tinh chỉnh (fine-tune) mô hình NHẬN DẠNG chữ của PaddleOCR từ biển đã gán nhãn.

Mô hình nhận dạng của Paddle đọc 1 DÒNG chữ mỗi lần, nên mỗi biển được cắt thành ảnh từng dòng
(bằng chính mô hình phát hiện dòng mà hệ thống dùng khi chạy - lpr/paddle_ocr.py: line_regions),
rồi chia đáp án của biển cho từng dòng:
    biển 1 dòng  30E-931.05      -> "30E93105"
    biển 2 dòng  29-AB / 901.75  -> "29AB" + "90175"
Ranh giới seri / dãy số của biển 2 dòng được chọn sao cho khớp nhất với chữ PaddleOCR đọc được
trên từng dòng (khoảng cách Levenshtein), vì 59X1-1234 và 59X-11234 đều có thể.

Chỉ dùng biển ĐÃ KIỂM TRA (checked=1) thuộc tập train / val (cắt từ data/lp_full/images/train|val).
Biển tập test KHÔNG BAO GIỜ được xuất - chúng dùng để đo độ chính xác sau khi tinh chỉnh.

    python scripts/export_paddle_rec.py            # -> data/paddle_rec/{train,val}.txt + images/
    python scripts/export_paddle_rec.py --zip      # đóng gói data/paddle_rec.zip để tải lên Colab

Định dạng nhãn của PaddleOCR: mỗi dòng "đường/dẫn/ảnh.jpg<TAB>NHÃN".
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_ocr import label_split, load_labels  # noqa: E402
from lpr.io import imread, imwrite  # noqa: E402
from lpr.metrics import levenshtein  # noqa: E402
from lpr.plate_format import normalize  # noqa: E402
from lpr.rectify import order_corners  # noqa: E402


def line_image(crop, polys):
    """Ảnh 1 dòng chữ: nắn thẳng hình chữ nhật xoay nhỏ nhất bao mọi vùng chữ của dòng, nới 8%."""
    pts = np.concatenate(polys).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    # Đưa góc về khoảng (-45°, 45°] quanh phương ngang. KHÔNG ép cạnh dài nằm ngang: vùng chữ cao hơn
    # rộng (biển 2 dòng bị gộp) sẽ bị xoay nhầm 90°.
    while ang > 45:
        ang -= 90
        w, h = h, w
    while ang <= -45:
        ang += 90
        w, h = h, w
    box = cv2.boxPoints(((cx, cy), (w * 1.04 + 0.08 * h, h * 1.08), ang))
    src = order_corners(box).astype(np.float32)
    W, H = int(round(w * 1.04 + 0.08 * h)), int(round(h * 1.08))
    if W < 4 or H < 4:
        return None
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return cv2.warpPerspective(crop, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# Ảnh 1 dòng chữ thật (7-9 ký tự) có rộng/cao >= ~2.3; vùng mô hình phát hiện gộp cả 2 dòng của biển
# 2 dòng thì gần vuông (đo trên dữ liệu train: 1.05-1.8). Ngưỡng nằm giữa 2 nhóm.
SQUARE_RATIO = 2.2


def split_two_lines(img):
    """Tách ảnh chứa 2 dòng chữ tại khe trống ngang giữa 2 dòng (split_lines). [] nếu không tách được."""
    from lpr.ocr import preprocess_steps, split_lines
    gray = preprocess_steps(img)["Lọc bilateral"]  # đã chuẩn hoá chiều cao -> đổi toạ độ về ảnh gốc
    parts = split_lines(gray)
    if len(parts) != 2:
        return []
    cut = int(round(parts[0].shape[0] * img.shape[0] / gray.shape[0]))
    pad = max(1, int(0.04 * img.shape[0]))  # chồng lấn nhẹ để không cắt lẹm chân / đầu chữ
    top, bottom = img[:cut + pad], img[max(0, cut - pad):]
    return [top, bottom] if min(top.shape[0], bottom.shape[0]) >= 8 else []


def split_label(gt, preds):
    """Chia đáp án gt cho 2 dòng sao cho khớp nhất với chữ đọc được preds[0], preds[1].
    Dòng dưới phải là 4-5 chữ số. Trả về (dòng trên, dòng dưới) hoặc None."""
    best = None
    for k in range(2, len(gt) - 3):
        top, bottom = gt[:k], gt[k:]
        if not (bottom.isdigit() and 4 <= len(bottom) <= 5):
            continue
        d = levenshtein(normalize(preds[0]), top) + levenshtein(normalize(preds[1]), bottom)
        if best is None or d < best[0]:
            best = (d, top, bottom)
    return best[1:] if best else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="data/ocr_labels/labels.csv")
    ap.add_argument("--out", default="data/paddle_rec")
    ap.add_argument("--zip", action="store_true", help="Đóng gói thành <out>.zip để tải lên Google Colab")
    args = ap.parse_args()

    from lpr.paddle_ocr import PaddlePlateOCR
    ocr = PaddlePlateOCR()  # bản chính xác: phát hiện dòng server, giống khi chạy ảnh tĩnh
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    stats = Counter()
    for split in ("train", "val"):
        rows = load_labels(ROOT / args.labels, subset=split)
        entries = []
        for r in rows:
            crop = imread(r["crop"])
            gt = normalize(r["text"])
            if crop is None or not gt:
                continue
            regions = ocr.line_regions(crop)
            imgs = [line_image(crop, ps) for ps in regions]
            imgs = [im for im in imgs if im is not None]
            merged = len(imgs) == 1 and imgs[0].shape[1] / imgs[0].shape[0] < SQUARE_RATIO
            if merged:
                # Mô hình phát hiện gộp CẢ 2 dòng của biển 2 dòng thành 1 vùng (ảnh gần vuông) -> tách
                # theo khe trống giữa 2 dòng; để nguyên thì ảnh 2 dòng mang nhãn 1 dòng, dạy sai mô hình.
                imgs = split_two_lines(imgs[0])
                stats[f"{split}: tách vùng gộp 2 dòng"] += 1
            if len(imgs) == 1:
                labels = [gt]
            elif len(imgs) == 2:
                preds = [ocr._predict(im)[0] for im in imgs]
                parts = split_label(gt, preds)
                if parts is None:
                    stats[f"{split}: bỏ (không chia được nhãn 2 dòng)"] += 1
                    continue
                labels = list(parts)
                if merged:
                    # Tách từ vùng gộp kém tin cậy hơn (biển xoay mạnh, khe giữa 2 dòng không rõ):
                    # chỉ giữ khi chữ đọc được khớp tương đối với nhãn, tránh dạy mô hình bằng ảnh cắt sai.
                    dist = sum(levenshtein(normalize(p), lab) for p, lab in zip(preds, labels))
                    if dist > 0.5 * len(gt):
                        stats[f"{split}: bỏ (tách vùng gộp không khớp nhãn)"] += 1
                        continue
            else:
                stats[f"{split}: bỏ ({len(imgs)} dòng)"] += 1
                continue
            stem = Path(r["crop"]).stem
            for k, (im, lab) in enumerate(zip(imgs, labels)):
                if im.shape[0] > im.shape[1]:  # dòng chữ biển số không bao giờ cao hơn rộng: cắt hỏng
                    stats[f"{split}: bỏ ảnh dòng cao hơn rộng"] += 1
                    continue
                rel = f"images/{split}/{stem}_L{k}.jpg"
                imwrite(out / rel, im)
                entries.append(f"{rel}\t{lab}")
            stats[f"{split}: biển {len(imgs)} dòng"] += 1
        (out / f"{split}.txt").parent.mkdir(parents=True, exist_ok=True)
        (out / f"{split}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
        stats[f"{split}: TỔNG ảnh dòng"] = len(entries)
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"Đã ghi {out}")
    if args.zip:
        z = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"Đã đóng gói {z} -> tải file này lên Google Drive (thư mục gốc My Drive)")


if __name__ == "__main__":
    main()
