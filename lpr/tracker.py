"""Tracker IoU đơn giản cho video: giữ ID biển số qua các frame và bỏ phiếu chuỗi OCR.

Đọc OCR trên nhiều frame rồi chọn chuỗi có tổng điểm cao nhất giúp kết quả
ổn định hơn nhiều so với tin vào một frame duy nhất (bị mờ, nghiêng...).
"""
from collections import defaultdict

from .metrics import iou


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _size(b):
    return max(b[2] - b[0], 1e-6), max(b[3] - b[1], 1e-6)


class Track:
    def __init__(self, tid, box):
        self.id, self.box = tid, box
        self.vel = (0.0, 0.0)  # độ dời tâm box giữa 2 lần cập nhật (px / khung được xử lý)
        self.hits, self.misses, self.ocr_count = 1, 0, 0
        self.votes = defaultdict(float)
        self.display = {}
        self.info = {}      # chuỗi -> OCRResult gần nhất (màu, loại biển, tỉnh)

    def add_vote(self, ocr):
        self.ocr_count += 1
        if not ocr.text:
            return
        self.votes[ocr.text] += ocr.conf * (2.0 if ocr.valid else 0.5)
        self.display[ocr.text] = ocr.display
        self.info[ocr.text] = ocr

    def best(self):
        """Trả về (chuỗi, chuỗi hiển thị, tỉ lệ phiếu của chuỗi thắng)."""
        if not self.votes:
            return "", "", 0.0
        text = max(self.votes, key=self.votes.get)
        share = self.votes[text] / max(sum(self.votes.values()), 1e-6)
        return text, self.display[text], share


    def predicted(self):
        """Box dự đoán ở khung kế tiếp theo vận tốc (giả định chuyển động đều)."""
        k = self.misses + 1
        dx, dy = self.vel[0] * k, self.vel[1] * k
        x1, y1, x2, y2 = self.box[:4]
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


class IoUTracker:
    """Ghép box giữa các khung: ưu tiên IoU với vị trí dự đoán; không chồng nhau thì ghép theo
    khoảng cách tâm (<= max_dist lần cạnh lớn của box) nếu kích thước tương đương.

    Video/webcam trên web chỉ xử lý ~3 khung/s trong khi xe vẫn chạy, nên giữa 2 khung được xử lý
    biển dịch đi xa hơn cả bề rộng của nó -> chỉ dùng IoU thì 1 biển bị tách thành nhiều xe.
    """

    def __init__(self, iou_thr=0.3, max_misses=15, max_dist=2.0):
        self.iou_thr, self.max_misses, self.max_dist = iou_thr, max_misses, max_dist
        self.tracks, self._next = {}, 1

    def _score(self, b, t):
        """Điểm ghép box b với track t (lớn hơn = khớp hơn), None nếu không thể là cùng biển."""
        p = t.predicted()
        o = max(iou(b, p), iou(b, t.box))
        if o >= self.iou_thr:
            return (1, o)
        (bw, bh), (tw, th) = _size(b), _size(t.box)
        if not (0.5 <= bw / tw <= 2.0 and 0.5 <= bh / th <= 2.0):
            return None
        (bx, by), (px, py) = _center(b), _center(p)
        d = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5 / max(tw, th)
        return (0, -d) if d <= self.max_dist else None

    def update(self, boxes):
        """Gán mỗi box một track id. Trả về list id theo đúng thứ tự boxes."""
        pairs = []
        for bi, b in enumerate(boxes):
            for tid, t in self.tracks.items():
                s = self._score(b, t)
                if s is not None:
                    pairs.append((s, bi, tid))
        pairs.sort(reverse=True)
        ids, used = [None] * len(boxes), set()
        for _, bi, tid in pairs:
            if ids[bi] is None and tid not in used:
                ids[bi] = tid
                used.add(tid)
                t = self.tracks[tid]
                (ox, oy), (nx, ny) = _center(t.box), _center(boxes[bi])
                k = t.misses + 1
                t.vel = ((nx - ox) / k, (ny - oy) / k)
                t.box, t.hits, t.misses = boxes[bi], t.hits + 1, 0
        for bi, b in enumerate(boxes):
            if ids[bi] is None:
                ids[bi] = self._next
                self.tracks[self._next] = Track(self._next, b)
                used.add(self._next)
                self._next += 1
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid].misses += 1
                if self.tracks[tid].misses > self.max_misses:
                    del self.tracks[tid]
        return ids
