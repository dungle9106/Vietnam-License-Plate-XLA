"""Các độ đo: IoU, khoảng cách Levenshtein, CER."""


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def levenshtein(s, t):
    prev = list(range(len(t) + 1))
    for i, cs in enumerate(s, 1):
        cur = [i]
        for j, ct in enumerate(t, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cs != ct)))
        prev = cur
    return prev[-1]


def match_boxes(preds, gts, thr=0.5):
    """Ghép tham lam box dự đoán với box ground-truth theo IoU giảm dần.

    Trả về dict {chỉ số gt: chỉ số pred}.
    """
    pairs = sorted(
        ((iou(p, g), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
        reverse=True,
    )
    used_p, matched = set(), {}
    for score, pi, gi in pairs:
        if score < thr:
            break
        if pi in used_p or gi in matched:
            continue
        used_p.add(pi)
        matched[gi] = pi
    return matched
