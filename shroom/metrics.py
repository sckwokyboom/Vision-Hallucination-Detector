import numpy as np


def _char_set(spans, resp_len):
    s = set()
    for sp in spans:
        a = max(0, min(int(sp["start"]), resp_len))
        b = max(0, min(int(sp["end"]), resp_len))
        s.update(range(a, b))
    return s


def char_iou(gold_spans, pred_spans, resp_len):
    g = _char_set(gold_spans, resp_len)
    p = _char_set(pred_spans, resp_len)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    return len(g & p) / len(g | p)


def gold_char_probs(spans, resp_len):
    probs = [0.0] * resp_len
    for sp in spans:
        a = max(0, min(int(sp["start"]), resp_len))
        b = max(0, min(int(sp["end"]), resp_len))
        pr = float(sp.get("prob", 1.0))
        for i in range(a, b):
            if pr > probs[i]:
                probs[i] = pr
    return probs


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.flatnonzero(obs), len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    return pearson(_rankdata(x), _rankdata(y))


def calibration(gold_probs, pred_probs):
    return {"pearson": pearson(gold_probs, pred_probs),
            "spearman": spearman(gold_probs, pred_probs)}


def trivial_baselines(items):
    """IoU of the two degenerate systems: predict nothing / predict everything."""
    nothing, allh = [], []
    for it in items:
        rl = len(it.response)
        nothing.append(char_iou(it.labels, [], rl))
        full = [{"start": 0, "end": rl}]
        allh.append(char_iou(it.labels, full, rl))
    n = max(1, len(items))
    return {"predict_nothing_iou": sum(nothing) / n,
            "predict_all_iou": sum(allh) / n}
