from collections import defaultdict

import numpy as np

from .predict import CATEGORIES


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


def _pred_char_probs(pred_rec, resp_len):
    """Per-char pred prob: use stored char_probs if present, else reconstruct from spans."""
    cp = pred_rec.get("char_probs")
    if cp:
        arr = list(cp)[:resp_len]
        arr += [0.0] * (resp_len - len(arr))
        return arr
    arr = [0.0] * resp_len
    for sp in pred_rec.get("pred_labels", []):
        a = max(0, min(int(sp["start"]), resp_len))
        b = max(0, min(int(sp["end"]), resp_len))
        pr = float(sp.get("prob", 1.0))
        for i in range(a, b):
            if pr > arr[i]:
                arr[i] = pr
    return arr


def evaluate(gold_items, pred_by_id):
    """Per-language report: mean IoU, per-label IoU, calibration, trivial baselines."""
    by_lang = defaultdict(list)
    for it in gold_items:
        by_lang[it.language].append(it)

    report = {}
    for lang, items in by_lang.items():
        ious, gp_all, pp_all = [], [], []
        label_inter = defaultdict(int)
        label_union = defaultdict(int)
        for it in items:
            rl = len(it.response)
            pred = pred_by_id.get(it.id, {"pred_labels": []})
            pred_spans = pred.get("pred_labels", [])
            ious.append(char_iou(it.labels, pred_spans, rl))
            gp_all.extend(gold_char_probs(it.labels, rl))
            pp_all.extend(_pred_char_probs(pred, rl))
            for lab in CATEGORIES:
                g = _char_set([s for s in it.labels if s.get("label") == lab], rl)
                p = _char_set([s for s in pred_spans if s.get("label") == lab], rl)
                label_inter[lab] += len(g & p)
                label_union[lab] += len(g | p)
        tb = trivial_baselines(items)
        cal = calibration(gp_all, pp_all)
        per_label = {lab: (label_inter[lab] / label_union[lab] if label_union[lab] else 1.0)
                     for lab in CATEGORIES}
        report[lang] = {
            "n": len(items),
            "iou": sum(ious) / len(ious),
            "per_label_iou": per_label,
            "pearson": cal["pearson"],
            "spearman": cal["spearman"],
            "predict_nothing_iou": tb["predict_nothing_iou"],
            "predict_all_iou": tb["predict_all_iou"],
        }
    return report
