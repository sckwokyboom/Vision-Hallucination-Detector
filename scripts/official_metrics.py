"""Full metric suite matching the organizers' scorer keys:
roc_auc, pr_auc, best_threshold, span_iou (at best threshold), calib_corr,
and class_roc (per-category one-vs-rest ROC AUC + macro_avg).

All are character-level. roc_auc / pr_auc / class_roc use the predicted per-character
probability (threshold-independent); span_iou is the char-IoU at the best threshold.

Usage:
  python scripts/official_metrics.py --gold splits/dev.en.jsonl \
      --pred gemma=preds/gemma.jsonl --pred hybrid=preds/hybrid.jsonl
Trivial predict-nothing / predict-everything baselines are added automatically.
Each --pred file has lines {id, pred_labels:[{start,end,prob,label}], char_probs:[...]}.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl                                       # noqa: E402
from shroom.predict import CATEGORIES                                    # noqa: E402
from shroom.metrics import _char_set, gold_char_probs, pearson           # noqa: E402


def load_by_id(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["id"]] = r
    return d


def cat_prob_array(spans, rl):
    arr = {c: np.zeros(rl) for c in CATEGORIES}
    for s in spans:
        c = s.get("label", "other")
        c = c if c in arr else "other"
        a = max(0, min(int(s["start"]), rl)); b = max(0, min(int(s["end"]), rl))
        p = float(s.get("prob", 1.0))
        arr[c][a:b] = np.maximum(arr[c][a:b], p)
    return arr


def safe_roc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")             # gold single-class -> undefined
    return roc_auc_score(y, s)


def suite(gold, get_pred):
    """get_pred(id) -> (spans, char_probs). Returns the metric dict."""
    gbin, pprob, gemp = [], [], []
    cg = {c: [] for c in CATEGORIES}
    cp_ = {c: [] for c in CATEGORIES}
    per_item = []
    for i, it in gold.items():
        rl = len(it.response)
        spans, probs = get_pred(i)
        probs = np.asarray((list(probs) + [0.0] * rl)[:rl], dtype=float)
        gset = _char_set(it.labels, rl)
        gb = np.zeros(rl);
        for k in gset: gb[k] = 1
        gbin.extend(gb.tolist()); pprob.extend(probs.tolist())
        gemp.extend(gold_char_probs(it.labels, rl))
        gcat = {c: _char_set([s for s in it.labels if s.get("label") == c], rl) for c in CATEGORIES}
        pcat = cat_prob_array(spans, rl)
        for c in CATEGORIES:
            gcarr = np.zeros(rl)
            for k in gcat[c]: gcarr[k] = 1
            cg[c].extend(gcarr.tolist()); cp_[c].extend(pcat[c].tolist())
        per_item.append((gset, rl, probs))

    gbin = np.asarray(gbin); pprob = np.asarray(pprob)
    roc = safe_roc(gbin, pprob)
    pr = average_precision_score(gbin, pprob) if gbin.max() > 0 else float("nan")
    calib = pearson(gemp, pprob)

    class_roc = {c: safe_roc(cg[c], cp_[c]) for c in CATEGORIES}
    macro = float(np.nanmean([v for v in class_roc.values()]))

    best_thr, best_iou = 0.0, -1.0
    for thr in np.linspace(0.05, 1.0, 20):
        ious = []
        for gset, rl, probs in per_item:
            pset = set(np.nonzero(probs >= thr)[0].tolist())
            ious.append(1.0 if not gset and not pset else
                        (0.0 if (not gset or not pset) else len(gset & pset) / len(gset | pset)))
        m = sum(ious) / len(ious)
        if m > best_iou:
            best_iou, best_thr = m, float(thr)

    return dict(roc_auc=roc, pr_auc=pr, best_threshold=best_thr, span_iou=best_iou,
                calib_corr=calib, class_roc=class_roc, class_macro=macro)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", action="append", default=[], help="name=path (repeatable)")
    args = ap.parse_args()

    gold = {it.id: it for it in load_jsonl(args.gold)}

    systems = []
    systems.append(("predict-nothing", lambda i: ([], [0.0] * len(gold[i].response))))
    systems.append(("predict-everything",
                    lambda i: ([{"start": 0, "end": len(gold[i].response), "label": "other"}],
                               [1.0] * len(gold[i].response))))
    for spec in args.pred:
        name, path = spec.split("=", 1)
        d = load_by_id(path)
        def mk(dd):
            return lambda i: (dd[i].get("pred_labels", []),
                              dd[i].get("char_probs") or [0.0] * len(gold[i].response))
        systems.append((name, mk(d)))

    results = [(name, suite(gold, fn)) for name, fn in systems]

    print(f"n={len(gold)}\n")
    print(f"{'system':22}{'span_iou':>9}{'bestThr':>8}{'roc_auc':>9}{'pr_auc':>8}{'calib':>8}{'clsMacro':>9}")
    for name, m in results:
        print(f"{name:22}{m['span_iou']:>9.4f}{m['best_threshold']:>8.2f}"
              f"{m['roc_auc']:>9.3f}{m['pr_auc']:>8.3f}{m['calib_corr']:>8.3f}{m['class_macro']:>9.3f}")
    print("\nclass_roc (per-category one-vs-rest ROC AUC)")
    print(f"{'system':22}" + "".join(f"{c[:7]:>9}" for c in CATEGORIES) + f"{'macro':>9}")
    for name, m in results:
        print(f"{name:22}" + "".join(f"{m['class_roc'][c]:>9.3f}" for c in CATEGORIES) + f"{m['class_macro']:>9.3f}")


if __name__ == "__main__":
    main()
