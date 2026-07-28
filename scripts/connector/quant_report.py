"""Compare decoder runs trained on different backbone precisions (or any variants).

For each --pred name=dev_pred.jsonl: decoded IoU (overall/dirty/cleanOK), raw char
ROC-AUC & PR-AUC, per-item ORACLE threshold IoU and oracle multi-span IoU (upper bounds
from the same probability fields). Plus paired image-cluster bootstrap deltas vs the
first variant. Gold = --gold jsonl (labeled).
"""
import argparse
import json
import random
import sys
import os

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shroom.data import load_jsonl                              # noqa: E402
from shroom.metrics import char_iou, gold_char_probs, _char_set  # noqa: E402


def runs(cp, thr):
    out, i = [], 0
    while i < len(cp):
        if cp[i] >= thr:
            j = i
            while j < len(cp) and cp[j] >= thr:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def oracle_thr(cp, labels, n):
    best = 0.0
    for thr in np.unique(np.round(cp, 2)):
        if thr <= 0:
            continue
        best = max(best, char_iou(labels, [{"start": a, "end": b} for a, b in runs(cp, thr)], n))
    return best


def oracle_multispan(cp, labels, n):
    gset = _char_set(labels, n)
    best = 0.0
    for thr in np.unique(np.round(cp, 2)):
        if thr <= 0:
            continue
        rs = runs(cp, thr)
        if not rs or len(rs) > 40:
            continue
        chosen, cur = [], 0.0
        improved = True
        while improved:
            improved = False
            for r in rs:
                if r in chosen:
                    continue
                s = set()
                for a, b in chosen + [r]:
                    s.update(range(a, b))
                iou = len(s & gset) / len(s | gset) if (s or gset) else 1.0
                if iou > cur + 1e-9:
                    cur = iou; chosen.append(r); improved = True
        best = max(best, cur)
    return best


def load_pred(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r["id"]] = r
    return d


def per_item(gold, pred):
    out = {}
    for i, it in gold.items():
        if i not in pred:
            continue
        n = len(it.response)
        cp = np.array(((pred[i].get("char_probs") or []) + [0.0] * n)[:n])
        rec = {"iou": char_iou(it.labels, pred[i].get("pred_labels", []), n),
               "clean": not it.labels, "img": it.image_name}
        if it.labels:
            gp = gold_char_probs(it.labels, n)
            gb = [1 if x > 0 else 0 for x in gp]
            if len(set(gb)) > 1:
                rec["auc"] = roc_auc_score(gb, cp)
                rec["pr"] = average_precision_score(gb, cp)
            rec["o_thr"] = oracle_thr(cp, it.labels, n)
            rec["o_ms"] = oracle_multispan(cp, it.labels, n)
        out[i] = rec
    return out


def agg(pi):
    dirty = [r for r in pi.values() if not r["clean"]]
    clean = [r for r in pi.values() if r["clean"]]
    return dict(
        n=len(pi), overall=np.mean([r["iou"] for r in pi.values()]),
        dirty=np.mean([r["iou"] for r in dirty]) if dirty else 0,
        cleanOK=np.mean([1.0 if r["iou"] == 1.0 else 0.0 for r in clean]) if clean else 1,
        auc=np.nanmean([r.get("auc", np.nan) for r in dirty]),
        pr=np.nanmean([r.get("pr", np.nan) for r in dirty]),
        o_thr=np.mean([r["o_thr"] for r in dirty]) if dirty else 0,
        o_ms=np.mean([r["o_ms"] for r in dirty]) if dirty else 0)


def paired_cluster_delta(pa, pb, key="iou", n_boot=2000, seed=0):
    ids = [i for i in pa if i in pb]
    by_img = {}
    for i in ids:
        by_img.setdefault(pa[i]["img"], []).append(i)
    imgs = sorted(by_img)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        sel = [i for im in (rng.choice(imgs) for _ in imgs) for i in by_img[im]]
        deltas.append(np.mean([pa[i][key] for i in sel]) - np.mean([pb[i][key] for i in sel]))
    deltas = np.array(deltas)
    d0 = np.mean([pa[i][key] for i in ids]) - np.mean([pb[i][key] for i in ids])
    return d0, np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", action="append", required=True, help="name=path (first = baseline)")
    args = ap.parse_args()
    gold = {it.id: it for it in load_jsonl(args.gold)}

    results = []
    for spec in args.pred:
        name, path = spec.split("=", 1)
        pi = per_item(gold, load_pred(path))
        results.append((name, pi, agg(pi)))

    print(f"{'variant':22}{'overall':>8}{'dirty':>7}{'cleanOK':>8}{'AUC':>7}{'PR-AUC':>8}"
          f"{'o-thr':>7}{'o-mspan':>8}")
    for name, _, a in results:
        print(f"{name:22}{a['overall']:>8.3f}{a['dirty']:>7.3f}{a['cleanOK']:>8.2f}"
              f"{a['auc']:>7.3f}{a['pr']:>8.3f}{a['o_thr']:>7.3f}{a['o_ms']:>8.3f}")

    if len(results) > 1:
        base_name, base_pi, _ = results[0]
        print(f"\npaired image-cluster deltas vs {base_name} (95% CI):")
        for name, pi, _ in results[1:]:
            for key, lbl in [("iou", "overall IoU"), ("o_thr", "oracle-thr")]:
                sub_a = {i: r for i, r in pi.items() if key in r or key == "iou"}
                d0, lo, hi = paired_cluster_delta(
                    {i: r for i, r in sub_a.items() if key == "iou" or not r["clean"]},
                    {i: r for i, r in base_pi.items() if key == "iou" or not r["clean"]}, key)
                sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "ns"
                print(f"  {name} - {base_name}  {lbl:12} {d0:+.4f}  [{lo:+.4f},{hi:+.4f}]  {sig}")


if __name__ == "__main__":
    main()
