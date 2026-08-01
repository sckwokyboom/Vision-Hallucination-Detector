"""Ensemble span predictions from several systems by averaging char probabilities.

Systems trained on DIFFERENT backbones (12B dense, 26B MoE) make decorrelated
errors; averaging their per-char probabilities and re-decoding recovers spans that
any single system was too uncertain about, and suppresses one-model noise.

Inputs are prediction JSONLs that carry "char_probs" (train_connector/predict_lora
both emit it). Decode: average probs -> threshold tau -> contiguous runs ->
min_len/merge_gap cleanup -> constant label. Sweep tau and pairwise weights on the
official scorer.

  # sweep on tune
  python scripts/official/ensemble_predictions.py \
      --pred a2=results/final/current_a2/tune_predictions.jsonl \
      --pred b26=results/lora_26b/dev_pred_....jsonl \
      --gold splits/dev.en.jsonl --eval_ids splits/en.eval_protocol.json --sweep

  # apply the chosen setting to test predictions
  python scripts/official/ensemble_predictions.py \
      --pred a2=... --pred b26=... --weights 0.5,0.5 --tau 0.35 \
      --out results/final/submission/ensemble_en.jsonl
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official_scorer import score_cor, score_cor_lbl, score_iou  # noqa: E402


def load_preds(spec):
    name, path = spec.split("=", 1)
    out = {}
    for ln in open(path):
        d = json.loads(ln)
        if "char_probs" not in d:
            raise SystemExit(f"{path}: row {d.get('id')} has no char_probs — "
                             "re-emit predictions with a current trainer/predictor")
        out[d["id"]] = np.asarray(d["char_probs"], dtype=np.float32)
    return name, out


def decode(p, tau, min_len=3, merge_gap=2, label="invention"):
    spans, i, n = [], 0, len(p)
    while i < n:
        if p[i] >= tau:
            j = i
            while j < n and p[j] >= tau:
                j += 1
            spans.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [{"start": a, "end": b, "prob": float(p[a:b].mean()), "label": label}
            for a, b in merged if b - a >= min_len]


def ensemble_rows(systems, weights, tau, ids=None):
    names = [n for n, _ in systems]
    w = np.asarray(weights, dtype=np.float32)
    w = w / w.sum()
    common = set.intersection(*(set(p) for _, p in systems))
    rows = []
    for rid in sorted(common):
        if ids is not None and rid not in ids:
            continue
        L = min(len(p[rid]) for _, p in systems)      # defensive: probs may differ in tail
        p = sum(wi * preds[rid][:L] for wi, (_, preds) in zip(w, systems))
        rows.append({"id": rid, "labels": decode(p, tau)})
    return rows, names


def official(rows, gold, ids):
    gold_by_id = {}
    for ln in open(gold):
        d = json.loads(ln)
        gold_by_id[d["id"]] = {"id": d["id"], "labels": d.get("labels", []),
                               "text_len": len(d["response"])}
    iou, cor, corl, n = 0.0, 0.0, 0.0, 0
    for r in rows:
        g = gold_by_id.get(r["id"])
        if g is None or (ids is not None and r["id"] not in ids):
            continue
        iou += score_iou(g, r)
        cor += score_cor(g, r)
        corl += score_cor_lbl(g, r)
        n += 1
    return dict(IoU=iou / n, Cor=cor / n, Cor_lbl=corl / n, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--gold", default=None)
    ap.add_argument("--eval_ids", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--weights", default=None, help="comma floats, len = #systems")
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--tau_grid", default="0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    systems = [load_preds(s) for s in args.pred]
    ids = None
    if args.eval_ids:
        prot = json.load(open(args.eval_ids))
        ids = set(prot.get("tune_dev") or [])

    if args.sweep:
        if not args.gold:
            raise SystemExit("--sweep needs --gold")
        k = len(systems)
        wgrids = ([tuple(g)] if (g := None) else
                  [(1.0,)] if k == 1 else
                  [(w, 1 - w) for w in (0.25, 0.35, 0.5, 0.65, 0.75)] if k == 2 else
                  [t for t in itertools.product((0.2, 0.35, 0.5), repeat=k)
                   if abs(sum(t) - 1) < 0.5])
        results = []
        for w in wgrids:
            for tau in [float(x) for x in args.tau_grid.split(",")]:
                rows, names = ensemble_rows(systems, list(w), tau, ids)
                m = official(rows, args.gold, ids)
                results.append((m["IoU"], m["Cor"], m["Cor_lbl"], w, tau, m["n"]))
        results.sort(reverse=True)
        print(f"{'IoU':>8} {'Cor':>8} {'Cor_lbl':>8} {'weights':>18} {'tau':>5} {'n':>5}")
        for iou, cor, corl, w, tau, n in results[:15]:
            print(f"{iou:>8.4f} {cor:>8.4f} {corl:>8.4f} "
                  f"{','.join(f'{x:.2f}' for x in w):>18} {tau:>5.2f} {n:>5}")
        # singles for reference
        for i, (nm, _) in enumerate(systems):
            w = [0.0] * len(systems)
            w[i] = 1.0
            rows, _ = ensemble_rows(systems, w, 0.35, ids)
            m = official(rows, args.gold, ids)
            print(f"single {nm:12} IoU={m['IoU']:.4f} Cor={m['Cor']:.4f} (tau=0.35)")
        return

    w = ([float(x) for x in args.weights.split(",")] if args.weights
         else [1.0 / len(systems)] * len(systems))
    rows, names = ensemble_rows(systems, w, args.tau, ids)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows ({'+'.join(names)}) -> {args.out}")
    if args.gold:
        print(official(rows, args.gold, ids))


if __name__ == "__main__":
    main()
