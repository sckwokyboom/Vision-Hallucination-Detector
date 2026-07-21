"""Offline sweep over the tau threshold on a raw-saved prediction file.

Re-aggregates stored raw samples (no re-inference) for a grid of tau, and reports
Char-IoU per tau plus the tau-independent calibration and the predict-nothing floor,
all restricted to the items actually present in the prediction file.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl                       # noqa: E402
from shroom.predict import parse_output                  # noqa: E402
from shroom.aggregate import aggregate                   # noqa: E402
from shroom.metrics import (char_iou, gold_char_probs,   # noqa: E402
                            calibration, trivial_baselines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True, help="prediction file saved with --save_raw")
    ap.add_argument("--taus", default="0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8")
    args = ap.parse_args()

    gold_by_id = {it.id: it for it in load_jsonl(args.gold)}
    pred = {}
    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                pred[d["id"]] = d

    # Pre-parse the raw samples once per item.
    prep = []
    no_raw = 0
    for iid, rec in pred.items():
        raws = rec.get("raw_samples")
        if raws is None:
            no_raw += 1
            continue
        it = gold_by_id.get(iid)
        if it is None:
            continue
        prep.append((it, [parse_output(r) for r in raws]))

    if not prep:
        print(f"No raw_samples found in {args.pred} (re-run predict with --save_raw). "
              f"items_without_raw={no_raw}")
        return

    n = len(prep)
    floor = trivial_baselines([it for it, _ in prep])

    # Calibration is tau-independent (per-char freq); compute once.
    gp, pp = [], []
    for it, samples in prep:
        _, per_char = aggregate(samples, it.response, tau=0.5)
        gp.extend(gold_char_probs(it.labels, len(it.response)))
        pp.extend(per_char)
    cal = calibration(gp, pp)

    print(f"subset n={n}  (items scored = those with raw_samples)")
    print(f"predict-nothing floor IoU = {floor['predict_nothing_iou']:.4f}   "
          f"predict-all = {floor['predict_all_iou']:.4f}")
    print(f"calibration  Pearson={cal['pearson']:.4f}  Spearman={cal['spearman']:.4f}")
    print("\n  tau   Char-IoU   vs-floor")
    best = (None, -1.0)
    for tau in [float(x) for x in args.taus.split(",")]:
        ious = []
        for it, samples in prep:
            spans, _ = aggregate(samples, it.response, tau=tau)
            ious.append(char_iou(it.labels, spans, len(it.response)))
        m = sum(ious) / len(ious)
        flag = "  <-- beats floor" if m > floor["predict_nothing_iou"] else ""
        print(f"  {tau:.2f}   {m:.4f}    {m - floor['predict_nothing_iou']:+.4f}{flag}")
        if m > best[1]:
            best = (tau, m)
    print(f"\nBEST: tau={best[0]:.2f}  IoU={best[1]:.4f}  "
          f"(floor {floor['predict_nothing_iou']:.4f})")


if __name__ == "__main__":
    main()
