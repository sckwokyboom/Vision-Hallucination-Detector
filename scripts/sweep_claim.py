"""Offline tau sweep for claim-level predictions (files with `claim_probs`)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl                       # noqa: E402
from shroom.metrics import (char_iou, gold_char_probs,   # noqa: E402
                            calibration, trivial_baselines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--taus", default="0.2,0.3,0.5,0.7,0.9,1.0")
    args = ap.parse_args()

    gold = {it.id: it for it in load_jsonl(args.gold)}
    pred = {}
    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                pred[d["id"]] = d

    ids = [i for i in pred if i in gold and "claim_probs" in pred[i]]
    if not ids:
        print("No claim_probs found.")
        return
    items = [gold[i] for i in ids]
    floor = trivial_baselines(items)["predict_nothing_iou"]

    # Calibration is tau-independent (per-char claim prob).
    gp, pp = [], []
    for i in ids:
        it = gold[i]
        cp = pred[i].get("char_probs") or []
        cp = (cp + [0.0] * len(it.response))[:len(it.response)]
        gp.extend(gold_char_probs(it.labels, len(it.response)))
        pp.extend(cp)
    cal = calibration(gp, pp)

    print(f"subset n={len(ids)}  floor={floor:.4f}  "
          f"calibration Pearson={cal['pearson']:.4f} Spearman={cal['spearman']:.4f}")
    print("\n  tau   Char-IoU   vs-floor")
    best = (None, -1.0)
    for tau in [float(x) for x in args.taus.split(",")]:
        ious = []
        for i in ids:
            it = gold[i]
            spans = [c for c in pred[i]["claim_probs"] if c["prob"] >= tau]
            ious.append(char_iou(it.labels, spans, len(it.response)))
        m = sum(ious) / len(ious)
        flag = "  <-- beats floor" if m > floor else ""
        print(f"  {tau:.2f}   {m:.4f}    {m - floor:+.4f}{flag}")
        if m > best[1]:
            best = (tau, m)
    print(f"\nBEST: tau={best[0]:.2f}  IoU={best[1]:.4f}  (floor {floor:.4f})")


if __name__ == "__main__":
    main()
