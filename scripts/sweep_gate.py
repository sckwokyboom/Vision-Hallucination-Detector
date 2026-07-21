"""Offline clean-gate sweep. Combines a gate file (prob_dirty per id) with an
extraction file saved via --save_raw, and sweeps the gate threshold theta and the
span threshold tau. If prob_dirty < theta the item is gated clean (spans = []),
otherwise spans come from aggregating the extraction samples at tau.
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


def load_jsonl_by_id(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["id"]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--extraction", required=True, help="pred file saved with --save_raw")
    ap.add_argument("--gate", required=True, help="gate file with prob_dirty per id")
    ap.add_argument("--taus", default="0.5,0.6,0.7,0.8")
    ap.add_argument("--thetas", default="0.0,0.2,0.4,0.6,0.8,1.0")
    args = ap.parse_args()

    gold = {it.id: it for it in load_jsonl(args.gold)}
    extr = load_jsonl_by_id(args.extraction)
    gate = load_jsonl_by_id(args.gate)

    ids = [i for i in extr if i in gate and extr[i].get("raw_samples") is not None and i in gold]
    prep = [(gold[i], [parse_output(r) for r in extr[i]["raw_samples"]], gate[i]["prob_dirty"])
            for i in ids]
    n = len(prep)
    if n == 0:
        print("No overlapping items with raw_samples + gate. Check inputs.")
        return

    floor = trivial_baselines([it for it, _, _ in prep])["predict_nothing_iou"]

    # Gate diagnostic: does prob_dirty separate clean from dirty?
    dirty_pd = [pd for it, _, pd in prep if it.labels]
    clean_pd = [pd for it, _, pd in prep if not it.labels]
    def mean(x):
        return sum(x) / len(x) if x else float("nan")
    print(f"subset n={n}  (clean={len(clean_pd)}, dirty={len(dirty_pd)})  floor={floor:.4f}")
    print(f"gate prob_dirty: mean on CLEAN items={mean(clean_pd):.2f}  on DIRTY items={mean(dirty_pd):.2f}"
          "   (want dirty > clean)")

    taus = [float(x) for x in args.taus.split(",")]
    thetas = [float(x) for x in args.thetas.split(",")]

    print("\nChar-IoU   rows=theta (gate; 0.0 = no gate)   cols=tau")
    header = "theta\\tau " + "".join(f"{t:>8.2f}" for t in taus)
    print(header)
    best = (None, None, -1.0)
    for th in thetas:
        cells = []
        for tau in taus:
            ious = []
            for it, samples, pd in prep:
                if pd < th:
                    spans = []
                else:
                    spans, _ = aggregate(samples, it.response, tau=tau)
                ious.append(char_iou(it.labels, spans, len(it.response)))
            m = sum(ious) / len(ious)
            cells.append(m)
            if m > best[2]:
                best = (th, tau, m)
        row = f"{th:>6.2f}   " + "".join(f"{c:>8.4f}" for c in cells)
        print(row)

    bth, btau, bm = best
    print(f"\nBEST: theta={bth:.2f}  tau={btau:.2f}  IoU={bm:.4f}   "
          f"floor={floor:.4f}   delta={bm-floor:+.4f}"
          f"{'   <-- beats floor' if bm > floor else ''}")

    # Calibration for the best config (gated per-char probs).
    gp, pp = [], []
    for it, samples, pd in prep:
        if pd < bth:
            per_char = [0.0] * len(it.response)
        else:
            _, per_char = aggregate(samples, it.response, tau=btau)
        gp.extend(gold_char_probs(it.labels, len(it.response)))
        pp.extend(per_char)
    cal = calibration(gp, pp)
    print(f"calibration @ best: Pearson={cal['pearson']:.4f}  Spearman={cal['spearman']:.4f}")


if __name__ == "__main__":
    main()
