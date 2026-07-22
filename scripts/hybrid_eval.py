"""Hybrid solution (strategy E): claim-level clean-gate + Gemma extraction spans.

For each item, claim-level decides clean vs dirty at the ITEM level (any claim with
prob >= tau => dirty). Clean items get empty spans (protecting their IoU=1.0); dirty
items take the Gemma extraction spans (better recall + tighter spans + calibrated probs).

Inputs are two prediction files over the same items:
  --extraction : a Gemma extraction file (pred_labels + char_probs), e.g. from mlx_predict.py
  --claim      : a claim-level file (claim_probs), from mlx_claim.py
Prints IoU / calibration / clean-protected for floor, the two components, and the hybrid,
and (optionally) writes the hybrid predictions in the standard pred format for run_eval.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl                                   # noqa: E402
from shroom.metrics import (_char_set, char_iou, gold_char_probs,    # noqa: E402
                            calibration, trivial_baselines)


def load_by_id(path):
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
    ap.add_argument("--extraction", required=True)
    ap.add_argument("--claim", required=True)
    ap.add_argument("--tau", type=float, default=0.5, help="claim-gate threshold")
    ap.add_argument("--out", default=None, help="write hybrid predictions here (optional)")
    args = ap.parse_args()

    gold = {it.id: it for it in load_jsonl(args.gold)}
    ext = load_by_id(args.extraction)
    clm = load_by_id(args.claim)
    ids = [i for i in gold if i in ext and i in clm]

    def resp_len(i):
        return len(gold[i].response)

    def gemma_spans(i):
        return ext[i].get("pred_labels", [])

    def gemma_probs(i):
        cp = ext[i].get("char_probs") or []
        return (cp + [0.0] * resp_len(i))[:resp_len(i)]

    def claim_spans(i):
        return [c for c in clm[i].get("claim_probs", []) if c["prob"] >= args.tau]

    def claim_probs(i):
        cp = clm[i].get("char_probs") or []
        return (cp + [0.0] * resp_len(i))[:resp_len(i)]

    def item_dirty(i):
        return any(c["prob"] >= args.tau for c in clm[i].get("claim_probs", []))

    def hybrid_spans(i):
        return [] if not item_dirty(i) else gemma_spans(i)

    def hybrid_probs(i):
        return [0.0] * resp_len(i) if not item_dirty(i) else gemma_probs(i)

    def evaluate(spans_fn, probs_fn):
        ious, gp, pp, clean_prot, n_clean = [], [], [], 0, 0
        for i in ids:
            it = gold[i]
            spans = spans_fn(i)
            ious.append(char_iou(it.labels, spans, resp_len(i)))
            gp.extend(gold_char_probs(it.labels, resp_len(i)))
            pp.extend(probs_fn(i))
            if not it.labels:
                n_clean += 1
                if not _char_set(spans, resp_len(i)):
                    clean_prot += 1
        cal = calibration(gp, pp)
        return sum(ious) / len(ious), cal, f"{clean_prot}/{n_clean}"

    floor = trivial_baselines([gold[i] for i in ids])
    zeros = lambda i: [0.0] * resp_len(i)

    rows = [
        ("predict-nothing (floor)", lambda i: [], zeros),
        ("Gemma extraction", gemma_spans, gemma_probs),
        (f"claim-level (tau={args.tau})", claim_spans, claim_probs),
        ("HYBRID E (claim-gate+Gemma)", hybrid_spans, hybrid_probs),
    ]
    print(f"n={len(ids)}  floor(predict-nothing)={floor['predict_nothing_iou']:.4f}  "
          f"predict-all={floor['predict_all_iou']:.4f}\n")
    print(f"{'system':32}{'IoU':>8}{'Pearson':>9}{'Spearman':>9}{'cleanOK':>9}")
    for name, sf, pf in rows:
        iou, cal, cp = evaluate(sf, pf)
        print(f"{name:32}{iou:>8.4f}{cal['pearson']:>9.3f}{cal['spearman']:>9.3f}{cp:>9}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for i in ids:
                it = gold[i]
                f.write(json.dumps({
                    "id": i, "language": it.language, "response": it.response,
                    "pred_labels": hybrid_spans(i),
                    "char_probs": [round(x, 3) for x in hybrid_probs(i)],
                }, ensure_ascii=False) + "\n")
        print(f"\nwrote hybrid predictions -> {args.out}")


if __name__ == "__main__":
    main()
