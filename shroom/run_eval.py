import argparse
import json

from .data import load_jsonl
from .metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    pred_by_id = {}
    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                pred_by_id[d["id"]] = d

    report = evaluate(gold, pred_by_id)
    for lang in sorted(report):
        r = report[lang]
        print(f"\n=== {lang}  (n={r['n']}) ===")
        print(f"  Char-IoU:            {r['iou']:.4f}")
        print(f"  [baseline nothing]:  {r['predict_nothing_iou']:.4f}")
        print(f"  [baseline all]:      {r['predict_all_iou']:.4f}")
        print(f"  Calibration Pearson: {r['pearson']:.4f}   Spearman: {r['spearman']:.4f}")
        print(f"  Per-label IoU: " +
              "  ".join(f"{k}={v:.3f}" for k, v in r["per_label_iou"].items()))


if __name__ == "__main__":
    main()
