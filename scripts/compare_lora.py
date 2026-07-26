"""Compare IoU: baseline vs LoRA-finetuned predictions.

Usage:
    python scripts/compare_lora.py --gold data.jsonl --base preds/base.jsonl --lora preds/lora.jsonl
"""

import argparse
import json

from shroom.data import load_jsonl
from shroom.metrics import evaluate, char_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    args = ap.parse_args()

    def load_preds(path):
        preds = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                preds[d["id"]] = d
        return preds

    gold_items = load_jsonl(args.gold)
    base_preds = load_preds(args.base)
    lora_preds = load_preds(args.lora)

    common_ids = set(base_preds) & set(lora_preds)
    gold_by_id = {it.id: it for it in gold_items}
    gold_items = [gold_by_id[i] for i in common_ids if i in gold_by_id]

    print(f"\n  Gold: {len(load_jsonl(args.gold))} total  |  Matched: {len(gold_items)} "
          f"(in both preds)")

    base_report = evaluate(gold_items, base_preds)
    lora_report = evaluate(gold_items, lora_preds)

    per_item_base, per_item_lora = [], []
    for it in gold_items:
        rl = len(it.response)
        b = base_preds.get(it.id, {"pred_labels": []})
        l = lora_preds.get(it.id, {"pred_labels": []})
        per_item_base.append(char_iou(it.labels, b.get("pred_labels", []), rl))
        per_item_lora.append(char_iou(it.labels, l.get("pred_labels", []), rl))

    print(f"\n{'='*60}")
    print(f"  Language       : {gold_items[0].language if gold_items else '?'}")
    print(f"  N items        : {len(gold_items)}")
    print(f"{'='*60}")
    print(f"  {'Metric':<24} {'Baseline':>12} {'LoRA':>12} {'Delta':>12}")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*12}")

    for lang in sorted(base_report):
        br = base_report[lang]
        lr = lora_report[lang]
        for metric, key_base, key_lora in [
            ("Char-IoU (mean)", br["iou"], lr["iou"]),
            ("calibr. Pearson", br["pearson"], lr["pearson"]),
        ]:
            print(f"  {metric:<24} {key_base:>12.4f} {key_lora:>12.4f} {key_lora - key_base:>+12.4f}")

    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*12}")
    base_mean = sum(per_item_base) / len(per_item_base)
    lora_mean = sum(per_item_lora) / len(per_item_lora)
    print(f"  {'per-item IoU mean':<24} {base_mean:>12.4f} {lora_mean:>12.4f} {lora_mean - base_mean:>+12.4f}")

    improved = sum(1 for b, l in zip(per_item_base, per_item_lora) if l > b)
    tied = sum(1 for b, l in zip(per_item_base, per_item_lora) if l == b)
    print(f"\n  Improved: {improved}/{len(gold_items)}  Tied: {tied}/{len(gold_items)}  "
          f"Worse: {len(gold_items) - improved - tied}/{len(gold_items)}")


if __name__ == "__main__":
    main()
