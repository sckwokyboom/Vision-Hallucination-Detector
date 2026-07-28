"""Build the deterministic quant-ablation subset: first N train-side items (file order,
image-grouped split seed 13) + ALL tune_dev items from the frozen eval protocol.
Identical output on any machine given the same train file + protocol.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shroom.data import load_jsonl                      # noqa: E402
from shroom.split import group_split_by_image           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = load_jsonl(args.train_file)
    tr_ids, _ = group_split_by_image(items, dev_frac=0.1, seed=13)
    tune = set(json.load(open(args.protocol))["tune_dev"])
    tr_set = set(tr_ids)
    sel, ntr = [], 0
    for it in items:                                     # file order
        if it.id in tune:
            sel.append(it)
        elif it.id in tr_set and ntr < args.n_train:
            sel.append(it); ntr += 1
    with open(args.out, "w", encoding="utf-8") as f:
        for it in sel:
            f.write(json.dumps({"id": it.id, "language": it.language, "prompt": it.prompt,
                                "image_name": it.image_name, "response": it.response,
                                "labels": it.labels}, ensure_ascii=False) + "\n")
    print(f"subset -> {args.out}: {len(sel)} items ({ntr} train + {len(sel)-ntr} tune)")


if __name__ == "__main__":
    main()
