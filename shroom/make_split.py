import argparse
import json
import os

import yaml

from .data import load_jsonl
from .split import group_split_by_image

LANGS = ["en", "fr", "it", "zh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distrib", default="../Shroom-Vision/distrib")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out_dir", default="splits")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out_dir, exist_ok=True)
    for lang in LANGS:
        path = os.path.join(args.distrib, f"shroom-vision.train.{lang}.labeled.jsonl")
        items = load_jsonl(path)
        train_ids, dev_ids = group_split_by_image(
            items, dev_frac=cfg.get("dev_frac", 0.1), seed=cfg.get("seed", 13))
        json.dump({"train": train_ids, "dev": dev_ids},
                  open(os.path.join(args.out_dir, f"{lang}.json"), "w"))
        dev_set = set(dev_ids)
        with open(os.path.join(args.out_dir, f"dev.{lang}.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                if it.id in dev_set:
                    f.write(json.dumps({
                        "id": it.id, "language": it.language, "prompt": it.prompt,
                        "image_name": it.image_name, "response": it.response,
                        "labels": it.labels}, ensure_ascii=False) + "\n")
        print(f"{lang}: train={len(train_ids)} dev={len(dev_ids)}")


if __name__ == "__main__":
    main()
