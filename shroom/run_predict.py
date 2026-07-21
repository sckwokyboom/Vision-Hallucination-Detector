import argparse
import json
import os

import yaml
from PIL import Image
from tqdm import tqdm

from .data import load_jsonl
from .predict import predict_item


def make_backend(cfg):
    kind = cfg.get("backend", "mock")
    if kind == "hf":
        from .backends.hf_backend import HFBackend
        return HFBackend(model_id=cfg["model_id"], max_pixels=cfg.get("max_pixels", 1024 * 1024))
    if kind == "mlx":
        from .backends.mlx_backend import MLXBackend
        return MLXBackend(model_id=cfg["model_id"])
    raise ValueError(f"Unknown/unsupported backend for prediction: {kind!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    backend = make_backend(cfg)
    items = load_jsonl(args.input)
    if args.max_samples:
        items = items[:args.max_samples]

    image_dir_real = os.path.realpath(args.image_dir) + os.sep
    with open(args.output, "w", encoding="utf-8") as out:
        for it in tqdm(items, desc="predict"):
            rec = {"id": it.id, "language": it.language, "response": it.response}
            img_path = os.path.realpath(os.path.join(args.image_dir, it.image_name))
            if not img_path.startswith(image_dir_real) or not os.path.exists(img_path):
                rec["pred_labels"] = []
                rec["char_probs"] = []
                rec["error"] = "image_missing"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            image = Image.open(img_path).convert("RGB")
            spans, per_char = predict_item(
                backend, it, image,
                n=cfg.get("n_samples", 5),
                temperature=cfg.get("temperature", 0.5),
                tau=cfg.get("tau", 0.5),
            )
            rec["pred_labels"] = spans
            rec["char_probs"] = [round(p, 3) for p in per_char]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
