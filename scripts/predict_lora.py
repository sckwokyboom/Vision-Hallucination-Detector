"""Predict with a LoRA-adapted model on the hallucination detection task.

Usage:
    python scripts/predict_lora.py \
        --config configs/base.yaml \
        --input splits/dev.en.jsonl \
        --image_dir ../Shroom-Vision/images \
        --lora_dir lora_checkpoint \
        --output preds/lora_output.jsonl
"""

import argparse
import json
import os

import torch
import yaml
from PIL import Image
from peft import PeftModel
from tqdm import tqdm

from shroom.data import load_jsonl
from shroom.predict import predict_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model_id"]
    max_pixels = cfg.get("max_pixels", 1024 * 1024)
    dtype = getattr(torch, args.dtype)

    print(f"Loading base model {model_id} ...")
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, device_map="auto", dtype=dtype,
        attn_implementation="sdpa", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model.eval()

    class LoRABackend:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor

        def generate(self, image, text, n=1, temperature=0.5):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text}
            ]}]
            templ = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=templ, images=image, return_tensors="pt",
                max_pixels=max_pixels).to(self.model.device)
            plen = inputs.input_ids.shape[-1]
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs, max_new_tokens=512,
                    do_sample=temperature > 0, temperature=temperature,
                    top_p=0.95, num_return_sequences=n)
            return [self.processor.decode(seq[plen:], skip_special_tokens=True).strip()
                    for seq in gen]

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    backend = LoRABackend(model, processor)

    items = load_jsonl(args.input)
    if args.max_samples:
        items = items[:args.max_samples]

    n_samples = cfg.get("n_samples", 1)
    temperature = cfg.get("temperature", 0.5)
    tau = cfg.get("tau", 0.5)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for it in tqdm(items, desc="predict lora"):
            rec = {"id": it.id, "language": it.language, "response": it.response}
            img_path = os.path.join(args.image_dir, it.image_name)
            image = Image.open(img_path).convert("RGB")
            spans, per_char = predict_item(
                backend, it, image,
                n=n_samples, temperature=temperature, tau=tau,
            )
            rec["pred_labels"] = spans
            rec["char_probs"] = [round(p, 3) for p in per_char]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
