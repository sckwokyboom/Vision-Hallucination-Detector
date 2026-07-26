"""LoRA fine-tune Gemma-4 on hallucination detection (supervised span extraction).

Usage:
    python scripts/finetune_lora.py \
        --config configs/base.yaml \
        --input splits/dev.en.jsonl \
        --image_dir ../Shroom-Vision/images \
        --output_dir lora_checkpoint \
        --max_samples 10
"""

import argparse
import json
import os
from dataclasses import dataclass

import torch
import yaml
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from shroom.data import load_jsonl
from shroom.predict import build_prompt, CATEGORIES


def gold_labels_to_json(item):
    if not item.labels:
        return "[]"
    entries = []
    for sp in item.labels:
        phrase = item.response[int(sp["start"]):int(sp["end"])]
        label = sp["label"]
        if label not in CATEGORIES:
            label = "other"
        entries.append({"phrase": phrase, "label": label})
    return json.dumps(entries, ensure_ascii=False)


class HallucinationDataset(Dataset):
    def __init__(self, items, image_dir, processor, max_pixels):
        self.data = []
        self.processor = processor
        self.max_pixels = max_pixels
        for it in items:
            img_path = os.path.join(image_dir, it.image_name)
            self.data.append({
                "image": Image.open(img_path).convert("RGB"),
                "user_text": build_prompt(it.prompt, it.response),
                "assistant_text": gold_labels_to_json(it),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        image = d["image"]
        user_text = d["user_text"]
        assistant_text = d["assistant_text"]

        user_msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": user_text}
        ]}]
        user_templ = self.processor.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True)

        full_msgs = user_msgs + [{"role": "assistant", "content": [
            {"type": "text", "text": assistant_text}
        ]}]
        full_templ = self.processor.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False)

        user_inputs = self.processor(
            text=user_templ, images=image, return_tensors="pt")
        user_len = user_inputs.input_ids.shape[-1]

        full_inputs = self.processor(
            text=full_templ, images=image, return_tensors="pt")

        input_ids = full_inputs.input_ids[0]
        labels = input_ids.clone()
        labels[:user_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": full_inputs.attention_mask[0],
            "labels": labels,
            "pixel_values": full_inputs.pixel_values[0],
        }
        if hasattr(full_inputs, "pixel_position_ids") and full_inputs.pixel_position_ids is not None:
            result["pixel_position_ids"] = full_inputs.pixel_position_ids[0]
        elif "pixel_position_ids" in full_inputs:
            result["pixel_position_ids"] = full_inputs["pixel_position_ids"][0]
        return result


@dataclass
class MultimodalCollator:
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            seqs = [f[key] for f in features]
            padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
            batch[key] = padded
        batch["pixel_values"] = pixel_values
        if "pixel_position_ids" in features[0]:
            batch["pixel_position_ids"] = torch.stack([f["pixel_position_ids"] for f in features])
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model_id"]
    max_pixels = cfg.get("max_pixels", 1024 * 1024)
    dtype = getattr(torch, args.dtype)

    if not torch.cuda.is_available():
        import sys
        sys.exit(
            "CUDA is NOT available. Check:\n"
            "  1. srun needs --gpus=1 for GPU allocation\n"
            "  2. Driver version vs PyTorch CUDA version mismatch\n"
            "Try: srun --partition=a100 --gpus=1 --cpus-per-task=8 "
            "--mem=64G --time=24:00:00 --pty bash"
        )

    print(f"Loading model {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, device_map="auto", dtype=dtype,
        attn_implementation="sdpa", trust_remote_code=True,
    )

    linear_modules = [name for name, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear)]
    print(f"Found {len(linear_modules)} nn.Linear modules for LoRA targeting")
    if not linear_modules:
        print("No nn.Linear modules found — model may use custom layer types. "
              "Trying regex pattern fallback.")
        target_modules = [r".*q_proj", r".*v_proj", r".*k_proj", r".*o_proj",
                          r".*gate_proj", r".*up_proj", r".*down_proj"]
    else:
        target_modules = linear_modules

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    items = load_jsonl(args.input)
    if args.max_samples:
        items = items[:args.max_samples]
    print(f"Training on {len(items)} examples")

    dataset = HallucinationDataset(items, args.image_dir, processor, max_pixels)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    print(f"CUDA available: {torch.cuda.is_available()}, bf16: {use_bf16}, fp16: {use_fp16}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="epoch",
            remove_unused_columns=False,
            report_to="none",
            bf16=use_bf16,
            fp16=use_fp16,
        ),
        train_dataset=dataset,
        data_collator=MultimodalCollator(),
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
