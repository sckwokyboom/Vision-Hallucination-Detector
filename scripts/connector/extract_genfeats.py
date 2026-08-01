"""Generative-surprise features: teacher-forced token statistics of the answer.

A fundamentally different signal from the hidden-state cache: the answer is placed
as the ASSISTANT turn (as if the model were generating it given image+question), and
one forward pass yields, per answer token:

  0  logp_actual   log P(token | prefix, image)  — would the model have said this?
  1  entropy       uncertainty of the next-token distribution
  2  top1_logp     log-prob of the model's preferred token
  3  margin        logp_actual - top1_logp (0 = model agrees, very negative = surprise)

Run twice — once normally and once with --no_image — and the DELTA of logp_actual
is a per-token visual-grounding signal: tokens whose probability the image raises
are grounded; tokens the image does not help (or hurts) are linguistic or contradicted.

Output: one {id}.npz per item with F [T, 4], tok_char, answer_len — same alignment
machinery as the H cache, so rows correspond 1:1 with the H cache's review tokens
(same tokenizer, same review_body -> same token count; the trainer verifies).

  python scripts/connector/extract_genfeats.py --model_id /path/gemma-4-12B-it \
      --train_file .../train.en.labeled.jsonl --image_dir .../images \
      --out_dir results/lp_en --device cuda:0
  python ... --no_image --out_dir results/lp_en_noimg
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shroom.data import load_jsonl                              # noqa: E402
from extract_features import (review_body, encode_review,       # noqa: E402
                              find_subseq, load_model)
from transformers import AutoProcessor                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--quant", choices=["bf16", "int8", "nf4"], default="bf16")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_image", action="store_true")
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tok = processor.tokenizer
    assert getattr(tok, "is_fast", False), "offset mapping needs a fast tokenizer"
    model = load_model(args.model_id, args.quant, args.device)
    dev = next(model.parameters()).device

    items = load_jsonl(args.train_file)
    if args.max_items:
        items = items[: args.max_items]

    t0, n_done, failed = time.time(), 0, []
    for idx, it in enumerate(items):
        out_path = os.path.join(args.out_dir, f"{it.id}.npz")
        if os.path.exists(out_path) and not args.probe:
            continue
        body = review_body(it.response)
        if not body.strip():
            continue
        img_path = os.path.join(args.image_dir, it.image_name)
        if not os.path.exists(img_path):
            continue
        image = None
        if not args.no_image:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            if max(w, h) > args.max_side:
                s = args.max_side / max(w, h)
                image = image.resize((int(w * s), int(h * s)))

        user = ([{"type": "image"}] if image is not None else []) + \
               [{"type": "text", "text": it.prompt}]
        msgs = [{"role": "user", "content": user},
                {"role": "assistant", "content": [{"type": "text", "text": body}]}]
        templ = processor.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=False)
        if image is not None:
            inputs = processor(text=templ, images=image, return_tensors="pt").to(dev)
        else:
            inputs = processor(text=templ, return_tensors="pt").to(dev)
        ids = inputs["input_ids"][0].tolist()

        ans_ids, offs = encode_review(tok, body)
        pos, trim = find_subseq(ids, ans_ids), 0
        while pos < 0 and trim < 3:
            trim += 1
            pos = find_subseq(ids, ans_ids[trim:])
        if pos < 0:
            failed.append(it.id)
            print(f"[warn] answer not found in chat template for {it.id}", flush=True)
            continue
        rt, tok_char = list(range(pos, pos + len(ans_ids[trim:]))), offs[trim:]
        while rt and not body[max(0, tok_char[0][0]):max(0, tok_char[0][1])].strip():
            rt.pop(0)
            tok_char = tok_char[1:]
        tok_char = [(max(0, a), max(0, b)) for a, b in tok_char]

        with torch.no_grad():
            logits = model(**inputs).logits[0].float()          # [S, V]
            # stats for token at position k come from logits at k-1
            pred_pos = torch.tensor([k - 1 for k in rt], device=logits.device)
            actual = torch.tensor([ids[k] for k in rt], device=logits.device)
            lg = logits[pred_pos]                               # [T, V]
            logp = torch.log_softmax(lg, dim=-1)
            p = logp.exp()
            lp_act = logp.gather(1, actual[:, None]).squeeze(1)
            entropy = -(p * logp).sum(-1)
            top1 = logp.max(-1).values
            F = torch.stack([lp_act, entropy, top1, lp_act - top1], dim=1)

        if args.probe:
            body_pieces = [body[a:b] for a, b in tok_char[:6]]
            print(f"[probe {it.id}] T={len(rt)} first_pieces={body_pieces} "
                  f"logp_mean={float(lp_act.mean()):.2f} ent_mean={float(entropy.mean()):.2f}",
                  flush=True)
            if idx + 1 >= args.probe:
                print("[probe done]")
                return
            continue

        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, F=F.cpu().numpy().astype(np.float16),
                            tok_char=np.array(tok_char, dtype=np.int32),
                            answer_len=len(it.response))
        os.replace(tmp, out_path)
        n_done += 1
        if n_done % 100 == 0:
            print(f"...{n_done} ({(time.time() - t0) / n_done:.2f}s/item)", flush=True)
    print(f"[done] {n_done} new -> {args.out_dir} in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    if failed:
        with open(os.path.join(args.out_dir, "_skipped.json"), "w") as f:
            json.dump({"unaligned": failed}, f)
        raise SystemExit(f"ERROR: {len(failed)} items unaligned — cache incomplete")


if __name__ == "__main__":
    main()
