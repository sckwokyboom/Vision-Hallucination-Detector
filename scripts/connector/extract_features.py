"""Stage 1: run FROZEN Gemma 4 12B once over the data and cache connector inputs.

Per item saves an .npz with:
  V        [P, L, D] fp16 — hidden states at visual-token positions (layers --layers)
  H        [T, L, D] fp16 — hidden states of the REVIEW copy of the answer
  tok_char [T, 2]        — char (start,end) of each review token, relative to the answer
Prompt format (per the spec):
  <image>\nQuestion: <prompt>\nCandidate answer: <response>\nReview token by token:\n<response>

Run --probe 3 first: prints token counts / shapes so the setup can be validated cheaply.
Resolution ablation: pass --processor_kwargs '{"images_kwargs": {...}}' (model-specific);
the default uses the processor's native settings — check visual token count in the probe.
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
from shroom.data import load_jsonl                      # noqa: E402

from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402


def build_text(prompt, response):
    return (f"Question: {prompt}\nCandidate answer: {response}\n"
            f"Review token by token:\n{response}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="google/gemma-4-12B-it")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", default="24,32,40,48",
                    help="hidden-state layers to cache (comma list)")
    ap.add_argument("--max_side", type=int, default=896, help="resize cap before processor")
    ap.add_argument("--processor_kwargs", default=None, help="JSON passthrough to processor")
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--probe", type=int, default=0, help="probe N items: print shapes, no save")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model_id} on {device} (frozen, bf16)", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    [p.requires_grad_(False) for p in model.parameters()]

    # visual token id: try the common config attributes
    img_tok = None
    for attr in ("image_token_id", "image_token_index"):
        img_tok = getattr(model.config, attr, None) or img_tok
    if img_tok is None and hasattr(processor, "image_token"):
        img_tok = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    assert img_tok is not None, "could not find image token id — inspect model.config in --probe"
    pk = json.loads(args.processor_kwargs) if args.processor_kwargs else {}

    items = load_jsonl(args.train_file)
    if args.max_items:
        items = items[:args.max_items]
    tok = processor.tokenizer

    t0, n_done = time.time(), 0
    for idx, it in enumerate(items):
        out_path = os.path.join(args.out_dir, f"{it.id}.npz")
        if os.path.exists(out_path) and not args.probe:
            continue
        img_path = os.path.join(args.image_dir, it.image_name)
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        if max(w, h) > args.max_side:
            s = args.max_side / max(w, h)
            image = image.resize((int(w * s), int(h * s)))

        text = build_text(it.prompt, it.response)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text}]}]
        templ = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
        inputs = processor(text=templ, images=image, return_tensors="pt", **pk).to(model.device)
        ids = inputs["input_ids"][0]

        # locate the REVIEW copy of the answer: last occurrence of the response in the template
        r_start = templ.rfind(it.response)
        assert r_start > 0, f"review copy not found in template for {it.id}"
        enc = tok(templ, return_offsets_mapping=True, add_special_tokens=False)
        offs = enc["offset_mapping"]
        # align enc ids to model ids (they can differ by specials; match by length heuristic)
        # robust approach: recompute positions on the tokenized template used by the processor
        # when lengths differ, fall back to enc-based indexing into hidden states of a fresh pass
        review_tok, tok_char = [], []
        for k, (s, e) in enumerate(offs):
            if s >= r_start and e <= r_start + len(it.response) and e > s:
                review_tok.append(k)
                tok_char.append((s - r_start, e - r_start))
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states                               # tuple: n_layers+1 x [1,S,D]
        L = [min(l, len(hs) - 1) for l in layers]
        stack = torch.stack([hs[l][0] for l in L], dim=1)    # [S, L, D]

        vis_pos = (ids == img_tok).nonzero(as_tuple=True)[0].tolist()
        # offset between enc positions and model positions (specials prepended by processor)
        shift = ids.shape[0] - len(enc["input_ids"])
        rt = [p + shift for p in review_tok if p + shift < stack.shape[0]]

        V = stack[vis_pos].to(torch.float16).cpu().numpy()
        H = stack[rt].to(torch.float16).cpu().numpy()

        if args.probe:
            print(f"[probe {it.id}] seq={ids.shape[0]}  visual_tokens={len(vis_pos)}  "
                  f"review_tokens={len(rt)}/{len(review_tok)}  shift={shift}  "
                  f"V={V.shape} H={H.shape}  answer_chars={len(it.response)}", flush=True)
            if idx + 1 >= args.probe:
                print("[probe done] check: visual_tokens>0, review_tokens covers the answer, "
                      "shift consistent"); return
            continue

        np.savez_compressed(out_path, V=V, H=H,
                            tok_char=np.array(tok_char[:len(rt)], dtype=np.int32),
                            answer_len=len(it.response))
        n_done += 1
        if n_done % 25 == 0:
            dt = time.time() - t0
            print(f"...{n_done} cached ({dt/n_done:.1f}s/item)", flush=True)
    print(f"[done] cached {n_done} new items -> {args.out_dir} in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
