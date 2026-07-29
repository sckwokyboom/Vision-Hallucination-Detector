"""Stage 1 (CUDA / H100): cache frozen Gemma-4-12B hidden states for the decoder,
with a precision ladder: --quant bf16 (reference) | int8 | nf4 (bitsandbytes).

Matches extract_features_mlx.py: same review-copy prompt, same layers {24,32,40,47},
same 768px resize, same npz schema (train_connector.py consumes it unchanged), atomic
writes, resumable. Use --probe 2 first to validate token alignment on your setup.

  python scripts/connector/extract_features.py --quant bf16 \
      --train_file splits/subset_quant.jsonl --image_dir ../Shroom-Vision/images \
      --out_dir results/cache_h100_bf16 --h_only
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


def review_body(response):
    """The response exactly as it survives chat templating.

    apply_chat_template rstrips the message content. The review copy ends the prompt, so
    a response with trailing whitespace loses it there — and then the tokenisation of the
    RAW response no longer occurs in input_ids and the id-subsequence search below fails
    (54/3799 en responses end in 1-4 newlines; train-en-414 is the first). Canonicalising
    here makes the string we search for the string that is actually present.

    Only the tail is stripped: gold spans are character offsets into the raw response, so
    trimming the front would silently shift every label. No gold span in any language
    reaches past the rstrip boundary, so nothing supervisable is lost.
    """
    return response.rstrip()


def build_text(prompt, response):
    body = review_body(response)
    return (f"Question: {prompt}\nCandidate answer: {body}\n"
            f"Review token by token:\n{body}")


def encode_review(tok, body):
    """Token ids for the review copy, plus each token's character span in `body`.

    Spans come from the tokenizer's own offset mapping. The obvious alternative —
    decode each token and str.find() it — silently corrupts non-ASCII text: a
    byte-fallback token that splits a multi-byte character (CJK, accented Latin)
    decodes to a replacement character that does not occur in `body`, the search
    misses, the write cursor advances anyway, and every later offset in that item is
    wrong. Measured on the corpus: 32 fr/it/zh items, offsets running up to 1.7x past
    the end of the response; en never triggered it, which is why it survived review.

    The leading "\\n" anchors the review copy to the same token boundary the prompt
    has; its offsets are shifted out and it is dropped by the whitespace skip below.
    """
    enc = tok("\n" + body, add_special_tokens=False, return_offsets_mapping=True)
    offs = [(a - 1, b - 1) for a, b in enc["offset_mapping"]]
    return enc["input_ids"], offs


def find_subseq(hay, needle):
    n, m = len(hay), len(needle)
    for i in range(n - m, -1, -1):          # last occurrence
        if hay[i:i + m] == needle:
            return i
    return -1


def load_model(model_id, quant, device="auto"):
    # device="auto" lets accelerate shard/offload across every visible GPU; an explicit
    # "cuda:N" pins the whole model to one card and fails loudly instead of silently
    # spilling to CPU (which turns a 30-minute extraction into an overnight one).
    kw = dict(device_map=("auto" if device == "auto" else {"": device}),
              trust_remote_code=True)
    if quant == "bf16":
        kw["dtype"] = torch.bfloat16
    else:
        from transformers import BitsAndBytesConfig
        if quant == "int8":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            raise ValueError(quant)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="google/gemma-4-12B-it")
    ap.add_argument("--quant", choices=["bf16", "int8", "nf4"], default="bf16")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", default="24,32,40,47")
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--h_only", action="store_true")
    ap.add_argument("--no_image", action="store_true")
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="'auto' (shard across all visible GPUs) or a single device such as "
                         "'cuda:0'. With CUDA_VISIBLE_DEVICES=3, 'cuda:0' IS physical GPU 3.")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    if args.device != "auto" and args.device.startswith("cuda"):
        idx = int(args.device.split(":")[1]) if ":" in args.device else 0
        n = torch.cuda.device_count()
        if idx >= n:
            raise SystemExit(f"--device {args.device}: only {n} CUDA device(s) visible "
                             f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})")
        print(f"[gpu ] {args.device} = {torch.cuda.get_device_name(idx)}", flush=True)
    print(f"[load] {args.model_id} quant={args.quant} device={args.device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tok = processor.tokenizer
    model = load_model(args.model_id, args.quant, args.device)

    img_tok = None
    for attr in ("image_token_id", "image_token_index"):
        img_tok = getattr(model.config, attr, None) or img_tok

    items = load_jsonl(args.train_file)
    if args.max_items:
        items = items[:args.max_items]

    t0, n_done = time.time(), 0
    failed, no_image, empty = [], [], []
    for idx, it in enumerate(items):
        out_path = os.path.join(args.out_dir, f"{it.id}.npz")
        if os.path.exists(out_path) and not args.probe:
            continue
        img_path = os.path.join(args.image_dir, it.image_name)
        if not os.path.exists(img_path):
            no_image.append(it.id)
            continue
        image = None
        if not args.no_image:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            if max(w, h) > args.max_side:
                s = args.max_side / max(w, h)
                image = image.resize((int(w * s), int(h * s)))

        text = build_text(it.prompt, it.response)
        content = ([{"type": "image"}] if image is not None else []) + \
                  [{"type": "text", "text": text}]
        templ = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=False)
        if image is not None:
            inputs = processor(text=templ, images=image, return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=templ, return_tensors="pt").to(model.device)
        ids = inputs["input_ids"][0].tolist()

        # locate the review copy by id-subsequence (robust to template specials)
        body = review_body(it.response)
        if not body.strip():
            # 4 responses in the corpus are whitespace only. There is nothing to label,
            # so there are no features to cache — legitimate, unlike the skips below.
            empty.append(it.id)
            continue
        ans_ids, offs = encode_review(tok, body)
        pos, trim = find_subseq(ids, ans_ids), 0
        while pos < 0 and trim < 3:
            trim += 1
            pos = find_subseq(ids, ans_ids[trim:])
        if pos < 0:
            # One pathological item must not kill a 40-minute extraction, but a partial
            # cache must never pass for a complete one either: collect and fail at the end.
            failed.append(it.id)
            print(f"[warn] review copy not found for {it.id} — skipped", flush=True)
            continue
        # the match guarantees rt[i] and offs[trim+i] are the same token
        rt, tok_char = list(range(pos, pos + len(ans_ids[trim:]))), offs[trim:]
        while rt and not body[max(0, tok_char[0][0]):max(0, tok_char[0][1])].strip():
            rt.pop(0)
            tok_char = tok_char[1:]
        tok_char = [(max(0, a), max(0, b)) for a, b in tok_char]

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states                            # 0 = embeddings, i = after layer i
        Ls = [min(l + 1, len(hs) - 1) for l in layers]
        stack = torch.stack([hs[l][0] for l in Ls], dim=1)  # [S, L, D]

        H = stack[rt].to(torch.float16).cpu().numpy()
        if args.h_only or img_tok is None or args.no_image:
            V = H[:0]
        else:
            vis = [k for k, t_ in enumerate(ids) if t_ == img_tok]
            V = stack[vis].to(torch.float16).cpu().numpy()

        if args.probe:
            print(f"[probe {it.id}] seq={len(ids)} review_toks={len(rt)} "
                  f"H={H.shape} V={V.shape} ans_chars={len(it.response)} "
                  f"first_piece={tok.decode([ids[rt[0]]])!r}", flush=True)
            if idx + 1 >= args.probe:
                print("[probe done]"); return
            continue

        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, V=V, H=H,
                            tok_char=np.array(tok_char, dtype=np.int32),
                            answer_len=len(it.response))
        os.replace(tmp, out_path)
        n_done += 1
        if n_done % 25 == 0:
            print(f"...{n_done} ({(time.time()-t0)/n_done:.2f}s/item)", flush=True)
    print(f"[done] {n_done} new -> {args.out_dir} in {(time.time()-t0)/60:.1f} min", flush=True)

    # A skipped item is a hole in the cache, and CacheDS drops holes without complaining.
    # Record every skip; abort only on the ones that indicate a bug or missing data.
    if failed or no_image or empty:
        rep = os.path.join(args.out_dir, "_skipped.json")
        with open(rep, "w") as f:
            json.dump({"unaligned": failed, "missing_image": no_image,
                       "empty_response": empty}, f, indent=2)
        print(f"\n[SKIPPED] {len(failed)} unaligned, {len(no_image)} missing image, "
              f"{len(empty)} empty response -> {rep}", flush=True)
        for i in (failed + no_image)[:10]:
            print(f"  {i}", flush=True)
    if failed or no_image:
        raise SystemExit(
            f"ERROR: {len(failed) + len(no_image)} item(s) produced no features. The cache "
            f"is incomplete; fix the cause (see {rep}) and re-run — extraction is resumable.")


if __name__ == "__main__":
    main()
