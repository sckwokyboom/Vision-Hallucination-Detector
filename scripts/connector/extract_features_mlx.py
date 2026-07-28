"""Stage 1 (Apple Silicon): cache frozen Gemma-4-12b (MLX, 4-bit) features for the connector.

Same output schema as extract_features.py (V/H/tok_char/answer_len npz), so
train_connector.py consumes it unchanged. Uses mlx-vlm's native hidden-state capture
(capture_layer_ids + hidden_sink) — no monkeypatching.

Run --probe 2 first: prints sink structure / token counts / alignment.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shroom.data import load_jsonl                      # noqa: E402

from mlx_vlm import load                                # noqa: E402
from mlx_vlm.utils import load_config, prepare_inputs   # noqa: E402


def build_text(prompt, response):
    return (f"Question: {prompt}\nCandidate answer: {response}\n"
            f"Review token by token:\n{response}")


def build_template(text):
    return f"<bos><|turn>user\n<|image|>{text}<turn|>\n"


def find_subseq(hay, needle):
    n, m = len(hay), len(needle)
    for i in range(n - m, -1, -1):          # last occurrence
        if hay[i:i + m] == needle:
            return i
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mlx-community/gemma-4-12b-it-4bit")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", default="24,32,40,47")   # 48-layer model: valid ids 0..47
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--no_image", action="store_true", help="true text-only cache (no image in prompt)")
    ap.add_argument("--h_only", action="store_true", help="do not store V (image still in context; for linear-readout track, ~6x smaller)")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[load] {args.model_id}", flush=True)
    model, processor = load(args.model_id)
    config = load_config(args.model_id)
    tok = getattr(processor, "tokenizer", processor)
    img_tok = getattr(config, "image_token_id", None) or getattr(config, "image_token_index", None)
    if img_tok is None:
        img_tok = getattr(model.config, "image_token_id", None)
    assert img_tok is not None, "no image_token_id found"
    print(f"[cfg] image_token_id={img_tok}  layers={layers}", flush=True)

    items = load_jsonl(args.train_file)
    if args.max_items:
        items = items[:args.max_items]

    t0, n_done = time.time(), 0
    for idx, it in enumerate(items):
        out_path = os.path.join(args.out_dir, f"{it.id}.npz")
        if os.path.exists(out_path) and not args.probe:
            continue
        img_path = os.path.join(args.image_dir, it.image_name)
        if not os.path.exists(img_path):
            continue
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        if max(w, h) > args.max_side:
            s = args.max_side / max(w, h)
            im = im.resize((int(w * s), int(h * s)))
        im.save("/tmp/_fx.png")

        text = build_text(it.prompt, it.response)
        templ = (f"<bos><|turn>user\n{text}<turn|>\n" if args.no_image else build_template(text))
        inputs = prepare_inputs(processor, images=(None if args.no_image else ["/tmp/_fx.png"]),
                                prompts=[templ], image_token_index=img_tok)
        ids = np.array(inputs["input_ids"][0].tolist())

        # review-copy token span: tokenize the answer alone (no specials) and find the
        # LAST occurrence of that id-subsequence in the full ids
        ans_ids = tok.encode("\n" + it.response, add_special_tokens=False)
        # drop the leading newline token(s) by re-searching with a 1-token trim fallback
        pos = find_subseq(ids.tolist(), ans_ids)
        trim = 0
        while pos < 0 and trim < 3:
            trim += 1
            pos = find_subseq(ids.tolist(), ans_ids[trim:])
        assert pos >= 0, f"review copy ids not found for {it.id}"
        sub = ans_ids[trim:]
        rt = list(range(pos, pos + len(sub)))
        # drop leading whitespace-only tokens (e.g. the '\n' before the answer)
        while rt and not tok.decode([int(ids[rt[0]])]).strip():
            rt.pop(0)

        # char offsets for the review tokens: incremental decode
        tok_char, cursor = [], 0
        for k in rt:
            piece = tok.decode([int(ids[k])])
            j = it.response.find(piece, cursor) if piece.strip() else cursor
            if j < 0:
                j = cursor
            tok_char.append((j, j + len(piece)))
            cursor = j + len(piece)

        sink = []
        kwargs = {k: v for k, v in inputs.items()
                  if k not in ("input_ids", "pixel_values", "attention_mask")}
        out = model(input_ids=inputs["input_ids"], pixel_values=(None if args.no_image else inputs.get("pixel_values")),
                    mask=inputs.get("attention_mask"),
                    capture_layer_ids=layers, hidden_sink=sink, **kwargs)
        mx.eval([s[1] if isinstance(s, tuple) else s for s in sink])

        # normalize sink entries -> {layer: [S,D]}
        cap = {}
        for e_i, e in enumerate(sink):
            if isinstance(e, tuple) and len(e) == 2:
                cap[int(e[0])] = np.array(e[1][0].astype(mx.float16))
            else:
                cap[layers[e_i % len(layers)]] = np.array(e[0].astype(mx.float16))
        Ls = [l for l in layers if l in cap]
        stack = np.stack([cap[l] for l in Ls], axis=1)        # [S, L, D]

        vis_pos = np.nonzero(ids == img_tok)[0]
        V = stack[vis_pos][:0] if args.h_only else stack[vis_pos]
        H = stack[rt]

        if args.probe:
            print(f"[probe {it.id}] seq={len(ids)}  sink_entries={len(sink)} "
                  f"captured_layers={Ls}  visual={len(vis_pos)}  review_toks={len(rt)}  "
                  f"V={V.shape} H={H.shape}  ans_chars={len(it.response)}", flush=True)
            print(f"   first review piece: {tok.decode([int(ids[rt[0]])])!r} "
                  f"answer starts: {it.response[:20]!r}", flush=True)
            if idx + 1 >= args.probe:
                print("[probe done]"); return
            continue

        tmp_path = out_path + ".tmp.npz"
        np.savez_compressed(tmp_path, V=V.astype(np.float16), H=H.astype(np.float16),
                            tok_char=np.array(tok_char, dtype=np.int32),
                            answer_len=len(it.response))
        os.replace(tmp_path, out_path)
        n_done += 1
        if n_done % 20 == 0:
            print(f"...{n_done} cached ({(time.time()-t0)/n_done:.1f}s/item)", flush=True)
    print(f"[done] {n_done} new -> {args.out_dir} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
