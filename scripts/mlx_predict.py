"""MLX (Apple Silicon) prediction driver.

Reuses the tested core (build_prompt / parse_output / aggregate); only the
generation call is MLX-specific. mlx-vlm takes image *paths* (not PIL), so this
is the natural Mac entry point instead of the PIL-based VLMBackend interface.
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl              # noqa: E402
from shroom.predict import build_prompt, build_prompt_original, parse_output  # noqa: E402
from shroom.aggregate import aggregate          # noqa: E402

from PIL import Image                           # noqa: E402
from mlx_vlm import load, generate              # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402
from mlx_vlm.utils import load_config           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_side", type=int, default=0,
                    help="If >0, downscale each image so its longest side <= max_side (RGB temp copy).")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--save_raw", action="store_true",
                    help="Store the N raw model outputs per item (for offline tau/align sweeps).")
    ap.add_argument("--prompt_style", choices=["skeptical", "original"], default="skeptical",
                    help="'original' = the starter label_with_gemma.py prompt (3 categories).")
    ap.add_argument("--eval_ids", default=None,
                    help="json file with {'tune_dev': [...]} — restrict --input to that split, "
                         "so no materialised gold subset has to exist on disk.")
    args = ap.parse_args()

    prompt_fn = build_prompt_original if args.prompt_style == "original" else build_prompt

    tmpdir = tempfile.mkdtemp(prefix="mlx_img_") if args.max_side else None

    def resolve_image(raw_path):
        if not args.max_side:
            return raw_path
        img = Image.open(raw_path).convert("RGB")
        w, h = img.size
        if max(w, h) > args.max_side:
            scale = args.max_side / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        dst = os.path.join(tmpdir, "cur.png")
        img.save(dst)
        return dst

    print(f"[load] {args.model_id}", flush=True)
    model, processor = load(args.model_id)
    config = load_config(args.model_id)

    items = load_jsonl(args.input)
    if args.eval_ids:
        prot = json.load(open(args.eval_ids))
        keep = set(prot.get("tune_dev") or prot.get("tune_dev200") or [])
        items = [it for it in items if it.id in keep]
        print(f"[eval_ids] {len(items)}/{len(keep)} items from {args.eval_ids}", flush=True)
    if args.max_samples:
        items = items[:args.max_samples]

    def gen_once(img_path, text):
        formatted = apply_chat_template(processor, config, text, num_images=1)
        kwargs = dict(image=[img_path], max_tokens=args.max_tokens, verbose=False)
        if args.temperature and args.temperature > 0:
            kwargs["temperature"] = args.temperature
        res = generate(model, processor, formatted, **kwargs)
        return res.text if hasattr(res, "text") else str(res)

    t0 = time.time()
    n_done = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for idx, it in enumerate(items):
            img_path = os.path.join(args.image_dir, it.image_name)
            rec = {"id": it.id, "language": it.language, "response": it.response}
            if not os.path.exists(img_path):
                rec.update(pred_labels=[], char_probs=[], error="image_missing")
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            text = prompt_fn(it.prompt, it.response)
            use_path = resolve_image(img_path)
            raws = [gen_once(use_path, text) for _ in range(args.n_samples)]
            if args.debug and idx < args.debug:
                print(f"\n[{it.id}] Q={it.prompt}\n  RESP={it.response[:140]}\n"
                      f"  GOLD={it.labels}\n  RAW={raws[0][:400]}", flush=True)
            samples = [parse_output(r) for r in raws]
            spans, per_char = aggregate(samples, it.response, tau=args.tau)
            rec["pred_labels"] = spans
            rec["char_probs"] = [round(p, 3) for p in per_char]
            if args.save_raw:
                rec["raw_samples"] = raws
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n_done += 1
            if n_done % 20 == 0:
                dt = time.time() - t0
                print(f"...{n_done}/{len(items)}  ({dt/n_done:.2f}s/item)", flush=True)
    print(f"[done] {n_done} items -> {args.output}  in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
