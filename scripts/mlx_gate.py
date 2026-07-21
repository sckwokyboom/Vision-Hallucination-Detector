"""Clean-gate runner (MLX): a short YES/NO question per item — 'does the answer
contain ANY hallucination?'. Stores prob_dirty = fraction of samples saying YES,
so the offline sweep can gate span extraction on it.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl                       # noqa: E402
from PIL import Image                                    # noqa: E402
from mlx_vlm import load, generate                       # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template     # noqa: E402
from mlx_vlm.utils import load_config                    # noqa: E402

GATE_PROMPT = (
    "You are a strict fact-checker. The IMAGE is the ground truth.\n"
    "Below is a QUESTION about the image and an ANSWER produced by another AI model.\n"
    "Does the answer contain ANY statement that is not supported by — or contradicts — the image?\n"
    "Think about whether every claim is truly visible in the image.\n"
    "Reply with exactly one word: YES (there is at least one unsupported statement) or NO."
)


def parse_yesno(s):
    m = re.search(r"\b(YES|NO)\b", s.upper())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--debug", type=int, default=0)
    args = ap.parse_args()

    print(f"[load] {args.model_id}", flush=True)
    model, processor = load(args.model_id)
    config = load_config(args.model_id)

    items = load_jsonl(args.input)
    if args.max_samples:
        items = items[:args.max_samples]

    def resize(path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > args.max_side:
            sc = args.max_side / max(w, h)
            img = img.resize((max(1, int(w * sc)), max(1, int(h * sc))))
        dst = "/tmp/_gate_cur.png"
        img.save(dst)
        return dst

    def gen(img_path, text):
        formatted = apply_chat_template(processor, config, text, num_images=1)
        kwargs = dict(image=[img_path], max_tokens=args.max_tokens, verbose=False)
        if args.temperature > 0:
            kwargs["temperature"] = args.temperature
        res = generate(model, processor, formatted, **kwargs)
        return (res.text if hasattr(res, "text") else str(res)).strip()

    t0 = time.time()
    with open(args.output, "w", encoding="utf-8") as out:
        for idx, it in enumerate(items):
            img_path = os.path.join(args.image_dir, it.image_name)
            rec = {"id": it.id, "language": it.language}
            if not os.path.exists(img_path):
                rec.update(prob_dirty=1.0, gate_raw=[], error="image_missing")
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            p = resize(img_path)
            text = (f"{GATE_PROMPT}\n\n"
                    f'QUESTION: "{it.prompt}"\nANSWER: "{it.response}"\nAnswer (YES/NO):')
            raws = [gen(p, text) for _ in range(args.n_samples)]
            votes = [parse_yesno(r) for r in raws]
            yes = sum(1 for v in votes if v == "YES")
            rec["prob_dirty"] = yes / args.n_samples
            rec["gate_raw"] = raws
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if args.debug and idx < args.debug:
                print(f"[{it.id}] #gold={len(it.labels)} votes={votes} "
                      f"prob_dirty={rec['prob_dirty']:.2f}", flush=True)
            if (idx + 1) % 20 == 0:
                print(f"...{idx+1}/{len(items)} ({(time.time()-t0)/(idx+1):.2f}s/item)", flush=True)
    print(f"[done] {len(items)} -> {args.output} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
