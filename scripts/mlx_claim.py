"""Claim-level verification (MLX, Method B).

Splits each answer into sentence-level claims (with char offsets), then asks the
model per claim whether the IMAGE supports it. Characters of unsupported claims
become the hallucination spans. Reuses shroom.metrics only at eval time; the
splitting/verdict logic is here.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl                       # noqa: E402
from PIL import Image                                    # noqa: E402
from mlx_vlm import load, generate                       # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template     # noqa: E402
from mlx_vlm.utils import load_config                    # noqa: E402

CAT_MAP = {"INVENTION": "invention", "MISCHARACTERIZATION": "mischaracterization",
           "OCR": "OCR", "MISCOUNTING": "miscounting", "OTHER": "other"}

CLAIM_PROMPT = (
    "You are a strict fact-checker. The IMAGE is the ground truth. AI answers often contain visual "
    "hallucinations. You will judge ONE statement taken from an answer.\n"
    "- If the statement is generic filler, meta-commentary, or has no checkable visual claim "
    "(e.g. 'Certainly!', \"Let's analyze the image\", 'Here is a breakdown'), reply SUPPORTED.\n"
    "- If the statement asserts something VISUAL that is not clearly visible in the image, or that "
    "contradicts the image (wrong object/attribute/count, misread text, invented detail), reply "
    "HALLUCINATION: <invention|mischaracterization|OCR|miscounting|other>.\n"
    "- If its visual claim is clearly true in the image, reply SUPPORTED.\n"
    "Reply on ONE line."
)


def split_claims(text, max_claims=15):
    """Split into sentence-ish claims, returning (start, end) char spans."""
    claims, n, start, i = [], len(text), 0, 0
    while i < n:
        ch = text[i]
        if ch in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?)\"'":
                j += 1
            claims.append((start, j))
            start = j = j
            i = j
        elif ch == "\n":
            if i > start:
                claims.append((start, i))
            start = i + 1
            i += 1
        else:
            i += 1
    if start < n:
        claims.append((start, n))
    out = []
    for a, b in claims:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        # keep only claims with at least 2 alphabetic characters
        if b > a and sum(c.isalpha() for c in text[a:b]) >= 2:
            out.append((a, b))
    return out[:max_claims]


def parse_verdict(s):
    u = s.upper()
    if "HALLUCIN" in u:
        for k, v in CAT_MAP.items():
            if k in u:
                return ("HALL", v)
        return ("HALL", "other")
    if "SUPPORT" in u:
        return ("OK", None)
    return (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mlx-community/gemma-3-12b-it-4bit")
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--max_tokens", type=int, default=16)
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
        dst = "/tmp/_claim_cur.png"
        img.save(dst)
        return dst

    def verify(img_path, prompt, response, claim):
        text = (f"{CLAIM_PROMPT}\n"
                f'QUESTION: "{prompt}"\nANSWER: "{response}"\nSTATEMENT: "{claim}"\nVerdict:')
        formatted = apply_chat_template(processor, config, text, num_images=1)
        kwargs = dict(image=[img_path], max_tokens=args.max_tokens, verbose=False)
        if args.temperature > 0:
            kwargs["temperature"] = args.temperature
        res = generate(model, processor, formatted, **kwargs)
        return (res.text if hasattr(res, "text") else str(res)).strip()

    t0, n_claims = time.time(), 0
    with open(args.output, "w", encoding="utf-8") as out:
        for idx, it in enumerate(items):
            img_path = os.path.join(args.image_dir, it.image_name)
            rec = {"id": it.id, "language": it.language, "response": it.response}
            if not os.path.exists(img_path):
                rec.update(pred_labels=[], char_probs=[], claim_probs=[], error="image_missing")
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            p = resize(img_path)
            claims = split_claims(it.response)
            per_char = [0.0] * len(it.response)
            claim_probs = []
            for (a, b) in claims:
                verdicts = [parse_verdict(verify(p, it.prompt, it.response, it.response[a:b]))
                            for _ in range(args.n_samples)]
                n_claims += args.n_samples
                hall = [c for (v, c) in verdicts if v == "HALL"]
                prob = len(hall) / args.n_samples
                label = Counter(hall).most_common(1)[0][0] if hall else "other"
                claim_probs.append({"start": a, "end": b, "prob": round(prob, 3), "label": label})
                for i in range(a, b):
                    per_char[i] = prob
            rec["claim_probs"] = claim_probs
            rec["char_probs"] = [round(x, 3) for x in per_char]
            rec["pred_labels"] = [c for c in claim_probs if c["prob"] >= args.tau]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if args.debug and idx < args.debug:
                print(f"\n[{it.id}] #gold={len(it.labels)} #claims={len(claims)}", flush=True)
                for c in claim_probs:
                    print(f"    p={c['prob']:.2f} [{c['label']}] {it.response[c['start']:c['end']][:70]!r}", flush=True)
            if (idx + 1) % 20 == 0:
                print(f"...{idx+1}/{len(items)}  claims={n_claims}  "
                      f"({(time.time()-t0)/(idx+1):.1f}s/item)", flush=True)
    print(f"[done] {len(items)} items, {n_claims} claim-checks -> {args.output} in {time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
