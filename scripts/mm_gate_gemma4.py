"""Multimodal few-shot hallucination GATE for Gemma 4 (mlx-vlm).

Each few-shot example is a real dataset sample as its own chat turn WITH its image
(manual interleaving, since apply_chat_template dumps all images at the end). The
assistant label is `NO` or `YES: <type>[, <type>]`. The target turn is thinking-primed;
the model reasons then emits the same verdict, which we parse (binary + types).

Variants (--variant): mm (images), textonly (no images), shuffle (target image replaced).
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl                     # noqa: E402
from PIL import Image                                  # noqa: E402
from mlx_vlm import load, generate                     # noqa: E402
from mlx_vlm.utils import load_config                  # noqa: E402

CATS = ["invention", "mischaracterization", "OCR", "miscounting", "other"]

SYS = (
    "You are a strict multimodal fact-checker. You are shown an IMAGE, plus a QUESTION a user asked "
    "about it and an ANSWER another AI model wrote about that image.\n"
    "Decide whether the ANSWER contains any HALLUCINATION - a statement in the TEXT that is NOT "
    "supported by, or contradicts, what is actually visible in the IMAGE. Trust the IMAGE as ground "
    "truth and look for errors in the text. An answer may be fully correct - then there is NO "
    "hallucination.\n"
    "Hallucination types:\n"
    "- invention: an object/entity/property/detail NOT present in the image (made up).\n"
    "- mischaracterization: something that IS visible but described incorrectly (wrong "
    "attribute/relation/position).\n"
    "- OCR: visible text read or transcribed incorrectly.\n"
    "- miscounting: an incorrect quantity of visible items.\n"
    "- other: any unsupported/contradictory claim fitting none of the above (garbled output, leaked "
    "template tokens, off-topic text).\n"
    "Reply on ONE line, exactly:\n"
    "  NO                         - if the answer is fully supported by the image\n"
    "  YES: <type>[, <type> ...]  - listing EVERY hallucination type present"
)

# (train id, gold verdict label). Q/A/image loaded from the train file at runtime.
FEWSHOT = [
    ("train-en-538", "YES: invention"),
    ("train-en-833", "YES: mischaracterization"),
    ("train-en-583", "YES: miscounting"),
    ("train-en-919", "YES: OCR"),
    ("train-en-656", "NO"),
    ("train-en-551", "NO"),
    ("train-en-904", "NO"),
    ("train-en-700", "NO"),
]


def rz(path, dst):
    im = Image.open(path).convert("RGB"); w, h = im.size
    if max(w, h) > 768:
        s = 768 / max(w, h); im = im.resize((int(w * s), int(h * s)))
    im.save(dst); return dst


def qa(prompt, response):
    return f'Question: "{prompt}"\nAnswer: "{response}"'


def uturn(text, with_image):
    return f"<|turn>user\n{'<|image|>' if with_image else ''}{text}<turn|>\n"


def mturn(label):
    return f"<|turn>model\n{label}<turn|>\n"


def parse_verdict(txt):
    t = (txt or "").strip()
    ms = list(re.finditer(r"\b(YES|NO)\b", t, re.I))
    if not ms:
        return None, [], t
    last = ms[-1]
    if last.group(1).upper() == "NO":
        return False, [], t
    seg = t[last.start():]
    types = [c for c in CATS if re.search(re.escape(c), seg, re.I)]
    return True, types, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mlx-community/gemma-4-12b-it-4bit")
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--train", default="../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--variant", choices=["mm", "textonly", "shuffle"], default="mm")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--debug", type=int, default=0)
    args = ap.parse_args()

    with_images = (args.variant != "textonly")
    train_by_id = {it.id: it for it in load_jsonl(args.train)}

    # resize the 8 few-shot images once; build the static prefix
    fs_imgs = []
    prefix = "<bos>" + f"<|turn>system\n{SYS}<turn|>\n"
    for k, (tid, label) in enumerate(FEWSHOT):
        it = train_by_id[tid]
        if with_images:
            fs_imgs.append(rz(os.path.join(args.image_dir, it.image_name), f"/tmp/_fs_{k}.png"))
        prefix += uturn(qa(it.prompt, it.response), with_images)
        prefix += mturn(label)

    items = load_jsonl(args.input)
    if args.max_samples:
        items = items[:args.max_samples]

    print(f"[load] {args.model_id}  variant={args.variant}  fewshot_imgs={len(fs_imgs)}", flush=True)
    model, processor = load(args.model_id)
    config = load_config(args.model_id)

    shuffle_src = fs_imgs[0] if fs_imgs else None  # a wrong image for the shuffle control
    t0, done = time.time(), 0
    with open(args.output, "w", encoding="utf-8") as out:
        for idx, it in enumerate(items):
            prompt = prefix + uturn(qa(it.prompt, it.response), with_images)
            prompt += "<|turn>model\n<|channel>thought\n<channel|>"
            imgs = list(fs_imgs)
            if with_images:
                tgt = rz(os.path.join(args.image_dir, it.image_name), "/tmp/_tgt.png")
                imgs.append(shuffle_src if args.variant == "shuffle" else tgt)
            raw = generate(model, processor, prompt, image=(imgs or None),
                           max_tokens=args.max_tokens, verbose=False)
            raw = (raw.text if hasattr(raw, "text") else str(raw)).strip()
            dirty, types, _ = parse_verdict(raw)
            gold_types = sorted({s.get("label") for s in it.labels})
            rec = {"id": it.id, "gold_dirty": bool(it.labels), "gold_types": gold_types,
                   "pred_dirty": dirty, "pred_types": types, "raw": raw}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            done += 1
            if args.debug and idx < args.debug:
                print(f"\n[{it.id}] gold_dirty={bool(it.labels)} gold_types={gold_types}\n"
                      f"  Q: {it.prompt}\n  RAW: {raw[:200]!r}\n  -> pred_dirty={dirty} pred_types={types}",
                      flush=True)
            if done % 10 == 0:
                print(f"...{done}/{len(items)} ({(time.time()-t0)/done:.1f}s/item)", flush=True)
    print(f"[done] {done} -> {args.output} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
