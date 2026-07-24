"""Continuous-score gate for Gemma 4: score = logP(YES) - logP(NO) at the first
generated token (single forward via stream_generate, reading the full logprob vector).
Enables ROC-AUC / PR-AUC / specificity@recall instead of a single greedy point.

Controls via --variant:
  mm            correct target image
  textonly      no images at all (few-shot text + no target image)
  shuffle_perm  each target gets ANOTHER item's image (within-set derangement)

Reuses the prompt construction from mm_gate_gemma4.py.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo

from shroom.data import load_jsonl                                       # noqa: E402
from mm_gate_gemma4 import (SYS_CORE, OUT_CLASSIFY, OUT_BINARY, FEWSHOT,  # noqa: E402
                            qa, uturn, mturn, label_str, rz)
from mlx_vlm import load, stream_generate                                # noqa: E402
from mlx_vlm.utils import load_config                                    # noqa: E402

YES_IDS = [26915, 51327, 10784]   # "YES", " YES", "Yes"
NO_IDS = [7018, 9424, 3771]       # "NO", " NO", "No"


def logsumexp(vals):
    import math
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def derangement(n, seed):
    rng = random.Random(seed)
    idx = list(range(n))
    for _ in range(100):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            return idx
    return idx[1:] + idx[:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mlx-community/gemma-4-12b-it-4bit")
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--train", default="../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--no_classify", action="store_true")
    ap.add_argument("--variant", choices=["mm", "textonly", "shuffle_perm"], default="mm")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--debug", type=int, default=0)
    args = ap.parse_args()

    classify = not args.no_classify
    with_images = (args.variant != "textonly")
    sys_prompt = SYS_CORE + (OUT_CLASSIFY if classify else OUT_BINARY)
    train_by_id = {it.id: it for it in load_jsonl(args.train)}

    fs_imgs, prefix = [], "<bos>" + f"<|turn>system\n{sys_prompt}<turn|>\n"
    for k, (tid, types) in enumerate(FEWSHOT[:args.shots]):
        it = train_by_id[tid]
        if with_images:
            fs_imgs.append(rz(os.path.join(args.image_dir, it.image_name), f"/tmp/_fs_{k}.png"))
        prefix += uturn(qa(it.prompt, it.response), with_images) + mturn(label_str(types, classify))

    items = load_jsonl(args.input)
    if args.max_samples:
        items = items[:args.max_samples]
    # target image assignment
    if with_images:
        tgt_names = [it.image_name for it in items]
        if args.variant == "shuffle_perm":
            perm = derangement(len(items), args.seed)
            tgt_names = [items[perm[i]].image_name for i in range(len(items))]

    import time
    print(f"[load] {args.model_id} shots={args.shots} classify={classify} variant={args.variant}", flush=True)
    model, processor = load(args.model_id); config = load_config(args.model_id)

    def score(prompt, imgs):
        for ch in stream_generate(model, processor, prompt, image=(imgs or None), max_tokens=1):
            lp = ch.logprobs
            sy = logsumexp([float(lp[i]) for i in YES_IDS])
            sn = logsumexp([float(lp[i]) for i in NO_IDS])
            return sy - sn, int(ch.token)
        return 0.0, -1

    t0 = time.time()
    with open(args.output, "w", encoding="utf-8") as out:
        for i, it in enumerate(items):
            prompt = prefix + uturn(qa(it.prompt, it.response), with_images) + \
                "<|turn>model\n<|channel>thought\n<channel|>"
            imgs = list(fs_imgs)
            if with_images:
                imgs.append(rz(os.path.join(args.image_dir, tgt_names[i]), "/tmp/_tgt.png"))
            s, tok = score(prompt, imgs)
            rec = {"id": it.id, "gold_dirty": bool(it.labels), "score": round(s, 4),
                   "first_token": tok, "argmax_yes": tok in YES_IDS}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if args.debug and i < args.debug:
                print(f"[{it.id}] gold_dirty={bool(it.labels)} score={s:+.3f} tok={tok} yes={tok in YES_IDS}",
                      flush=True)
            if (i + 1) % 10 == 0:
                print(f"...{i+1}/{len(items)} ({(time.time()-t0)/(i+1):.1f}s/item)", flush=True)
    print(f"[done] {len(items)} -> {args.output} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
