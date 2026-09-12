"""Eval-time visual-grounding controls for ONE LoRA checkpoint, plus the raw dumps that
every offline sweep (thresholds, min-span-length, merge gap, threshold transfer) reads.

One model load, then one pass over the eval split per condition:

  correct    the item's own image                        (reference)
  shuffled   another item's image — a seeded derangement over the split's unique
             images, so no item keeps its own image (and no image maps to itself)
  no_image   the prompt without any image

Per condition it writes
  <out_dir>/<condition>.raw.jsonl   per item: char probs p, gate g, BIO argmax, type
                                    argmax — everything decode_spans('bio') consumes
  <out_dir>/<condition>.pred.jsonl  trainer-style decode at FIXED thresholds (official
                                    format, no post-processing)
and prints official IoU / Cor per condition. Thresholds default to the ones the
checkpoint selected on its tune split (ck['metrics']); pass --tau/--gate_threshold to
pin the submitted operating point. Post-processing and re-sweeps are deliberately NOT
done here — see scripts/official/ablation_sweeps.py (CPU, seconds).

  python scripts/connector/eval_ablation.py \
      --checkpoint results/lora_h100/a2/best_iou_lora_linear_f24_r16_s13.pt \
      --model_id /workspace/data/models/gemma-4-12B-it \
      --gold splits/dev.en.jsonl --eval_ids splits/en.eval_protocol.json --split tune_dev \
      --image_dir ../Shroom-Vision/images --out_dir results/ablation/en_tune \
      --conditions correct,shuffled,no_image --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "official"))

from shroom.data import load_jsonl                                    # noqa: E402
from model import Connector                                           # noqa: E402
from train_connector import build_example, decode_spans               # noqa: E402
from train_lora import LiveBackbone, single_batch                     # noqa: E402
from lora import inject_lora, load_lora_state, set_lora_training      # noqa: E402
import eval_official                                                  # noqa: E402

CONDITIONS = ("correct", "shuffled", "no_image")


def derangement(names, seed):
    """A permutation of `names` with no fixed point (seeded, deterministic)."""
    names = sorted(set(names))
    if len(names) < 2:
        raise SystemExit("shuffled-image control needs at least two distinct images")
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(len(names))
        if not np.any(perm == np.arange(len(names))):
            return {names[i]: names[int(perm[i])] for i in range(len(names))}


def select_items(gold_path, eval_ids, split):
    items = load_jsonl(gold_path)
    if split == "all" or not eval_ids:
        return items
    proto = json.load(open(eval_ids))
    key = {"tune_dev": "tune_dev", "heldout": "heldout"}[split]
    keep = set(proto.get(key) or proto.get("tune_dev200") or [])
    if not keep:
        raise SystemExit(f"{eval_ids} has no '{key}' ids")
    return [it for it in items if it.id in keep]


def raw_one(bb, dec, item, image_dir, device, max_chars, image_name=None, no_image=False):
    prep = bb.prepare(item, image_dir, image_name=image_name, no_image=no_image)
    if prep is None:
        return None
    H, V = bb.features(prep, grad=False)
    ex = build_example(item, V, H, prep["tok_char"], prep["answer_len"], max_chars=max_chars)
    Hb, Vb, vmask, t2c, inpos, tgt = single_batch(ex, device)
    ql, pl, tl, gl, bl, _ = dec(Hb, Vb, t2c, inpos, vmask)
    n = int(tgt["valid"][0].sum().item())
    return dict(p=torch.sigmoid(pl)[0, :n].cpu().numpy(),
                t=torch.sigmoid(tl)[0, :n].cpu().numpy(),
                g=float(torch.sigmoid(gl)[0].cpu()),
                bio=bl.argmax(-1)[0, :n].cpu().numpy(),
                n_vis=len(prep["vis"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--gold", required=True, help="labeled JSONL (dev split of the track)")
    ap.add_argument("--eval_ids", default=None, help="eval protocol JSON (en)")
    ap.add_argument("--split", choices=["tune_dev", "heldout", "all"], default="tune_dev")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conditions", default="correct,shuffled,no_image")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--gate_threshold", type=float, default=None)
    ap.add_argument("--seed", type=int, default=13, help="derangement seed")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N items")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown condition(s) {bad}; choose from {CONDITIONS}")
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ck_args, ck_metrics = ck.get("args") or {}, ck.get("metrics") or {}
    tau = args.tau if args.tau is not None else ck_metrics.get("tau")
    g_thr = (args.gate_threshold if args.gate_threshold is not None
             else ck_metrics.get("g_thr"))
    if tau is None or g_thr is None:
        raise SystemExit("checkpoint stores no thresholds; pass --tau and --gate_threshold")
    tau, g_thr = float(tau), float(g_thr)
    layers = [int(x) for x in str(ck_args.get("layers", "24,32,40,47")).split(",")]
    single_copy = bool(ck_args.get("single_copy", False))
    max_chars = int(ck_args.get("max_chars", 4000))
    print(f"[ck ] {args.checkpoint} epoch={ck.get('epoch')} tune_iou="
          f"{ck_metrics.get('span_iou')} thresholds: tau={tau} g_thr={g_thr} "
          f"prompt={'single-copy' if single_copy else 'echo'}", flush=True)

    items = select_items(args.gold, args.eval_ids, args.split)
    if args.limit:
        items = items[:args.limit]
    print(f"[data] {len(items)} items from {args.gold} split={args.split}", flush=True)

    bb = LiveBackbone(args.model_id, args.device, layers,
                      max_side=int(ck_args.get("max_side", 768)),
                      quant=ck_args.get("quant", "bf16"), single_copy=single_copy)
    inject_lora(bb.model, from_layer=int(ck_args.get("lora_from", 24)),
                r=int(ck_args.get("lora_r", 16)), alpha=int(ck_args.get("lora_alpha", 32)),
                dropout=0.0)
    if "lora" not in ck:
        raise SystemExit(f"{args.checkpoint} has no LoRA adapter state")
    load_lora_state(bb.model, ck["lora"])
    set_lora_training(bb.model, False)
    D = (bb.model.config.text_config.hidden_size
         if hasattr(bb.model.config, "text_config") else bb.model.config.hidden_size)
    dec = Connector(D, len(layers), dim=int(ck_args.get("dim", 768)),
                    blocks=int(ck_args.get("blocks", 3)),
                    arch=ck_args.get("arch", "linear")).to(args.device)
    dec.load_state_dict(ck["model"], strict=False)
    dec.eval()
    bb.model.eval()

    gold = {it.id: {"id": it.id, "labels": it.labels, "text_len": len(it.response),
                    "image": it.image_name} for it in items}
    shuffle_map = derangement([it.image_name for it in items], args.seed)
    with open(os.path.join(args.out_dir, "shuffle_map.json"), "w") as f:
        json.dump(shuffle_map, f, indent=1, sort_keys=True)
    manifest = dict(checkpoint=args.checkpoint, epoch=ck.get("epoch"), tau=tau, g_thr=g_thr,
                    single_copy=single_copy, gold=args.gold, split=args.split,
                    n_items=len(items), seed=args.seed, conditions=conds)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    summary = {}
    with torch.no_grad():
        for cond in conds:
            t0, rows, preds, n_vis = time.time(), [], {}, []
            for k, it in enumerate(items):
                raw = raw_one(bb, dec, it, args.image_dir, args.device, max_chars,
                              image_name=(shuffle_map[it.image_name] if cond == "shuffled"
                                          else None),
                              no_image=(cond == "no_image"))
                if raw is None:
                    rows.append({"id": it.id, "p": [], "g": 0.0, "bio": [], "targ": []})
                    preds[it.id] = {"id": it.id, "labels": []}
                    continue
                n_vis.append(raw["n_vis"])
                spans = decode_spans("bio", raw["p"], raw["t"], raw["g"], tau, g_thr,
                                     raw["bio"], None, resp_len=len(it.response))
                preds[it.id] = {"id": it.id, "labels": spans}
                rows.append({"id": it.id, "p": [round(float(x), 4) for x in raw["p"]],
                             "g": round(raw["g"], 5), "bio": raw["bio"].tolist(),
                             "targ": raw["t"].argmax(-1).tolist()})
                if (k + 1) % 50 == 0:
                    print(f"  [{cond}] {k + 1}/{len(items)} "
                          f"({(time.time() - t0) / (k + 1):.2f}s/item)", flush=True)
            with open(os.path.join(args.out_dir, f"{cond}.raw.jsonl"), "w",
                      encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(args.out_dir, f"{cond}.pred.jsonl"), "w",
                      encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(preds[it.id], ensure_ascii=False) + "\n")
            per = eval_official.per_item(gold, preds)
            a = eval_official.agg(per)
            empty = sum(1 for it in items if not preds[it.id]["labels"]) / max(1, len(items))
            dirty = [per[i]["iou"] for i in per if not per[i]["clean"]]
            clean_ok = [1.0 if not preds[i]["labels"] else 0.0 for i in per if per[i]["clean"]]
            summary[cond] = dict(IoU=a["IoU"], Cor=a["Cor"], n=a["n"], empty_rate=empty,
                                 dirty_iou=float(np.mean(dirty)) if dirty else float("nan"),
                                 clean_ok=float(np.mean(clean_ok)) if clean_ok else float("nan"),
                                 mean_visual_tokens=float(np.mean(n_vis)) if n_vis else 0.0,
                                 minutes=(time.time() - t0) / 60)
            print(f"[{cond:8}] IoU={a['IoU']:.4f} Cor={a['Cor']:.4f} empty={empty:.2f} "
                  f"dirtyIoU={summary[cond]['dirty_iou']:.3f} cleanOK={summary[cond]['clean_ok']:.2f} "
                  f"visTok={summary[cond]['mean_visual_tokens']:.0f} "
                  f"[{summary[cond]['minutes']:.1f} min]", flush=True)

    floor = float(np.mean([1.0 if not it.labels else 0.0 for it in items]))
    summary["_floor_predict_nothing"] = floor
    summary["_manifest"] = manifest
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nfloor(predict-nothing)={floor:.4f}  fixed thresholds tau={tau} g_thr={g_thr}")
    print(f"{'condition':10}{'IoU':>8}{'Cor':>8}{'empty':>7}{'dirtyIoU':>9}{'cleanOK':>8}")
    for cond in conds:
        s = summary[cond]
        print(f"{cond:10}{s['IoU']:>8.4f}{s['Cor']:>8.4f}{s['empty_rate']:>7.2f}"
              f"{s['dirty_iou']:>9.3f}{s['clean_ok']:>8.2f}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
