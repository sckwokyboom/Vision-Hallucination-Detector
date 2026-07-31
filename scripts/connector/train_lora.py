"""A2/A3 of the Unified Visual Adaptation matrix: LoRA-adapted Gemma backbone with a
live forward per step (no feature cache — the cache is only valid for a frozen model).

  A2:  --arch linear      answer states only, adapted backbone
  A3:  --arch connector   answer states + visual-token cross-attention, adapted backbone

Shares with the cached pipeline everything that must not drift: alignment
(extract_features.review_body/encode_review/find_subseq), supervision
(train_connector.build_example), evaluation (train_connector.collate + run_eval on
precomputed features) and the decoder (model.Connector). New here: the live backbone
wrapper, hand-rolled LoRA (lora.py) and the v3 loss re-stated for batch-size-1
accumulation — WITHOUT the contrastive term, which needs same-image pairs inside one
graph; the paired frozen baseline is therefore the minus-contrastive ablation
(summary_linear_bio_tv_bio_gc_notype_s13), not full v3.

  python scripts/connector/train_lora.py --model_id /path/gemma-4-12B-it \
      --train_file .../shroom-vision.train.en.labeled.jsonl --image_dir .../images \
      --eval_ids splits/en.eval_protocol.json --out_dir results/lora_h100/a2 \
      --arch linear --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shroom.data import load_jsonl                                    # noqa: E402
from model import Connector, tversky_loss, ranking_loss              # noqa: E402
from train_connector import build_example, collate, run_eval, decode_spans  # noqa: E402
from extract_features import (build_text, review_body, encode_review,  # noqa: E402
                              find_subseq, load_model)
from lora import (inject_lora, lora_parameters, lora_state_dict,      # noqa: E402
                  load_lora_state, set_lora_training)


# ------------------------------------------------------------------- live backbone
class LiveBackbone:
    """Frozen(+LoRA) Gemma: item -> (H, V) torch tensors on device, with the autograd
    graph reaching the LoRA parameters when grad=True."""

    def __init__(self, model_id, device, layers, max_side=768, quant="bf16"):
        from transformers import AutoProcessor
        from PIL import Image
        self.Image = Image
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.tok = self.processor.tokenizer
        assert getattr(self.tok, "is_fast", False), "offset mapping needs a fast tokenizer"
        self.model = load_model(model_id, quant, device)               # frozen, eval mode
        self.layers = layers
        self.max_side = max_side
        self.img_tok = None
        for attr in ("image_token_id", "image_token_index"):
            self.img_tok = getattr(self.model.config, attr, None) or self.img_tok

    def prepare(self, it, image_dir):
        """Tokenise + align one item; None if it cannot be trained on (same skip rules
        as extract_features: whitespace-only response, missing image, no alignment)."""
        body = review_body(it.response)
        if not body.strip():
            return None
        img_path = os.path.join(image_dir, it.image_name)
        if not os.path.exists(img_path):
            return None
        image = self.Image.open(img_path).convert("RGB")
        w, h = image.size
        if max(w, h) > self.max_side:
            s = self.max_side / max(w, h)
            image = image.resize((int(w * s), int(h * s)))
        text = build_text(it.prompt, it.response)
        templ = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=templ, images=image, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        ans_ids, offs = encode_review(self.tok, body)
        pos, trim = find_subseq(ids, ans_ids), 0
        while pos < 0 and trim < 3:
            trim += 1
            pos = find_subseq(ids, ans_ids[trim:])
        if pos < 0:
            return None
        rt, tok_char = list(range(pos, pos + len(ans_ids[trim:]))), offs[trim:]
        while rt and not body[max(0, tok_char[0][0]):max(0, tok_char[0][1])].strip():
            rt.pop(0)
            tok_char = tok_char[1:]
        tok_char = [(max(0, a), max(0, b)) for a, b in tok_char]
        vis = [k for k, t_ in enumerate(ids) if t_ == self.img_tok]
        return dict(inputs=inputs, rt=rt, vis=vis, tok_char=tok_char,
                    answer_len=len(it.response))

    def features(self, prep, grad=True):
        """One forward. H [T,L,D], V [P,L,D], float32 (decoder dtype)."""
        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) for k, v in prep["inputs"].items()}
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = self.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            Ls = [min(l + 1, len(hs) - 1) for l in self.layers]
            stack = torch.stack([hs[l][0] for l in Ls], dim=1)         # [S, L, D]
            H = stack[prep["rt"]].float()
            V = (stack[prep["vis"]].float() if prep["vis"]
                 else stack[:0].float())
        return H, V


# ------------------------------------------------------------------- v3 loss (b=1)
def v3_loss(outs, tgt, lam=(1.0, 1.0, 0.5, 0.2), l_bio=0.5, l_consist=0.3):
    """train_connector's inline v3 loss, restated for live training: BCE on soft gold,
    Tversky, ranking, gate BCE, BIO CE (w=[1,4,2]), gate-consistency (MSE of
    sigmoid(gate) vs per-example top-k mean). No contrastive (needs same-image pairs
    in one graph), no type loss (v3 runs --no_type_loss)."""
    ql, pl, tl, gl, bl, _ = outs
    y, bio_t, valid, gate = tgt["y"], tgt["bio"], tgt["valid"], tgt["gate"]
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def exmean(x):
        return ((x * valid).sum(1) / valid.sum(1).clamp(min=1)).mean()

    m = (y > 0).float()
    loss = (lam[0] * exmean(bce(pl, y))
            + lam[1] * tversky_loss(ql, m, valid)
            + lam[2] * ranking_loss(pl, y, valid)
            + lam[3] * bce(gl, gate).mean())
    w = torch.tensor([1.0, 4.0, 2.0], device=bl.device)
    ce = nn.functional.cross_entropy(bl.reshape(-1, 3), bio_t.reshape(-1),
                                     weight=w, reduction="none").reshape(bio_t.shape)
    loss = loss + l_bio * exmean(ce)
    p_sig = torch.sigmoid(pl) * valid
    topk = torch.stack([
        (p_sig[bi][valid[bi] > 0].topk(max(3, int(valid[bi].sum()) // 20))[0].mean()
         if int(valid[bi].sum()) >= 3 else p_sig[bi].max())
        for bi in range(p_sig.shape[0])])
    loss = loss + l_consist * nn.functional.mse_loss(torch.sigmoid(gl), topk)
    return loss


def focal_bce_with_logits(logits, target, gamma=2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1 - prob) * (1 - target)
    return ((1 - pt).clamp(min=1e-6) ** gamma * bce).mean()


def cascade_loss(outs, tgt, lam=(1.0, 1.0, 0.5), l_gate=1.0, l_bio=0.5,
                 focal_gamma=2.0):
    """Shared LoRA backbone, independent tasks: gate on every item, locator only dirty."""
    ql, pl, tl, gl, bl, _ = outs
    y, bio_t, valid, gate = tgt["y"], tgt["bio"], tgt["valid"], tgt["gate"]
    loss = l_gate * focal_bce_with_logits(gl, gate, gamma=focal_gamma)
    if float(gate.detach().mean()) < 0.5:
        return loss

    bce = nn.BCEWithLogitsLoss(reduction="none")

    def exmean(x):
        return ((x * valid).sum(1) / valid.sum(1).clamp(min=1)).mean()

    m = (y > 0).float()
    loss = loss + (lam[0] * exmean(bce(pl, y))
                   + lam[1] * tversky_loss(ql, m, valid)
                   + lam[2] * ranking_loss(pl, y, valid))
    w = torch.tensor([1.0, 4.0, 2.0], device=bl.device)
    ce = nn.functional.cross_entropy(bl.reshape(-1, 3), bio_t.reshape(-1),
                                     weight=w, reduction="none").reshape(bio_t.shape)
    return loss + l_bio * exmean(ce)


def single_batch(ex, device):
    """One live example -> model inputs + targets, batch dim 1. H/V keep their graph
    (this is why the cached collate, which round-trips numpy, cannot be used here)."""
    H = ex["H"].unsqueeze(0).to(device)                    # [1,T,L,D]
    V = ex["V"].unsqueeze(0).to(device)                    # [1,P,L,D] (P may be 0)
    vmask = torch.zeros(1, V.shape[1], dtype=torch.bool, device=device)
    t2c = torch.from_numpy(ex["t2c"]).unsqueeze(0).to(device)
    inpos = torch.from_numpy(ex["inpos"]).unsqueeze(0).to(device)
    tgt = dict(
        y=torch.from_numpy(ex["y"]).unsqueeze(0).to(device),
        bio=torch.from_numpy(ex["bio"]).unsqueeze(0).to(device),
        valid=torch.from_numpy((ex["t2c"] >= 0).astype(np.float32)).unsqueeze(0).to(device),
        gate=torch.tensor([ex["gate"]], device=device))
    return H, V, vmask, t2c, inpos, tgt


class MemDS(torch.utils.data.Dataset):
    """Precomputed (numpy) live features for the eval split, CacheDS-shaped, so the
    stock collate + run_eval work unchanged."""

    def __init__(self, examples):
        self.ex = examples
        self.items = [e["item"] for e in examples]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def eval_epoch(bb, dec, ev, image_dir, device, max_chars, batch=16):
    bb.model.eval()
    dec.eval()
    exs = []
    for it in ev:
        prep = bb.prepare(it, image_dir)
        if prep is None:
            continue
        H, V = bb.features(prep, grad=False)
        exs.append(build_example(it, V.cpu().numpy().astype(np.float16),
                                 H.cpu().numpy().astype(np.float16),
                                 prep["tok_char"], prep["answer_len"], max_chars))
    dl = torch.utils.data.DataLoader(MemDS(exs), batch_size=batch,
                                     shuffle=False, collate_fn=collate)
    return run_eval(dec, dl, device, decoder="bio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--eval_ids", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arch", choices=["linear", "connector"], default="linear")
    ap.add_argument("--layers", default="24,32,40,47")
    ap.add_argument("--lora_from", type=int, default=24,
                    help="first adapted layer; 24 covers every tapped layer, 40 = top-8 only")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--freeze_lora_epochs", type=int, default=1,
                    help="decoder-only warm-up epochs before LoRA unfreezes")
    ap.add_argument("--cascade_train", action="store_true",
                    help="train gate on all examples and locator/BIO only on dirty examples")
    ap.add_argument("--lambda_gate", type=float, default=1.0)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--l_bio", type=float, default=0.5)
    ap.add_argument("--accum", type=int, default=16, help="grad accumulation (live batch is 1)")
    ap.add_argument("--lr_dec", type=float, default=3e-4)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_eval", type=int, default=None)
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--init_from", default=None, help="warm-start decoder from a checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    proto = json.load(open(args.eval_ids))
    tune_ids = set(proto["tune_dev"])
    held = set(proto.get("heldout", []))
    items = load_jsonl(args.train_file)
    ev = [it for it in items if it.id in tune_ids]
    tr = [it for it in items if it.id not in tune_ids and it.id not in held]
    if args.max_train:
        tr = tr[:args.max_train]
    if args.max_eval:
        ev = ev[:args.max_eval]
    print(f"[data] train={len(tr)} tune_dev={len(ev)} (heldout untouched)", flush=True)

    bb = LiveBackbone(args.model_id, args.device, layers)
    wrapped = inject_lora(bb.model, from_layer=args.lora_from, r=args.lora_r,
                          alpha=args.lora_alpha, dropout=args.lora_dropout)
    set_lora_training(bb.model, False)
    print(f"[lora] wrapped {len(wrapped)} projections from layer {args.lora_from} "
          f"(r={args.lora_r}, alpha={args.lora_alpha})", flush=True)

    D = (bb.model.config.text_config.hidden_size
         if hasattr(bb.model.config, "text_config") else bb.model.config.hidden_size)
    dec = Connector(D, len(layers), dim=args.dim, blocks=args.blocks,
                    arch=args.arch).to(args.device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=args.device)
        dec.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
        print(f"[dec ] warm-started from {args.init_from}", flush=True)
        if "lora" in ck:
            # resuming one of OUR checkpoints: restore the adapters too, so
            # --epochs 0 --init_from <best> reproduces and re-scores that model
            n_res = load_lora_state(bb.model, ck["lora"])
            print(f"[lora] restored {n_res} adapter tensors", flush=True)

    lp = lora_parameters(bb.model)
    opt = torch.optim.AdamW([{"params": dec.parameters(), "lr": args.lr_dec},
                             {"params": lp, "lr": args.lr_lora}])
    n_dec = sum(p.numel() for p in dec.parameters())
    n_lora = sum(p.numel() for p in lp)
    print(f"[model] arch={args.arch} decoder={n_dec/1e6:.1f}M lora={n_lora/1e6:.1f}M",
          flush=True)

    frozen_run = args.freeze_lora_epochs >= args.epochs      # continuation control: no LoRA ever
    tag = (f"lora_{args.arch}{'_cascade' if args.cascade_train else ''}"
           f"{'_frozen' if frozen_run else ''}"
           f"_f{args.lora_from}_r{args.lora_r}_s{args.seed}")
    best_iou, best_path, t0 = -1.0, None, time.time()
    rng = np.random.default_rng(args.seed)

    if args.init_from and args.epochs > 0:
        # step-0 check: a warm-started decoder must reproduce its source checkpoint's
        # dev score through the LIVE pipeline (fp16-cache vs live-fp32 rounding aside).
        # A big gap here means weights didn't load or the live features differ.
        _, m0 = eval_epoch(bb, dec, ev, args.image_dir, args.device, args.max_chars)
        print(f"[ep 0] warm-start reproduction: iou={m0['span_iou']:.4f} "
              f"Cor={m0['official_cor']:.3f} poolR={m0['pooled_pearson_debug']:.3f} "
              f"(compare against the source checkpoint's dev iou)",
              flush=True)

    for ep in range(1, args.epochs + 1):
        lora_on = ep > args.freeze_lora_epochs
        for p in lp:
            p.requires_grad_(lora_on)
        set_lora_training(bb.model, lora_on)
        dec.train()
        order = rng.permutation(len(tr))
        run_loss, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, idx in enumerate(order):
            prep = bb.prepare(tr[idx], args.image_dir)
            if prep is None:
                continue
            H, V = bb.features(prep, grad=lora_on)
            ex = build_example(tr[idx], V, H, prep["tok_char"], prep["answer_len"],
                               max_chars=args.max_chars)
            Hb, Vb, vmask, t2c, inpos, tgt = single_batch(ex, args.device)
            outs = dec(Hb, Vb, t2c, inpos, vmask)
            raw_loss = (cascade_loss(outs, tgt, l_gate=args.lambda_gate,
                                     l_bio=args.l_bio, focal_gamma=args.focal_gamma)
                        if args.cascade_train else v3_loss(outs, tgt, l_bio=args.l_bio))
            loss = raw_loss / args.accum
            loss.backward()
            run_loss += float(loss.detach()) * args.accum
            seen += 1
            if seen % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [q for g in opt.param_groups for q in g["params"]], args.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if seen % 200 == 0:
                print(f"[ep {ep} b{seen}/{len(tr)}] loss={run_loss/seen:.4f}", flush=True)
        opt.step()
        opt.zero_grad(set_to_none=True)

        _, m = eval_epoch(bb, dec, ev, args.image_dir, args.device, args.max_chars)
        print(f"[ep {ep}] loss={run_loss/max(seen,1):.4f} dev: iou={m['official_iou']:.4f} "
              f"(fl={m['floor']:.3f}, tau={m['tau']}, g={m['g_thr']}) "
              f"dirty={m['dirty_iou']:.3f} cleanOK={m['clean_empty']:.2f} "
              f"gateRec={m['dirty_gate_recall']:.2f} gateSpec={m['gate_specificity']:.2f} "
              f"Cor={m['official_cor']:.3f} Cor_lbl={m['official_cor_lbl']:.3f} "
              f"poolR={m['pooled_pearson_debug']:.3f} lora={'on' if lora_on else 'warmup'} "
              f"loss={'cascade' if args.cascade_train else 'v3'} "
              f"[{(time.time()-t0)/60:.1f}m]", flush=True)
        if m["span_iou"] > best_iou:
            best_iou = m["span_iou"]
            best_path = os.path.join(args.out_dir, f"best_iou_{tag}.pt")
            torch.save({"model": dec.state_dict(), "lora": lora_state_dict(bb.model),
                        "args": vars(args), "epoch": ep, "metrics": m},
                       best_path)

    if best_path:
        ck = torch.load(best_path, map_location=args.device)
        dec.load_state_dict(ck["model"], strict=False)
        load_lora_state(bb.model, ck["lora"])
        print(f"[best] loaded epoch {ck.get('epoch')} from {best_path} for final predictions",
              flush=True)
    per, m = eval_epoch(bb, dec, ev, args.image_dir, args.device, args.max_chars)
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump({"variant": tag, "metrics": m, "best_iou_seen": best_iou,
                   "best_checkpoint": best_path}, f, indent=2)
    pred_path = os.path.join(args.out_dir, f"dev_pred_{tag}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, (it, q, p, t, g, bt, qsp) in per.items():
            spans = decode_spans("bio", p, t, g, m["tau"], m["g_thr"], bt, qsp,
                                 resp_len=len(it.response))
            f.write(json.dumps({"id": i, "labels": spans,          # official field
                                "language": it.language, "response": it.response,
                                "pred_labels": spans,
                                "char_probs": [round(float(x), 3) for x in p]},
                               ensure_ascii=False) + "\n")
    print(f"DONE {tag}: final iou={m['span_iou']:.4f} best={best_iou:.4f} "
          f"-> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
