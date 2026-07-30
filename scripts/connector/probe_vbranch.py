"""Autopsy of the connector's visual branch on a REAL checkpoint and REAL cache.

Answers, with numbers, why correct-V and shuffled-V evaluations coincide:

  1. alpha            — the trained ReZero gate value;
  2. residual ratio   — ||alpha * fuse(...)|| / ||h|| per item (how much the visual
                        branch contributes to the representation at all);
  3. sensitivity      — max |Delta logit| of the char-probability head under
                        correct V vs shuffled V vs zeroed V;
  4. forced alpha=1   — the same sensitivities with the gate forced open, which
                        separates "gate closed" (small 1-3, large 4) from
                        "wiring broken" (small everywhere).

Decoder-only: needs the cached features, not Gemma. CPU is fine.

  python scripts/connector/probe_vbranch.py \
      --ckpt results/uva_h100/best_iou_connector_bio_tv_bio_gc_ctr_notype_s13.pt \
      --cache_dir results/cache_h100_hv \
      --train_file ../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl \
      --eval_ids splits/en.eval_protocol.json --n 64
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shroom.data import load_jsonl                      # noqa: E402
from model import Connector                             # noqa: E402
from train_connector import CacheDS, collate            # noqa: E402


def branch_parts(model, H, V, vmask):
    """Mirror of Connector.forward's connector branch, up to the fused z."""
    h = model.proj_h(model.mix_h(H))
    v = model.proj_v(model.mix_v(V))
    c = h
    for blk in model.blocks:
        c = blk(c, v, vmask)
    contrib = model.alpha * model.fuse(torch.cat([h, c, h - c, h * c], dim=-1))
    return h, contrib


@torch.no_grad()
def probe_batch(model, batch, device="cpu"):
    H, V, vmask, t2c, inpos, y, ytype, bio, segs, valid, gate, items = batch
    H, V, vmask = H.to(device), V.to(device), vmask.to(device)
    t2c, inpos = t2c.to(device), inpos.to(device)

    h, contrib = branch_parts(model, H, V, vmask)
    ratio = (contrib.norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-9)).mean().item()

    def p_logits(Vx, vm):
        return model(H, Vx, t2c, inpos, vm)[1]

    p_ok = p_logits(V, vmask)
    p_sh = p_logits(torch.roll(V, 1, 0), torch.roll(vmask, 1, 0))
    p_z = p_logits(torch.zeros_like(V), vmask)
    return dict(ratio=ratio,
                d_shuf=(p_ok - p_sh).abs().max().item(),
                d_zero=(p_ok - p_z).abs().max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_ids", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--blocks", type=int, default=3)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device)
    state = ck["model"] if "model" in ck else ck
    if "alpha" not in state:
        raise SystemExit("checkpoint has no alpha -> not a connector-arch checkpoint")

    proto = json.load(open(args.eval_ids))
    ids = set(proto["tune_dev"])
    items = [it for it in load_jsonl(args.train_file) if it.id in ids]
    ds = CacheDS(items, args.cache_dir, max_chars=4000)
    if len(ds) == 0:
        raise SystemExit(f"no cached items found in {args.cache_dir}")
    z0 = ds[0]
    if z0["V"].shape[0] == 0:
        raise SystemExit(f"{args.cache_dir} is an --h_only cache (V empty); use the H+V cache")
    D, L = z0["H"].shape[2], z0["H"].shape[1]

    model = Connector(D, L, dim=args.dim, blocks=args.blocks, arch="connector").to(args.device)
    missing, _ = model.load_state_dict(state, strict=False)
    model.eval()
    if any(k.startswith(("proj_v", "mix_v", "blocks", "fuse")) for k in missing):
        raise SystemExit(f"checkpoint is missing visual-branch weights: {missing[:6]}")

    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    batches = []
    for b in dl:
        batches.append(b)
        if sum(x[0].shape[0] for x in batches) >= args.n:
            break

    alpha = float(model.alpha.detach())
    print(f"ckpt: {args.ckpt}")
    print(f"[1] alpha (init 0.0) ............... {alpha:+.6e}")

    def sweep(tag):
        rs, dsh, dz = [], [], []
        for b in batches:
            r = probe_batch(model, b, args.device)
            rs.append(r["ratio"]); dsh.append(r["d_shuf"]); dz.append(r["d_zero"])
        print(f"[2] residual ratio ||aF||/||h|| .... mean {sum(rs)/len(rs):.3e}   ({tag})")
        print(f"[3] max|dlogit| shuffled V ......... {max(dsh):.3e}")
        print(f"    max|dlogit| zeroed  V ......... {max(dz):.3e}")

    sweep(f"trained alpha={alpha:+.4e}")
    with torch.no_grad():
        model.alpha.fill_(1.0)
    print("[4] --- forced alpha = 1.0 ---")
    sweep("forced")
    print("\nverdict guide: trained sens ~1e-7 AND forced sens >1e-2  -> gate closed, wiring fine")
    print("               forced sens also tiny                     -> wiring bug, dig deeper")


if __name__ == "__main__":
    main()
