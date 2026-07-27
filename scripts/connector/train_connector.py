"""Stage 2: train the connector + heads on cached frozen-Gemma features.

Variants (the spec's experiment grid):
  --arch linear        frozen readout baseline (no cross-attention, no image)
  --arch connector     main method (cross-attention over visual patches)
  --no_image           text-only control (V zeroed at train AND eval)
  --eval_shuffle       extra eval of the trained model with deranged images (grounding check)

Loss (per-example averaged, then batch): l1*BCE_soft + l2*softJaccard + l3*ranking
+ l4*BCE_type + l5*BCE_gate. Postprocessing: threshold sweep on dev -> spans, argmax
type, merge same-type neighbours -> official JSONL. Metrics: span_iou vs floor,
Pearson calib (Cor), per-type ROC & label-aware calib (Cor lbl approx.), ROC/PR-AUC.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shroom.data import load_jsonl                              # noqa: E402
from shroom.split import group_split_by_image                   # noqa: E402
from shroom.metrics import gold_char_probs, char_iou, pearson   # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Connector, soft_jaccard_loss, ranking_loss, CATS  # noqa: E402


# --------------------------------------------------------------------------- data
class CacheDS(Dataset):
    def __init__(self, items, cache_dir, max_chars=1200, max_vis=1200):
        self.items = [it for it in items
                      if os.path.exists(os.path.join(cache_dir, f"{it.id}.npz"))]
        self.dir = cache_dir
        self.max_chars, self.max_vis = max_chars, max_vis

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        z = np.load(os.path.join(self.dir, f"{it.id}.npz"))
        V, H, tc = z["V"], z["H"], z["tok_char"]
        n_ch = min(int(z["answer_len"]), self.max_chars)
        # char-level gold
        cp = gold_char_probs(it.labels, int(z["answer_len"]))
        y = np.zeros(n_ch, dtype=np.float32)
        y[:n_ch] = np.clip(cp[:n_ch], 0, 1)
        ytype = np.zeros((n_ch, len(CATS)), dtype=np.float32)
        for sp in it.labels:
            k = CATS.index(sp.get("label", "other")) if sp.get("label") in CATS else 4
            a, b = max(0, sp["start"]), min(n_ch, sp["end"])
            ytype[a:b, k] = np.maximum(ytype[a:b, k], float(sp.get("prob", 1.0)))
        # char -> token map + in-token position
        t2c = np.full(n_ch, -1, dtype=np.int64)
        inpos = np.zeros(n_ch, dtype=np.int64)
        for t, (s, e) in enumerate(tc):
            s, e = int(s), min(int(e), n_ch)
            if s >= n_ch:
                break
            t2c[s:e] = t
            inpos[s:e] = np.arange(e - s)
        return dict(V=V[: self.max_vis], H=H, t2c=t2c, inpos=inpos, y=y, ytype=ytype,
                    gate=float(bool(it.labels)), item=it)


def collate(batch):
    B = len(batch)
    T = max(b["H"].shape[0] for b in batch)
    P = max(b["V"].shape[0] for b in batch)
    C = max(len(b["y"]) for b in batch)
    L, D = batch[0]["H"].shape[1], batch[0]["H"].shape[2]
    H = torch.zeros(B, T, L, D); V = torch.zeros(B, P, L, D)
    vmask = torch.ones(B, P, dtype=torch.bool)
    t2c = torch.full((B, C), -1, dtype=torch.long)
    inpos = torch.zeros(B, C, dtype=torch.long)
    y = torch.zeros(B, C); ytype = torch.zeros(B, C, len(CATS))
    valid = torch.zeros(B, C); gate = torch.zeros(B)
    items = []
    for i, b in enumerate(batch):
        t, p, c = b["H"].shape[0], b["V"].shape[0], len(b["y"])
        H[i, :t] = torch.from_numpy(b["H"].astype(np.float32))
        V[i, :p] = torch.from_numpy(b["V"].astype(np.float32))
        vmask[i, :p] = False
        t2c[i, :c] = torch.from_numpy(b["t2c"]); inpos[i, :c] = torch.from_numpy(b["inpos"])
        y[i, :c] = torch.from_numpy(b["y"]); ytype[i, :c] = torch.from_numpy(b["ytype"])
        valid[i, :c] = torch.from_numpy((b["t2c"] >= 0).astype(np.float32))
        gate[i] = b["gate"]; items.append(b["item"])
    return H, V, vmask, t2c, inpos, y, ytype, valid, gate, items


# --------------------------------------------------------------------------- post/eval
def merge_spans(q, ptype, tau):
    spans, i, n = [], 0, len(q)
    while i < n:
        if q[i] >= tau:
            j = i
            while j < n and q[j] >= tau:
                j += 1
            seg = ptype[i:j].mean(0)
            spans.append({"start": i, "end": j, "prob": float(q[i:j].mean()),
                          "label": CATS[int(seg.argmax())]})
            i = j
        else:
            i += 1
    # merge adjacent same-type
    out = []
    for s in spans:
        if out and out[-1]["label"] == s["label"] and out[-1]["end"] == s["start"]:
            out[-1]["end"] = s["end"]
            out[-1]["prob"] = max(out[-1]["prob"], s["prob"])
        else:
            out.append(s)
    return out


@torch.no_grad()
def run_eval(model, dl, device, no_image=False, shuffle=False, taus=(0.1,0.2,0.3,0.4,0.5,0.6)):
    model.eval()
    per = {}
    prev_V = None
    for H, V, vmask, t2c, inpos, y, ytype, valid, gate, items in dl:
        if no_image:
            V = torch.zeros_like(V)
        if shuffle:                                   # derange within batch
            V = torch.roll(V, 1, dims=0); vmask = torch.roll(vmask, 1, dims=0)
        ql, pl, tl, gl = model(H.to(device), V.to(device), t2c.to(device),
                               inpos.to(device), vmask.to(device))
        q = torch.sigmoid(ql).cpu().numpy(); p = torch.sigmoid(pl).cpu().numpy()
        t = torch.sigmoid(tl).cpu().numpy()
        for b, it in enumerate(items):
            n = int(valid[b].sum())
            per[it.id] = (it, q[b, :n], p[b, :n], t[b, :n])
    ids = list(per)
    ga, pa = [], []
    for i in ids:
        it, q, p, t = per[i]
        ga += gold_char_probs(it.labels, len(q)); pa += p.tolist()
    gb = [1 if g > 0 else 0 for g in ga]
    roc = roc_auc_score(gb, pa) if len(set(gb)) > 1 else float("nan")
    pr = average_precision_score(gb, pa)
    cal = pearson(ga, pa)
    # label-aware calibration (Cor lbl approx): pearson over per-char per-type gold vs pred
    gt, pt = [], []
    for i in ids:
        it, q, p, t = per[i]
        yt = np.zeros((len(q), len(CATS)))
        for sp in it.labels:
            k = CATS.index(sp["label"]) if sp.get("label") in CATS else 4
            a, b2 = max(0, sp["start"]), min(len(q), sp["end"])
            yt[a:b2, k] = np.maximum(yt[a:b2, k], float(sp.get("prob", 1.0)))
        gt += yt.flatten().tolist(); pt += t.flatten().tolist()
    cal_lbl = pearson(gt, pt)
    floor = float(np.mean([1.0 if not per[i][0].labels else 0.0 for i in ids]))
    best = (0.3, -1)
    for tau in taus:
        ious = [char_iou(per[i][0].labels, merge_spans(per[i][1], per[i][3], tau), len(per[i][1]))
                for i in ids]
        m = float(np.mean(ious))
        if m > best[1]:
            best = (tau, m)
    return per, dict(span_iou=best[1], tau=best[0], floor=floor, roc_auc=roc, pr_auc=pr,
                     calib=cal, calib_lbl=cal_lbl)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arch", choices=["connector", "linear"], default="connector")
    ap.add_argument("--no_image", action="store_true")
    ap.add_argument("--eval_shuffle", action="store_true")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambdas", default="1,1,0.5,0.5,0.2")
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    l1, l2, l3, l4, l5 = [float(x) for x in args.lambdas.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = load_jsonl(args.train_file)
    tr_ids, dv_ids = group_split_by_image(items, dev_frac=args.dev_frac, seed=args.seed)
    tr = [it for it in items if it.id in set(tr_ids)]
    dv = [it for it in items if it.id in set(dv_ids)]
    tr_ds, dv_ds = CacheDS(tr, args.cache_dir), CacheDS(dv, args.cache_dir)
    print(f"[data] cached train={len(tr_ds)} dev={len(dv_ds)} (of {len(tr)}/{len(dv)})", flush=True)
    assert len(tr_ds) > 50, "too few cached items — run extract_features.py first"
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dv_dl = DataLoader(dv_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    z0 = tr_ds[0]
    D, L = z0["H"].shape[2], z0["H"].shape[1]
    model = Connector(D, L, dim=args.dim, blocks=args.blocks, arch=args.arch).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] arch={args.arch} no_image={args.no_image} trainable={n_par/1e6:.1f}M "
          f"(backbone frozen, cached)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    tag = f"{args.arch}{'_noimg' if args.no_image else ''}"
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = nb = 0
        for H, V, vmask, t2c, inpos, y, ytype, valid, gate, _ in tr_dl:
            if args.no_image:
                V = torch.zeros_like(V)
            H, V, vmask = H.to(device), V.to(device), vmask.to(device)
            t2c, inpos = t2c.to(device), inpos.to(device)
            y, ytype, valid, gate = (y.to(device), ytype.to(device),
                                     valid.to(device), gate.to(device))
            ql, pl, tl, gl = model(H, V, t2c, inpos, vmask)
            def exmean(x):                                     # per-example, then batch
                return ((x * valid).sum(1) / valid.sum(1).clamp(min=1)).mean()
            m = (y > 0).float()
            loss = (l1 * exmean(bce(pl, y))
                    + l2 * soft_jaccard_loss(ql, m, valid)
                    + l3 * ranking_loss(pl, y, valid)
                    + l4 * ((bce(tl, ytype).mean(-1) * valid).sum(1)
                            / valid.sum(1).clamp(min=1)).mean()
                    + l5 * bce(gl, gate).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        _, m = run_eval(model, dv_dl, device, no_image=args.no_image)
        print(f"[ep {ep}] loss={tot/nb:.4f} dev: span_iou={m['span_iou']:.4f} "
              f"(floor={m['floor']:.4f}, tau={m['tau']}) roc={m['roc_auc']:.3f} "
              f"calib={m['calib']:.3f} calib_lbl={m['calib_lbl']:.3f} "
              f"[{(time.time()-t0)/60:.1f}m]", flush=True)

    per, m = run_eval(model, dv_dl, device, no_image=args.no_image)
    results = {"variant": tag, "metrics": m}
    if args.eval_shuffle and not args.no_image:
        _, ms = run_eval(model, dv_dl, device, shuffle=True)
        results["metrics_shuffled_image"] = ms
        print(f"[control] shuffled-image: span_iou={ms['span_iou']:.4f} roc={ms['roc_auc']:.3f}")
    # official JSONL
    pred_path = os.path.join(args.out_dir, f"dev_pred_{tag}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, (it, q, p, t) in per.items():
            f.write(json.dumps({"id": i, "language": it.language, "response": it.response,
                                "pred_labels": merge_spans(q, t, m["tau"]),
                                "char_probs": [round(float(x), 3) for x in p]},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"connector_{tag}.pt"))
    print(json.dumps(results, indent=2))
    print(f"predictions -> {pred_path}")


if __name__ == "__main__":
    main()
