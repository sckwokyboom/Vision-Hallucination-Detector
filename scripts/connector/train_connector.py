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
from model import Connector, soft_jaccard_loss, tversky_loss, ranking_loss, CATS  # noqa: E402


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
        try:
            z = np.load(os.path.join(self.dir, f"{it.id}.npz"))
        except Exception:              # partial/corrupt cache file -> fall back to a neighbour
            return self.__getitem__((i + 1) % len(self.items))
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
        m = (y > 0).astype(np.int64)
        bio = np.zeros(n_ch, dtype=np.int64)                 # 0=O
        for k in range(n_ch):
            if m[k]:
                bio[k] = 1 if (k == 0 or not m[k - 1]) else 2  # B / I
        return dict(V=V[: self.max_vis], H=H, t2c=t2c, inpos=inpos, y=y, ytype=ytype,
                    bio=bio, gate=float(bool(it.labels)), item=it)


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
    bio = torch.zeros(B, C, dtype=torch.long)
    valid = torch.zeros(B, C); gate = torch.zeros(B)
    items = []
    for i, b in enumerate(batch):
        t, p, c = b["H"].shape[0], b["V"].shape[0], len(b["y"])
        H[i, :t] = torch.from_numpy(b["H"].astype(np.float32))
        V[i, :p] = torch.from_numpy(b["V"].astype(np.float32))
        vmask[i, :p] = False
        t2c[i, :c] = torch.from_numpy(b["t2c"]); inpos[i, :c] = torch.from_numpy(b["inpos"])
        y[i, :c] = torch.from_numpy(b["y"]); ytype[i, :c] = torch.from_numpy(b["ytype"])
        bio[i, :c] = torch.from_numpy(b["bio"])
        valid[i, :c] = torch.from_numpy((b["t2c"] >= 0).astype(np.float32))
        gate[i] = b["gate"]; items.append(b["item"])
    return H, V, vmask, t2c, inpos, y, ytype, bio, valid, gate, items


class GroupedBatchSampler(torch.utils.data.Sampler):
    """Batches that keep same-(image,prompt) answer variants together (contrastive)."""

    def __init__(self, ds, batch_size, seed=0):
        self.groups = {}
        for idx, it in enumerate(ds.items):
            self.groups.setdefault((it.image_name, it.prompt), []).append(idx)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        keys = list(self.groups)
        rng.shuffle(keys)
        batch = []
        for k in keys:
            batch.extend(self.groups[k])
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def __len__(self):
        n = sum(len(v) for v in self.groups.values())
        return (n + self.batch_size - 1) // self.batch_size


# --------------------------------------------------------------------------- post/eval
def peak_spans(p, ptype, rel_hi=0.6, rel_lo=0.35, abs_min=0.08, min_len=2, max_len=150):
    """Per-item RELATIVE thresholds: enter at p >= max(rel_hi*peak, abs_min), expand while
    p >= rel_lo*peak. Attacks global-threshold miscalibration and over-long spans."""
    import numpy as _np
    p = _np.asarray(p)
    if p.size == 0 or p.max() <= abs_min:
        return []
    peak = float(p.max())
    hi = max(rel_hi * peak, abs_min)
    lo = max(rel_lo * peak, abs_min * 0.5)
    spans, i, n = [], 0, len(p)
    while i < n:
        if p[i] >= hi:
            a = i
            while a > 0 and p[a - 1] >= lo and (i - a) < max_len:
                a -= 1
            b = i
            while b < n and p[b] >= lo and (b - a) < max_len:
                b += 1
            if b - a >= min_len:
                seg = ptype[a:b].mean(0)
                spans.append({"start": a, "end": b, "prob": float(p[a:b].mean()),
                              "label": CATS[int(seg.argmax())]})
            i = b if b > i else i + 1          # guarantee progress (fix: infinite loop
        else:                                   # when a long high region hits max_len)
            i += 1
    return spans


def bio_spans(tags, p, ptype, min_len=1):
    """Spans from BIO tags (1=B, 2=I; orphan I after O also starts a span)."""
    spans, i, n = [], 0, len(tags)
    while i < n:
        if tags[i] > 0:
            a = i
            b = i + 1
            while b < n and tags[b] == 2:
                b += 1
            if b - a >= min_len:
                seg = ptype[a:b].mean(0)
                spans.append({"start": a, "end": b, "prob": float(p[a:b].mean()),
                              "label": CATS[int(seg.argmax())]})
            i = b
        else:
            i += 1
    return spans


def hysteresis_spans(p, ptype, tau_hi, tau_lo, min_len=2):
    """Enter a span at p>=tau_hi, extend while p>=tau_lo, drop spans shorter than min_len."""
    spans, i, n = [], 0, len(p)
    while i < n:
        if p[i] >= tau_hi:
            a = i
            while a > 0 and p[a - 1] >= tau_lo:
                a -= 1
            b = i
            while b < n and p[b] >= tau_lo:
                b += 1
            if b - a >= min_len:
                seg = ptype[a:b].mean(0)
                spans.append({"start": a, "end": b, "prob": float(p[a:b].mean()),
                              "label": CATS[int(seg.argmax())]})
            i = b
        else:
            i += 1
    out = []
    for sp in spans:
        if out and out[-1]["label"] == sp["label"] and sp["start"] - out[-1]["end"] <= 1:
            out[-1]["end"] = sp["end"]; out[-1]["prob"] = max(out[-1]["prob"], sp["prob"])
        else:
            out.append(sp)
    return out


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
def run_eval(model, dl, device, no_image=False, shuffle=False, decoder="gate_hyst", taus=None):
    if taus is None:
        taus = ((0.0,) if decoder == "bio" else
                (0.4, 0.5, 0.6, 0.7, 0.8) if decoder == "v2" else
                (0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95))
    model.eval()
    per = {}
    prev_V = None
    for H, V, vmask, t2c, inpos, y, ytype, bio_t, valid, gate, items in dl:
        if no_image:
            V = torch.zeros_like(V)
        if shuffle:                                   # derange within batch
            V = torch.roll(V, 1, dims=0); vmask = torch.roll(vmask, 1, dims=0)
        ql, pl, tl, gl, bl = model(H.to(device), V.to(device), t2c.to(device),
                                   inpos.to(device), vmask.to(device))
        q = torch.sigmoid(ql).cpu().numpy(); p = torch.sigmoid(pl).cpu().numpy()
        t = torch.sigmoid(tl).cpu().numpy(); g = torch.sigmoid(gl).cpu().numpy()
        bt = bl.argmax(-1).cpu().numpy()
        for b, it in enumerate(items):
            n = int(valid[b].sum())
            per[it.id] = (it, q[b, :n], p[b, :n], t[b, :n], float(g[b]), bt[b, :n])
    ids = list(per)
    ga, pa = [], []
    for i in ids:
        it, q, p, t, g, bt = per[i]
        ga += gold_char_probs(it.labels, len(q)); pa += p.tolist()
    gb = [1 if g > 0 else 0 for g in ga]
    roc = roc_auc_score(gb, pa) if len(set(gb)) > 1 else float("nan")
    pr = average_precision_score(gb, pa)
    cal = pearson(ga, pa)
    # label-aware calibration (Cor lbl approx): pearson over per-char per-type gold vs pred
    gt, pt = [], []
    for i in ids:
        it, q, p, t, g, bt = per[i]
        yt = np.zeros((len(q), len(CATS)))
        for sp in it.labels:
            k = CATS.index(sp["label"]) if sp.get("label") in CATS else 4
            a, b2 = max(0, sp["start"]), min(len(q), sp["end"])
            yt[a:b2, k] = np.maximum(yt[a:b2, k], float(sp.get("prob", 1.0)))
        gt += yt.flatten().tolist(); pt += t.flatten().tolist()
    cal_lbl = pearson(gt, pt)
    floor = float(np.mean([1.0 if not per[i][0].labels else 0.0 for i in ids]))
    def decode(p, t, g, tau, g_thr, bt=None):
        if decoder == "bio":
            spans = bio_spans(bt, p, t)
            return [] if (g_thr > 0 and g < g_thr) else spans
        if decoder == "v2":
            # soft token-derived gate: p' = p * g_eff^alpha, alpha encoded via g_thr slot
            import numpy as _np
            topk = _np.sort(_np.asarray(p))[-max(3, len(p)//20):].mean() if len(p) else 0.0
            g_eff = 0.5 * g + 0.5 * topk
            p2 = _np.asarray(p) * (g_eff ** g_thr)
            return peak_spans(p2, t, rel_hi=tau, rel_lo=0.55 * tau)
        if decoder in ("gate", "gate_hyst") and g < g_thr:
            return []
        if decoder in ("hyst", "gate_hyst"):
            return hysteresis_spans(p, t, tau, 0.6 * tau)
        return merge_spans(p, t, tau)                     # simple threshold
    if decoder == "bio":
        g_grid = (0.0, 0.3, 0.5, 0.7)
    elif decoder == "v2":
        g_grid = (0.0, 0.5, 1.0, 2.0)                     # soft-gate exponent alpha
    elif decoder in ("simple", "hyst"):
        g_grid = (0.0,)
    else:
        g_grid = (0.0, 0.3, 0.5, 0.7)
    best = {"iou": -1}
    for g_thr in g_grid:
        for tau in taus:
            ious = []
            for i in ids:
                it, q, p, t, g, bt = per[i]
                ious.append(char_iou(it.labels, decode(p, t, g, tau, g_thr, bt), len(p)))
            m = float(np.mean(ious))
            if m > best["iou"]:
                best = {"iou": m, "tau": tau, "g_thr": g_thr}
    # diagnostics at the best operating point
    d_ious, open_ious, clean_ok, n_clean, n_dirty, dirty_open, nspans = [], [], 0, 0, 0, 0, []
    sub_g, sub_p = [], []
    for i in ids:
        it, q, p, t, g, bt = per[i]
        spans = decode(p, t, g, best["tau"], best["g_thr"], bt)
        nspans.append(len(spans))
        gated_out = (decoder in ("gate", "gate_hyst") and g < best["g_thr"])
        sub = np.zeros(len(p)) if gated_out else p          # submission-time probs
        sub_g += gold_char_probs(it.labels, len(p)); sub_p += sub.tolist()
        if it.labels:
            n_dirty += 1
            iou_i = char_iou(it.labels, spans, len(p))
            d_ious.append(iou_i)
            if not gated_out:
                dirty_open += 1; open_ious.append(iou_i)
        else:
            n_clean += 1
            clean_ok += (0 if spans else 1)
    from shroom.metrics import spearman as _sp
    return per, dict(span_iou=best["iou"], tau=best["tau"], g_thr=best["g_thr"], floor=floor,
                     dirty_iou=float(np.mean(d_ious)) if d_ious else 0.0,
                     clean_empty=(clean_ok / n_clean) if n_clean else 1.0,
                     dirty_gate_recall=(dirty_open / n_dirty) if n_dirty else 0.0,
                     iou_gate_open=float(np.mean(open_ious)) if open_ious else 0.0,
                     avg_spans=float(np.mean(nspans)),
                     roc_auc=roc, pr_auc=pr, cor_raw=cal, cor_spearman=float(_sp(ga, pa)),
                     cor_submission=float(pearson(sub_g, sub_p)), cor_lbl=cal_lbl)


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
    ap.add_argument("--max_train", type=int, default=None, help="cap train items (learning curve)")
    ap.add_argument("--eval_ids", default=None, help="json file with {'tune_dev': [...]} to eval on")
    ap.add_argument("--decoder", choices=["simple","gate","hyst","gate_hyst","v2","bio"], default="gate_hyst")
    ap.add_argument("--tversky", action="store_true", help="precision-weighted Tversky instead of soft-Jaccard")
    ap.add_argument("--no_type_loss", action="store_true")
    ap.add_argument("--bio", action="store_true", help="BIO boundary head + loss")
    ap.add_argument("--gate_consistency", action="store_true")
    ap.add_argument("--contrastive", action="store_true", help="group same image+question in batches + margin loss")
    ap.add_argument("--l_bio", type=float, default=1.0)
    ap.add_argument("--l_consist", type=float, default=0.3)
    ap.add_argument("--l_contrast", type=float, default=0.3)
    ap.add_argument("--no_gru", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--init_from", default=None, help="load model weights before train/eval")
    args = ap.parse_args()

    l1, l2, l3, l4, l5 = [float(x) for x in args.lambdas.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = load_jsonl(args.train_file)
    tr_ids, dv_ids = group_split_by_image(items, dev_frac=args.dev_frac, seed=13)  # split seed FIXED
    if args.eval_ids:
        prot = json.load(open(args.eval_ids))
        dv_keep = set(prot.get("tune_dev") or prot.get("tune_dev200"))
        dv_ids = [i for i in dv_ids if i in dv_keep]
    tr = [it for it in items if it.id in set(tr_ids)]
    dv = [it for it in items if it.id in set(dv_ids)]
    if args.max_train:
        tr = tr[:args.max_train]
    tr_ds, dv_ds = CacheDS(tr, args.cache_dir), CacheDS(dv, args.cache_dir)
    manifest = {"train_ids": [it.id for it in tr_ds.items], "dev_ids": [it.id for it in dv_ds.items],
                "seed": args.seed, "decoder": args.decoder, "gru": not args.no_gru}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"manifest_seed{args.seed}.json"), "w") as f:
        json.dump(manifest, f)
    print(f"[data] cached train={len(tr_ds)} dev={len(dv_ds)} (of {len(tr)}/{len(dv)}) — manifest frozen", flush=True)
    assert len(tr_ds) > 50, "too few cached items — run extract_features.py first"
    if args.contrastive:
        tr_dl = DataLoader(tr_ds, batch_sampler=GroupedBatchSampler(tr_ds, args.batch, args.seed),
                           collate_fn=collate)
    else:
        tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dv_dl = DataLoader(dv_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    z0 = tr_ds[0]
    D, L = z0["H"].shape[2], z0["H"].shape[1]
    model = Connector(D, L, dim=args.dim, blocks=args.blocks, arch=args.arch,
                      use_gru=not args.no_gru).to(device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=device), strict=False)
        print(f"[model] loaded weights from {args.init_from}", flush=True)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] arch={args.arch} no_image={args.no_image} trainable={n_par/1e6:.1f}M "
          f"(backbone frozen, cached)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    tag = f"{args.arch}{'_noimg' if args.no_image else ''}_{args.decoder}{'_tv' if args.tversky else ''}{'_bio' if args.bio else ''}{'_gc' if args.gate_consistency else ''}{'_ctr' if args.contrastive else ''}{'_notype' if args.no_type_loss else ''}{'_nogru' if args.no_gru else ''}_s{args.seed}"
    t0 = time.time()
    for ep in range(1, 0 if args.eval_only else args.epochs + 1):
        model.train(); tot = nb = 0
        for H, V, vmask, t2c, inpos, y, ytype, bio_t, valid, gate, bitems in tr_dl:
            if args.no_image:
                V = torch.zeros_like(V)
            H, V, vmask = H.to(device), V.to(device), vmask.to(device)
            t2c, inpos = t2c.to(device), inpos.to(device)
            y, ytype, valid, gate = (y.to(device), ytype.to(device),
                                     valid.to(device), gate.to(device))
            bio_t = bio_t.to(device)
            ql, pl, tl, gl, bl = model(H, V, t2c, inpos, vmask)
            def exmean(x):                                     # per-example, then batch
                return ((x * valid).sum(1) / valid.sum(1).clamp(min=1)).mean()
            m = (y > 0).float()
            span_l = tversky_loss(ql, m, valid) if args.tversky else soft_jaccard_loss(ql, m, valid)
            l4_eff = 0.0 if args.no_type_loss else l4
            loss = (l1 * exmean(bce(pl, y))
                    + l2 * span_l
                    + l3 * ranking_loss(pl, y, valid)
                    + l4_eff * ((bce(tl, ytype).mean(-1) * valid).sum(1)
                            / valid.sum(1).clamp(min=1)).mean()
                    + l5 * bce(gl, gate).mean())
            if args.bio:
                w = torch.tensor([1.0, 4.0, 2.0], device=device)   # O / B / I
                ce = nn.functional.cross_entropy(
                    bl.reshape(-1, 3), bio_t.reshape(-1), weight=w, reduction="none"
                ).reshape(bio_t.shape)
                loss = loss + args.l_bio * exmean(ce)
            k_top = max(3, y.shape[1] // 20)
            p_sig = torch.sigmoid(pl) * valid
            topk = p_sig.topk(min(k_top, p_sig.shape[1]), dim=1).values.mean(1)
            if args.gate_consistency:
                loss = loss + args.l_consist * nn.functional.mse_loss(torch.sigmoid(gl), topk)
            if args.contrastive:
                score = 0.5 * torch.sigmoid(gl) + 0.5 * topk
                keys = [(it.image_name, it.prompt) for it in bitems]
                closs, npair = score.new_zeros(()), 0
                for a in range(len(keys)):
                    for b2 in range(len(keys)):
                        if a != b2 and keys[a] == keys[b2] and gate[a] > 0.5 > gate[b2]:
                            closs = closs + torch.relu(0.2 - (score[a] - score[b2]))
                            npair += 1
                if npair:
                    loss = loss + args.l_contrast * closs / npair
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        _, m = run_eval(model, dv_dl, device, no_image=args.no_image, decoder=args.decoder)
        print(f"[ep {ep}] loss={tot/nb:.4f} dev: iou={m['span_iou']:.4f} (fl={m['floor']:.3f}, "
              f"tau={m['tau']}, g={m['g_thr']}) dirty={m['dirty_iou']:.3f} cleanOK={m['clean_empty']:.2f} "
              f"gateRec={m['dirty_gate_recall']:.2f} corR={m['cor_raw']:.3f} corS={m['cor_submission']:.3f} "
              f"[{(time.time()-t0)/60:.1f}m]", flush=True)

    per, m = run_eval(model, dv_dl, device, no_image=args.no_image, decoder=args.decoder)
    results = {"variant": tag, "metrics": m}
    if args.eval_shuffle and not args.no_image:
        _, ms = run_eval(model, dv_dl, device, shuffle=True, decoder=args.decoder)
        results["metrics_shuffled_image"] = ms
        print(f"[control] shuffled-image: span_iou={ms['span_iou']:.4f} roc={ms['roc_auc']:.3f}")
    # official JSONL
    pred_path = os.path.join(args.out_dir, f"dev_pred_{tag}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, (it, q, p, t, g, bt) in per.items():
            if args.decoder == "bio":
                spans = bio_spans(bt, p, t) if (m["g_thr"] == 0 or g >= m["g_thr"]) else []
            elif args.decoder == "v2":
                spans = peak_spans(p, t, rel_hi=m["tau"], rel_lo=0.55 * m["tau"])
            else:
                spans = [] if g < m["g_thr"] else hysteresis_spans(p, t, m["tau"], 0.6 * m["tau"])
            f.write(json.dumps({"id": i, "language": it.language, "response": it.response,
                                "pred_labels": spans,
                                "char_probs": [round(float(x), 3) for x in p]},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"connector_{tag}.pt"))
    print(json.dumps(results, indent=2))
    print(f"predictions -> {pred_path}")


if __name__ == "__main__":
    main()
