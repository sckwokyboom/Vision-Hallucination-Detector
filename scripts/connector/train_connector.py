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
from shroom.metrics import gold_char_probs, gold_char_probs_sum, pearson  # noqa: E402
try:
    from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
except Exception:  # pragma: no cover - exercised only in lean inference envs
    def _rankdata_np(a):
        a = np.asarray(a, dtype=float)
        sorter = np.argsort(a, kind="mergesort")
        inv = np.empty(len(a), dtype=int)
        inv[sorter] = np.arange(len(a))
        a_sorted = a[sorter]
        obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
        dense = obs.cumsum()[inv]
        count = np.r_[np.flatnonzero(obs), len(a)]
        return 0.5 * (count[dense] + count[dense - 1] + 1)

    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true) > 0
        if y_true.min() == y_true.max():
            raise ValueError("Only one class present in y_true")
        ranks = _rankdata_np(y_score)
        n_pos = int(y_true.sum())
        n_neg = int((~y_true).sum())
        return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def average_precision_score(y_true, y_score):
        y_true = np.asarray(y_true) > 0
        if not y_true.any():
            return 0.0
        order = np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")
        y = y_true[order]
        tp = y.cumsum()
        precision = tp / (np.arange(len(y)) + 1)
        return float(precision[y].mean())

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "official"))
try:
    from official_scorer import (score_cor as official_score_cor,  # noqa: E402
                                 score_cor_lbl as official_score_cor_lbl,
                                 score_iou as official_score_iou)
except Exception:  # pragma: no cover - fallback for inference-only environments
    from shroom.metrics import spearman as _fallback_spearman  # noqa: E402

    def official_score_cor(ref_dict, pred_dict, label_filtered_=None):
        assert ref_dict["id"] == pred_dict["id"]
        ref_vec = [0.0] * ref_dict["text_len"]
        pred_vec = [0.0] * ref_dict["text_len"]
        ref_labels = (ref_dict["labels"] if label_filtered_ is None
                      else [s for s in ref_dict["labels"] if s["label"] == label_filtered_])
        pred_labels = (pred_dict["labels"] if label_filtered_ is None
                       else [s for s in pred_dict["labels"] if s["label"] == label_filtered_])
        for span in ref_labels:
            for idx in range(span["start"], span["end"]):
                ref_vec[idx] += span["prob"]
        for span in pred_labels:
            for idx in range(span["start"], span["end"]):
                pred_vec[idx] = span["prob"]
        ref_cmps = {round(flt, 8) for flt in ref_vec}
        pred_cmps = {round(flt, 8) for flt in pred_vec}
        if len(pred_cmps) == 1 or len(ref_cmps) == 1:
            if len(pred_cmps) != len(ref_cmps):
                return 0.0
            if ref_cmps == {0.0}:
                return float(pred_cmps == {0.0})
            return float(pred_cmps != {0.0})
        return _fallback_spearman(ref_vec, pred_vec)

    def official_score_cor_lbl(ref_dict, pred_dict):
        labels = {span["label"] for rec in (ref_dict, pred_dict) for span in rec["labels"]}
        return (sum(official_score_cor(ref_dict, pred_dict, label) for label in labels)
                / len(labels)) if labels else 1.0

    def official_score_iou(ref_dict, pred_dict):
        assert ref_dict["id"] == pred_dict["id"]
        ref_indices = {idx for span in ref_dict["labels"] for idx in range(span["start"], span["end"])}
        pred_indices = {idx for span in pred_dict["labels"] for idx in range(span["start"], span["end"])}
        if not pred_indices and not ref_indices:
            return 1.0
        return len(ref_indices & pred_indices) / len(ref_indices | pred_indices)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (Connector, SetDecoder, set_decode, SegmentScorer, dp_select, SemiCRF,  # noqa: E402
                   soft_jaccard_loss, tversky_loss, ranking_loss, CATS)


# --------------------------------------------------------------------------- data
def build_example(it, V, H, tc, answer_len, max_chars=4000, max_vis=1200):
    """Supervision for one item from its backbone features and token->char map.

    This is THE definition of the training target — official SUM aggregation of
    annotator spans, char->token scatter map, BIO tags, contiguous gold segments —
    shared by the cached-feature trainer (CacheDS) and the live-backbone LoRA
    trainer, so the two can never drift apart on gold semantics.
    """
    n_ch = min(int(answer_len), max_chars)
    # char-level gold
    cp = gold_char_probs_sum(it.labels, int(answer_len))   # official aggregation
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
    segs = []
    k = 0
    while k < n_ch:
        if m[k]:
            a = k
            while k < n_ch and m[k]:
                k += 1
            tmean = ytype[a:k].mean(0)
            segs.append((a, k, int(tmean.argmax()), float(y[a:k].max())))
        else:
            k += 1
    bio = np.zeros(n_ch, dtype=np.int64)                 # 0=O
    for k in range(n_ch):
        if m[k]:
            bio[k] = 1 if (k == 0 or not m[k - 1]) else 2  # B / I
    return dict(V=V[:max_vis], H=H, t2c=t2c, inpos=inpos, y=y, ytype=ytype,
                bio=bio, segs=segs, gate=float(bool(it.labels)), item=it)


class CacheDS(Dataset):
    def __init__(self, items, cache_dir, max_chars=1200, max_vis=1200, shuffle_v=False):
        self.items = [it for it in items
                      if os.path.exists(os.path.join(cache_dir, f"{it.id}.npz"))]
        self.dir = cache_dir
        self.max_chars, self.max_vis = max_chars, max_vis
        # shuffle_v: train-time grounding control. Every item gets the V of a DIFFERENT
        # image (deterministic derangement), so a connector trained this way has the same
        # capacity as the real one but no usable visual information. Distinct from
        # --eval_shuffle, which deranges at inference on a normally-trained model.
        self.vmap = None
        if shuffle_v:
            n = len(self.items)
            rng = np.random.default_rng(0)
            perm = rng.permutation(n)
            for i in range(n):     # no fixed points on the same image
                if self.items[perm[i]].image_name == self.items[i].image_name:
                    j = (i + 1) % n
                    perm[i], perm[j] = perm[j], perm[i]
            self.vmap = perm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        try:
            z = np.load(os.path.join(self.dir, f"{it.id}.npz"))
        except Exception:              # partial/corrupt cache file -> fall back to a neighbour
            return self.__getitem__((i + 1) % len(self.items))
        V, H, tc = z["V"], z["H"], z["tok_char"]
        if self.vmap is not None:
            other = self.items[self.vmap[i]]
            try:
                V = np.load(os.path.join(self.dir, f"{other.id}.npz"))["V"]
            except Exception:
                pass                                   # keep own V rather than crash
        return build_example(it, V, H, tc, int(z["answer_len"]),
                             self.max_chars, self.max_vis)


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
    segs = [b["segs"] for b in batch]
    return H, V, vmask, t2c, inpos, y, ytype, bio, segs, valid, gate, items


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


def sanitize_spans(spans, resp_len=None):
    """Official-format span records with exact Python scalar types."""
    out = []
    for sp in spans or []:
        a, b = int(sp["start"]), int(sp["end"])
        if resp_len is not None:
            a = max(0, min(a, resp_len))
            b = max(0, min(b, resp_len))
        if not (a < b):
            continue
        prob = float(sp.get("prob", 1.0))
        if not np.isfinite(prob):
            prob = 0.0
        label = str(sp.get("label", "other"))
        if label not in CATS:
            label = "other"
        out.append({"start": a, "end": b, "prob": prob, "label": label})
    return out


def topk_confidence(p):
    arr = np.asarray(p, dtype=float)
    if arr.size == 0:
        return 0.0
    k = max(3, int(arr.size) // 20)
    k = min(k, arr.size)
    return float(np.sort(arr)[-k:].mean())


def decoder_null_decision(decoder, p, g, tau, g_thr, bt=None, qsp=None):
    """Whether inference intentionally returns the null submission for this item."""
    if decoder == "crf":
        return (not qsp) or qsp.get("p_empty", 0.0) > tau
    if decoder == "bio":
        if g_thr > 0 and g < g_thr:
            return True
        if tau > 0 and topk_confidence(p) < tau:
            return True
        return False
    if decoder in ("gate", "gate_hyst"):
        return g < g_thr
    return False


def decode_spans(decoder, p, t, g, tau, g_thr, bt=None, qsp=None, resp_len=None):
    """Decode exactly the spans that the system would serialize for submission."""
    if decoder_null_decision(decoder, p, g, tau, g_thr, bt, qsp):
        return []
    if decoder == "crf":
        return sanitize_spans(qsp["by_bias"].get(g_thr, []), resp_len)
    if decoder == "seg":
        if not qsp:
            return []
        cands = [(a, b, ti) for _, a, b, ti in qsp]
        scores = [c[0] for c in qsp]
        sel = dp_select(cands, scores, tau)
        spans = ({"start": cands[i][0], "end": cands[i][1],
                  "prob": scores[i], "label": CATS[cands[i][2]]}
                 for i in sel)
        return sanitize_spans(sorted(spans, key=lambda sp: sp["start"]), resp_len)
    if decoder == "set":
        spans = []
        for conf, a, b, ti in sorted(qsp or [], key=lambda c: -c[0]):
            if conf < tau:
                break
            if any(not (b <= sp["start"] or a >= sp["end"]) for sp in spans):
                continue
            spans.append({"start": a, "end": b, "prob": conf, "label": CATS[ti]})
        return sanitize_spans(sorted(spans, key=lambda sp: sp["start"]), resp_len)
    if decoder == "bio":
        return sanitize_spans(bio_spans(bt, p, t), resp_len)
    if decoder == "v2":
        p2 = np.asarray(p) * ((0.5 * g + 0.5 * topk_confidence(p)) ** g_thr)
        return sanitize_spans(peak_spans(p2, t, rel_hi=tau, rel_lo=0.55 * tau), resp_len)
    if decoder in ("hyst", "gate_hyst"):
        return sanitize_spans(hysteresis_spans(p, t, tau, 0.6 * tau), resp_len)
    return sanitize_spans(merge_spans(p, t, tau), resp_len)


def official_gold_record(it):
    return {"id": it.id, "labels": it.labels, "text_len": len(it.response)}


def official_pred_record(it, spans):
    return {"id": it.id, "labels": sanitize_spans(spans, len(it.response))}


@torch.no_grad()
def run_eval(model, dl, device, no_image=False, shuffle=False, decoder="gate_hyst", taus=None, setdec=None, segsc=None, crf=None, gate_model=None):
    if taus is None:
        taus = ((0.05, 0.15, 0.3, 0.5, 0.7, 0.85) if decoder == "crf" else
                (0.3, 0.5, 0.7, 0.9, 0.95) if decoder == "set" else
                (0.2, 0.35, 0.5, 0.65, 0.8) if decoder == "seg" else
                (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5) if decoder == "bio" else
                (0.4, 0.5, 0.6, 0.7, 0.8) if decoder == "v2" else
                (0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95))
    model.eval()
    per = {}
    prev_V = None
    for H, V, vmask, t2c, inpos, y, ytype, bio_t, seg_l, valid, gate, items in dl:
        if no_image:
            V = torch.zeros_like(V)
        if shuffle:                                   # derange within batch
            V = torch.roll(V, 1, dims=0); vmask = torch.roll(vmask, 1, dims=0)
        t2c_d = t2c.to(device)
        Hd, Vd, vmd, ipd = H.to(device), V.to(device), vmask.to(device), inpos.to(device)
        ql, pl, tl, gl, bl, feats = model(Hd, Vd, t2c_d, ipd, vmd)
        q = torch.sigmoid(ql).cpu().numpy(); p = torch.sigmoid(pl).cpu().numpy()
        t = torch.sigmoid(tl).cpu().numpy(); g = torch.sigmoid(gl).cpu().numpy()
        if gate_model is not None:
            # V4 cascade: the clean/dirty decision comes from a SEPARATE expert,
            # so a dirty-only locator never has to learn to stay silent.
            gl2 = gate_model(Hd, Vd, t2c_d, ipd, vmd)[3]
            g = torch.sigmoid(gl2).cpu().numpy()
        bt = bl.argmax(-1).cpu().numpy()
        qcand = [None] * len(items)
        if crf is not None:
            for b in range(len(items)):
                n3 = int(valid[b].sum())
                if n3 < 2:
                    qcand[b] = {}
                    continue
                by_bias = {}
                for bias in (-0.5, 0.0, 0.5):
                    segs4 = crf.viterbi(feats[b], n3, bias=bias)
                    by_bias[bias] = [
                        {"start": a4, "end": b5,
                         "prob": float(p[b, a4:b5].mean()) if b5 > a4 else 0.0,
                         "label": CATS[int(t[b, a4:b5].mean(0).argmax())]}
                        for a4, b5 in segs4]
                import math as _math
                qcand[b] = {"by_bias": by_bias,
                            "p_empty": _math.exp(crf.log_p_empty(feats[b], n3))}
        if segsc is not None:
            for b in range(len(items)):
                n3 = int(valid[b].sum())
                if n3 < 4:
                    qcand[b] = []
                    continue
                pb = p[b, :n3]
                cset = set()
                for tau2 in (0.08, 0.15, 0.25, 0.4, 0.6):
                    i3 = 0
                    while i3 < n3:
                        if pb[i3] >= tau2:
                            j3 = i3
                            while j3 < n3 and pb[j3] >= tau2 and j3 - i3 < 150:
                                j3 += 1
                            if j3 - i3 >= 2:
                                cset.add((i3, j3))
                            i3 = j3 + 1
                        else:
                            i3 += 1
                cands = sorted(cset)[:60]
                if cands:
                    with torch.no_grad():
                        sc, tp2 = segsc(feats[b, :n3], cands)
                    scs = torch.sigmoid(sc).cpu().numpy()
                    tps = tp2.argmax(-1).cpu().numpy()
                    qcand[b] = [(float(scs[i4]), a, b4, int(tps[i4]))
                                for i4, (a, b4) in enumerate(cands)]
                else:
                    qcand[b] = []
        if setdec is not None:
            sl_, el_, cf_, tp_, sp_ = setdec(feats, (t2c_d < 0))
            sln = sl_.cpu().numpy(); eln = el_.cpu().numpy()
            cfn = torch.sigmoid(cf_).cpu().numpy(); tpn = tp_.cpu().numpy()
            for b in range(len(items)):
                cands = []
                for k in range(sln.shape[1]):
                    a = int(sln[b, k].argmax())
                    e = eln[b, k].copy(); e[:a] = -1e30; e[a + 150:] = -1e30
                    cands.append((float(cfn[b, k]), a, int(e.argmax()) + 1,
                                  int(tpn[b, k].argmax())))
                qcand[b] = cands
        for b, it in enumerate(items):
            n = int(valid[b].sum())
            per[it.id] = (it, q[b, :n], p[b, :n], t[b, :n], float(g[b]), bt[b, :n], qcand[b])
    ids = list(per)
    ga, pa = [], []
    for i in ids:
        it, q, p, t, g, bt, qsp = per[i]
        ga += gold_char_probs(it.labels, len(q)); pa += p.tolist()
    gb = [1 if g > 0 else 0 for g in ga]
    roc = roc_auc_score(gb, pa) if len(set(gb)) > 1 else float("nan")
    pr = average_precision_score(gb, pa)
    cal = pearson(ga, pa)
    # label-aware calibration (Cor lbl approx): pearson over per-char per-type gold vs pred
    gt, pt = [], []
    for i in ids:
        it, q, p, t, g, bt, qsp = per[i]
        yt = np.zeros((len(q), len(CATS)))
        for sp in it.labels:
            k = CATS.index(sp["label"]) if sp.get("label") in CATS else 4
            a, b2 = max(0, sp["start"]), min(len(q), sp["end"])
            yt[a:b2, k] = np.maximum(yt[a:b2, k], float(sp.get("prob", 1.0)))
        gt += yt.flatten().tolist(); pt += t.flatten().tolist()
    cal_lbl = pearson(gt, pt)
    floor = float(np.mean([1.0 if not per[i][0].labels else 0.0 for i in ids]))
    def decode(p, t, g, tau, g_thr, bt=None, qsp=None, it=None):
        return decode_spans(decoder, p, t, g, tau, g_thr, bt, qsp,
                            resp_len=(len(it.response) if it is not None else None))
    if decoder == "crf":
        g_grid = (-0.5, 0.0, 0.5)                    # viterbi bias
    elif decoder in ("set", "seg"):
        g_grid = (0.0,)
    elif decoder == "bio":
        g_grid = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
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
                it, q, p, t, g, bt, qsp = per[i]
                spans = decode(p, t, g, tau, g_thr, bt, qsp, it)
                ious.append(official_score_iou(official_gold_record(it),
                                               official_pred_record(it, spans)))
            m = float(np.mean(ious))
            if m > best["iou"]:
                best = {"iou": m, "tau": tau, "g_thr": g_thr}
    # diagnostics at the best operating point
    d_ious, d_ungated, open_ious, clean_ok = [], [], [], 0
    n_clean, n_dirty, dirty_open, clean_closed, nspans = 0, 0, 0, 0, []
    official_gold, official_pred = [], []
    for i in ids:
        it, q, p, t, g, bt, qsp = per[i]
        spans = decode(p, t, g, best["tau"], best["g_thr"], bt, qsp, it)
        nspans.append(len(spans))
        nulled = decoder_null_decision(decoder, p, g, best["tau"], best["g_thr"], bt, qsp)
        gold_rec = official_gold_record(it)
        pred_rec = official_pred_record(it, spans)
        official_gold.append(gold_rec)
        official_pred.append(pred_rec)
        if it.labels:
            n_dirty += 1
            iou_i = official_score_iou(gold_rec, pred_rec)
            d_ious.append(iou_i)
            ungated_spans = decode(p, t, 1.0, best["tau"], 0.0, bt, qsp, it)
            d_ungated.append(official_score_iou(gold_rec, official_pred_record(it, ungated_spans)))
            if not nulled:
                dirty_open += 1; open_ious.append(iou_i)
        else:
            n_clean += 1
            if nulled:
                clean_closed += 1
            clean_ok += (0 if spans else 1)
    official_iou = float(np.mean([official_score_iou(g, p)
                                  for g, p in zip(official_gold, official_pred)]))
    official_cor = float(np.mean([official_score_cor(g, p)
                                  for g, p in zip(official_gold, official_pred)]))
    official_cor_lbl = float(np.mean([official_score_cor_lbl(g, p)
                                      for g, p in zip(official_gold, official_pred)]))
    # gold-gate ceiling: a PERFECT clean/dirty decision (clean -> empty, dirty -> the
    # locator's spans, no gating). This is the locator's headroom: if it is below the
    # target score, no gate work can get there — the locator itself must improve.
    gg = []
    for i in ids:
        it, q, p, t, g, bt, qsp = per[i]
        if not it.labels:
            gg.append(1.0)
        else:
            spans = decode(p, t, 1.0, best["tau"], 0.0, bt, qsp, it)
            gg.append(official_score_iou(official_gold_record(it),
                                         official_pred_record(it, spans)))
    # cleanOK <-> dirty Pareto sweep over the gate threshold at the best tau
    sweep = []
    for g_thr in (g_grid if decoder in ("bio", "gate", "gate_hyst") else ()):
        io, dio, cok, dopen, cclosed, nd, nc = [], [], 0, 0, 0, 0, 0
        for i in ids:
            it, q, p, t, g, bt, qsp = per[i]
            spans = decode(p, t, g, best["tau"], g_thr, bt, qsp, it)
            nulled = decoder_null_decision(decoder, p, g, best["tau"], g_thr, bt, qsp)
            score = official_score_iou(official_gold_record(it),
                                       official_pred_record(it, spans))
            io.append(score)
            if it.labels:
                nd += 1
                dio.append(score)
                dopen += (0 if nulled else 1)
            else:
                nc += 1
                cclosed += (1 if nulled else 0)
                cok += (0 if spans else 1)
        sweep.append(dict(threshold=g_thr, g_thr=g_thr,
                          gate_recall=round(dopen / max(1, nd), 3),
                          specificity=round(cclosed / max(1, nc), 3),
                          dirty_iou=round(float(np.mean(dio)), 4) if dio else 0.0,
                          cleanOK=round(cok / max(1, nc), 3),
                          overall_iou=round(float(np.mean(io)), 4)))
    # gate quality as a standalone expert: AUC + recall/specificity operating points
    g_all = np.array([float(per[i][4]) for i in ids])
    y_all = np.array([1.0 if per[i][0].labels else 0.0 for i in ids])
    try:
        gate_auc = float(roc_auc_score(y_all, g_all))
    except ValueError:
        gate_auc = 0.5
    gate_ops = []
    for thr in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        pred = g_all >= thr
        gate_ops.append(dict(
            thr=thr,
            dirty_recall=round(float(pred[y_all > 0].mean()), 3) if (y_all > 0).any() else 0.0,
            clean_spec=round(float((~pred[y_all == 0]).mean()), 3) if (y_all == 0).any() else 0.0))
    from shroom.metrics import spearman as _sp
    return per, dict(span_iou=official_iou, official_iou=official_iou,
                     official_cor=official_cor, official_cor_lbl=official_cor_lbl,
                     tau=best["tau"], g_thr=best["g_thr"], floor=floor,
                     gate_auc=gate_auc, gate_ops=gate_ops,
                     dirty_iou=float(np.mean(d_ious)) if d_ious else 0.0,
                     dirty_iou_ungated=float(np.mean(d_ungated)) if d_ungated else 0.0,
                     clean_empty=(clean_ok / n_clean) if n_clean else 1.0,
                     dirty_gate_recall=(dirty_open / n_dirty) if n_dirty else 0.0,
                     gate_specificity=(clean_closed / n_clean) if n_clean else 1.0,
                     iou_gate_open=float(np.mean(open_ious)) if open_ious else 0.0,
                     avg_spans=float(np.mean(nspans)),
                     gold_gate_iou=float(np.mean(gg)),
                     gate_sweep=sweep,
                     roc_auc=roc, pr_auc=pr,
                     pooled_pearson_debug=cal, pooled_spearman_debug=float(_sp(ga, pa)),
                     pooled_cor_lbl_debug=cal_lbl,
                     cor_submission=official_cor, cor_lbl=official_cor_lbl)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--arch", choices=["connector", "linear"], default="connector")
    ap.add_argument("--no_image", action="store_true")
    ap.add_argument("--eval_shuffle", action="store_true")
    ap.add_argument("--dirty_only", action="store_true",
                    help="V4 cascade locator: train only on answers WITH hallucinations")
    ap.add_argument("--gate_from", default=None,
                    help="V4 cascade: checkpoint of a separately trained gate expert; "
                         "its gate head replaces this model's at eval time")
    ap.add_argument("--gate_only", action="store_true",
                    help="V4 gate expert: train ONLY the clean/dirty decision — focal "
                         "loss, class-balanced sampling, best checkpoint by gate AUC")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--shuffle_v", action="store_true",
                    help="train-time control: every item gets the V of a different image "
                         "(same capacity, no visual signal); connector arch only")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambdas", default="1,1,0.5,0.5,0.2")
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_train", type=int, default=None, help="cap train items (learning curve)")
    ap.add_argument("--max_chars", type=int, default=4000,
                    help="char cap per answer (official scorer uses FULL length; 4000 covers all)")
    ap.add_argument("--eval_ids", default=None, help="json file with {'tune_dev': [...]} to eval on")
    ap.add_argument("--decoder", choices=["simple","gate","hyst","gate_hyst","v2","bio","set","seg","crf"], default="gate_hyst")
    ap.add_argument("--tversky", action="store_true", help="precision-weighted Tversky instead of soft-Jaccard")
    ap.add_argument("--no_type_loss", action="store_true")
    ap.add_argument("--bio", action="store_true", help="BIO boundary head + loss")
    ap.add_argument("--head", choices=["none", "set", "seg", "crf"], default="none",
                    help="'set' = DETR-style set-of-spans decoder (Hungarian matching)")
    ap.add_argument("--set_k", type=int, default=12)
    ap.add_argument("--l_set", type=float, default=1.0)
    ap.add_argument("--gate_consistency", action="store_true")
    ap.add_argument("--contrastive", action="store_true", help="group same image+question in batches + margin loss")
    ap.add_argument("--l_bio", type=float, default=1.0)
    ap.add_argument("--l_consist", type=float, default=0.3)
    ap.add_argument("--l_contrast", type=float, default=0.3)
    ap.add_argument("--no_gru", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--init_from", default=None, help="load model weights before train/eval")
    ap.add_argument("--device", default=None,
                    help="'cuda:N' / 'cpu' / 'mps'. Default: cuda if available, else cpu "
                         "('mps' stays opt-in). With CUDA_VISIBLE_DEVICES=3, 'cuda:0' IS GPU 3.")
    args = ap.parse_args()

    l1, l2, l3, l4, l5 = [float(x) for x in args.lambdas.split(",")]
    if args.dirty_only and not args.gate_only:
        # V4 loss hygiene: a dirty-only locator sees gate targets that are ALL 1, so the
        # gate BCE only teaches its (unused) gate head to say "dirty" and the
        # gate-consistency term drags char probs toward that constant. Both off.
        if l5 > 0 or args.gate_consistency:
            print("[loss] dirty_only: gate BCE and gate-consistency disabled "
                  "(the gate belongs to the gate expert)", flush=True)
        l5, args.gate_consistency = 0.0, False
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    # Default is unchanged (cuda if present, else cpu) — the Mac results were produced on
    # cpu, so 'mps' stays opt-in via --device to keep those runs reproducible.
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
            raise SystemExit(f"--device {device} unavailable (visible CUDA devices: "
                             f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}, "
                             f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})")
        torch.cuda.set_device(idx)
        print(f"[gpu ] {device} = {torch.cuda.get_device_name(idx)}", flush=True)
    else:
        print(f"[gpu ] running on {device}", flush=True)

    items = load_jsonl(args.train_file)
    tr_ids, dv_ids = group_split_by_image(items, dev_frac=args.dev_frac, seed=13)  # split seed FIXED
    if args.eval_ids:
        prot = json.load(open(args.eval_ids))
        dv_keep = set(prot.get("tune_dev") or prot.get("tune_dev200"))
        dv_ids = [i for i in dv_ids if i in dv_keep]
    tr = [it for it in items if it.id in set(tr_ids)]
    dv = [it for it in items if it.id in set(dv_ids)]
    if args.dirty_only:
        # V4 cascade locator: never sees a clean answer, so nothing pushes it toward
        # predicting empty — the clean/dirty decision belongs to the gate expert.
        n0 = len(tr)
        tr = [it for it in tr if it.labels]
        print(f"[data] dirty-only locator: train {n0} -> {len(tr)} (dev untouched)", flush=True)
    if args.max_train:
        tr = tr[:args.max_train]
    tr_ds = CacheDS(tr, args.cache_dir, max_chars=args.max_chars, shuffle_v=args.shuffle_v)
    dv_ds = CacheDS(dv, args.cache_dir, max_chars=args.max_chars, shuffle_v=args.shuffle_v)
    manifest = {"train_ids": [it.id for it in tr_ds.items], "dev_ids": [it.id for it in dv_ds.items],
                "seed": args.seed, "decoder": args.decoder, "gru": not args.no_gru}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"manifest_seed{args.seed}.json"), "w") as f:
        json.dump(manifest, f)
    print(f"[data] cached train={len(tr_ds)} dev={len(dv_ds)} (of {len(tr)}/{len(dv)}) — manifest frozen", flush=True)
    assert len(tr_ds) > 50, "too few cached items — run extract_features.py first"
    if args.gate_only:
        # class-balanced batches: clean answers are ~21% of train, and an unbalanced
        # gate expert just learns the prior. Weight so clean/dirty are drawn 50/50.
        lab = np.array([1.0 if it.labels else 0.0 for it in tr_ds.items])
        w = np.where(lab > 0, 0.5 / max(1, lab.sum()), 0.5 / max(1, (1 - lab).sum()))
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), num_samples=len(tr_ds),
            replacement=True, generator=torch.Generator().manual_seed(args.seed))
        tr_dl = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler, collate_fn=collate)
        print(f"[data] gate expert: balanced sampling over {int(lab.sum())} dirty / "
              f"{int((1 - lab).sum())} clean", flush=True)
    elif args.contrastive:
        tr_dl = DataLoader(tr_ds, batch_sampler=GroupedBatchSampler(tr_ds, args.batch, args.seed),
                           collate_fn=collate)
    else:
        tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dv_dl = DataLoader(dv_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    z0 = tr_ds[0]
    D, L = z0["H"].shape[2], z0["H"].shape[1]
    if args.arch == "connector" and z0["V"].shape[0] == 0:
        raise SystemExit(
            f"ERROR: --arch connector needs visual states, but the cache in {args.cache_dir} "
            "was extracted with --h_only (V is empty). Re-extract without --h_only.")
    model = Connector(D, L, dim=args.dim, blocks=args.blocks, arch=args.arch,
                      use_gru=not args.no_gru).to(device)
    # aux heads must exist BEFORE --init_from so a resumed checkpoint can restore them
    setdec = SetDecoder(args.dim, K=args.set_k).to(device) if args.head == "set" else None
    segsc = SegmentScorer(args.dim).to(device) if args.head == "seg" else None
    crf = SemiCRF(args.dim).to(device) if args.head == "crf" else None
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        if "model" in ck:
            model.load_state_dict(ck["model"], strict=False)
            if crf is not None and "crf" in ck: crf.load_state_dict(ck["crf"])
            if setdec is not None and "setdec" in ck: setdec.load_state_dict(ck["setdec"])
            if segsc is not None and "segsc" in ck: segsc.load_state_dict(ck["segsc"])
        else:
            model.load_state_dict(ck, strict=False)
        print(f"[model] loaded weights from {args.init_from}", flush=True)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] arch={args.arch} no_image={args.no_image} trainable={n_par/1e6:.1f}M "
          f"(backbone frozen, cached)", flush=True)
    gate_model = None
    if args.gate_from:
        gck = torch.load(args.gate_from, map_location=device)
        gstate = gck["model"] if "model" in gck else gck
        garch = "connector" if "alpha" in gstate else "linear"
        gate_model = Connector(D, L, dim=args.dim, blocks=args.blocks, arch=garch,
                               use_gru=not args.no_gru).to(device)
        gate_model.load_state_dict(gstate, strict=False)
        gate_model.eval()
        for p in gate_model.parameters():
            p.requires_grad_(False)
        print(f"[gate] external gate expert ({garch}) from {args.gate_from}", flush=True)
    params = (list(model.parameters()) + (list(setdec.parameters()) if setdec else [])
              + (list(segsc.parameters()) if segsc else []) + (list(crf.parameters()) if crf else []))
    opt = torch.optim.AdamW(params, lr=args.lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    tag = f"{args.arch}{'_noimg' if args.no_image else ''}{'_shufv' if args.shuffle_v else ''}{'_donly' if args.dirty_only else ''}{'_xgate' if args.gate_from else ''}{'_gateonly' if args.gate_only else ''}_{args.decoder}{'_tv' if args.tversky else ''}{'_bio' if args.bio else ''}{'_gc' if args.gate_consistency else ''}{'_ctr' if args.contrastive else ''}{'_set' if args.head=='set' else ''}{'_seg' if args.head=='seg' else ''}{'_crf' if args.head=='crf' else ''}{'_notype' if args.no_type_loss else ''}{'_nogru' if args.no_gru else ''}_s{args.seed}"
    t0 = time.time()
    best_iou_seen = (-1.0, 0)
    best_path = None
    for ep in range(1, 0 if args.eval_only else args.epochs + 1):
        model.train(); tot = nb = 0
        for H, V, vmask, t2c, inpos, y, ytype, bio_t, seg_l, valid, gate, bitems in tr_dl:
            if args.no_image:
                V = torch.zeros_like(V)
            H, V, vmask = H.to(device), V.to(device), vmask.to(device)
            t2c, inpos = t2c.to(device), inpos.to(device)
            y, ytype, valid, gate = (y.to(device), ytype.to(device),
                                     valid.to(device), gate.to(device))
            bio_t = bio_t.to(device)
            ql, pl, tl, gl, bl, feats = model(H, V, t2c, inpos, vmask)
            if args.gate_only:
                # focal BCE on the gate head alone — the expert's whole job
                pt = torch.sigmoid(gl) * gate + (1 - torch.sigmoid(gl)) * (1 - gate)
                loss = ((1 - pt).clamp(min=1e-6) ** args.focal_gamma * bce(gl, gate)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
                if nb % 100 == 0:
                    print(f"[ep {ep} b{nb}/{len(tr_dl)}] loss={tot/nb:.4f}", flush=True)
                continue
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
            p_sig = torch.sigmoid(pl) * valid
            topk = torch.stack([                      # per-example k over VALID chars only
                (p_sig[bi][valid[bi] > 0].topk(max(3, int(valid[bi].sum()) // 20))[0].mean()
                 if int(valid[bi].sum()) >= 3 else p_sig[bi].max())
                for bi in range(p_sig.shape[0])])
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
            if setdec is not None:
                from scipy.optimize import linear_sum_assignment
                pad = (t2c < 0)
                sl_, el_, cf_, tp_, sp_ = setdec(feats, pad)
                lsm_s = torch.log_softmax(sl_, -1); lsm_e = torch.log_softmax(el_, -1)
                sloss, nb2 = feats.new_zeros(()), 0
                for b3 in range(feats.shape[0]):
                    segs = seg_l[b3]
                    K = sl_.shape[1]
                    tgt_conf = torch.zeros(K, device=device)
                    if segs:
                        with torch.no_grad():
                            C_ = np.zeros((K, len(segs)))
                            for gi, (a, b4, ti, pr) in enumerate(segs):
                                C_[:, gi] = (-lsm_s[b3, :, a] - lsm_e[b3, :, b4 - 1]).cpu().numpy()
                        rows, cols = linear_sum_assignment(C_)
                        for kq, gi in zip(rows, cols):
                            a, b4, ti, pr = segs[gi]
                            sloss = sloss - lsm_s[b3, kq, a] - lsm_e[b3, kq, b4 - 1]
                            sloss = sloss + nn.functional.cross_entropy(
                                tp_[b3, kq].unsqueeze(0),
                                torch.tensor([ti], device=device))
                            sloss = sloss + (torch.sigmoid(sp_[b3, kq]) - pr) ** 2
                            tgt_conf[kq] = 1.0
                    w = torch.where(tgt_conf > 0, torch.tensor(1.0, device=device),
                                    torch.tensor(0.2, device=device))
                    sloss = sloss + (nn.functional.binary_cross_entropy_with_logits(
                        cf_[b3], tgt_conf, weight=w))
                    nb2 += 1
                loss = loss + args.l_set * sloss / max(nb2, 1)
            if segsc is not None:
                gloss, ng = feats.new_zeros(()), 0
                for b3 in range(feats.shape[0]):
                    n3 = int(valid[b3].sum())
                    if n3 < 4:
                        continue
                    xb = feats[b3, :n3]
                    cands, labels, tids = [], [], []
                    for (a, b4, ti, pr) in seg_l[b3]:
                        if b4 <= n3:
                            cands.append((a, b4)); labels.append(1.0); tids.append(ti)
                            for da, db in ((-6, 0), (6, 0), (0, 8), (0, -8) if b4-a > 10 else (0, 4),
                                           (-15, 15)):
                                a2, b5 = max(0, a + da), min(n3, b4 + db)
                                if b5 - a2 >= 2 and (a2, b5) != (a, b4):
                                    cands.append((a2, b5)); labels.append(0.0); tids.append(-1)
                    rng2 = random.Random(int(gate[b3].item() * 7) + b3)
                    for _ in range(12):
                        L2 = rng2.choice((3, 6, 12, 25, 50))
                        a2 = rng2.randrange(0, max(1, n3 - L2))
                        cands.append((a2, min(n3, a2 + L2))); labels.append(0.0); tids.append(-1)
                    sc, tp2 = segsc(xb, cands)
                    lab_t = torch.tensor(labels, device=device)
                    gloss = gloss + nn.functional.binary_cross_entropy_with_logits(sc, lab_t)
                    pos = [i2 for i2, t3 in enumerate(tids) if t3 >= 0]
                    if pos:
                        gloss = gloss + 0.5 * nn.functional.cross_entropy(
                            tp2[pos], torch.tensor([tids[i2] for i2 in pos], device=device))
                    ng += 1
                loss = loss + args.l_set * gloss / max(ng, 1)
            if crf is not None:
                seg_ab = [[(a, b4) for (a, b4, ti, pr) in sl_] for sl_ in seg_l]
                loss = loss + args.l_set * crf.nll(feats, valid, seg_ab)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if nb % 100 == 0:
                print(f"[ep {ep} b{nb}/{len(tr_dl)}] loss={tot/nb:.4f}", flush=True)
        _, m = run_eval(model, dv_dl, device, no_image=args.no_image, decoder=args.decoder, setdec=setdec, segsc=segsc, crf=crf, gate_model=gate_model)
        sel = m["gate_auc"] if args.gate_only else m["span_iou"]
        if sel > best_iou_seen[0]:
            best_iou_seen = (sel, ep)
            st_b = {"model": model.state_dict(), "epoch": ep, "metrics": m}
            if crf is not None: st_b["crf"] = crf.state_dict()
            if setdec is not None: st_b["setdec"] = setdec.state_dict()
            if segsc is not None: st_b["segsc"] = segsc.state_dict()
            best_path = os.path.join(args.out_dir, f"best_iou_{tag}.pt")
            torch.save(st_b, best_path)
        alpha_s = (f" alpha={float(model.alpha):+.5f}"
                   if args.arch == "connector" else "")
        print(f"[ep {ep}] loss={tot/nb:.4f} dev: iou={m['official_iou']:.4f} (fl={m['floor']:.3f}, "
              f"tau={m['tau']}, g={m['g_thr']}) dirty={m['dirty_iou']:.3f} cleanOK={m['clean_empty']:.2f} "
              f"gateRec={m['dirty_gate_recall']:.2f} gateSpec={m['gate_specificity']:.2f} "
              f"Cor={m['official_cor']:.3f} Cor_lbl={m['official_cor_lbl']:.3f} "
              f"poolR={m['pooled_pearson_debug']:.3f} "
              f"gAUC={m['gate_auc']:.3f}"
              f"{alpha_s} [{(time.time()-t0)/60:.1f}m]", flush=True)

    if best_path:
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck["model"], strict=False)
        if crf is not None and "crf" in ck: crf.load_state_dict(ck["crf"])
        if setdec is not None and "setdec" in ck: setdec.load_state_dict(ck["setdec"])
        if segsc is not None and "segsc" in ck: segsc.load_state_dict(ck["segsc"])
        print(f"[best] loaded epoch {ck.get('epoch')} from {best_path} for final predictions", flush=True)
    per, m = run_eval(model, dv_dl, device, no_image=args.no_image, decoder=args.decoder, setdec=setdec, segsc=segsc, crf=crf, gate_model=gate_model)
    results = {"variant": tag, "metrics": m, "best_checkpoint": best_path}
    if args.eval_shuffle and not args.no_image:
        _, ms = run_eval(model, dv_dl, device, shuffle=True, decoder=args.decoder)
        results["metrics_shuffled_image"] = ms
        print(f"[control] shuffled-image: span_iou={ms['span_iou']:.4f} roc={ms['roc_auc']:.3f}")
    # official JSONL
    pred_path = os.path.join(args.out_dir, f"dev_pred_{tag}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, (it, q, p, t, g, bt, qsp) in per.items():
            spans = decode_spans(args.decoder, p, t, g, m["tau"], m["g_thr"],
                                 bt, qsp, resp_len=len(it.response))
            f.write(json.dumps({"id": i, "labels": spans,          # official field
                                "language": it.language, "response": it.response,
                                "pred_labels": spans,
                                "char_probs": [round(float(x), 3) for x in p]},
                               ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    state = {"model": model.state_dict()}
    if setdec is not None: state["setdec"] = setdec.state_dict()
    if segsc is not None: state["segsc"] = segsc.state_dict()
    if crf is not None: state["crf"] = crf.state_dict()
    torch.save(state, os.path.join(args.out_dir, f"connector_{tag}.pt"))
    print(json.dumps(results, indent=2))
    print(f"predictions -> {pred_path}")


if __name__ == "__main__":
    main()
