"""Span-type classifier: the missing type head, trained standalone on cached features.

Every system so far ships a dead type head (Cor_lbl at/below floor; the submission
used a constant label). This trains a small class-balanced classifier on GOLD spans
— X = mean cached hidden state over the span's tokens (+ length/position scalars),
y = the span's type — and then RELABELS the spans of any prediction file. Because it
operates at postprocess level it composes with every locator, every language and
every backbone whose features are cached.

  # train (minutes; excludes protocol eval items from training)
  python scripts/connector/span_type_classifier.py train \
      --cache_dir results/cache_h100_bf16 \
      --train_file ../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl \
      --eval_ids splits/en.eval_protocol.json --out results/type_clf_en.pt

  # relabel a prediction file (dev preds to measure, or a submission to ship)
  python scripts/connector/span_type_classifier.py apply \
      --clf results/type_clf_en.pt --cache_dir results/cache_h100_bf16 \
      --pred results/final/a2_dev/dev_pred_....jsonl --out relabeled.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shroom.data import load_jsonl                       # noqa: E402
from model import CATS                                   # noqa: E402


def span_feature(z, start, end):
    """Mean layer-averaged hidden state over the tokens overlapping [start, end),
    plus normalized length and position scalars."""
    H, tc = z["H"], z["tok_char"]
    L = int(z["answer_len"])
    rows = [i for i, (a, b) in enumerate(tc) if b > start and a < end]
    if not rows:
        return None
    x = H[rows].astype(np.float32).mean(axis=(0, 1))     # [D]
    extra = np.array([(end - start) / 50.0, start / max(1, L), end / max(1, L)],
                     dtype=np.float32)
    return np.concatenate([x, extra])


class TypeClf(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(256, len(CATS)))

    def forward(self, x):
        return self.net(x)


def load_cached(cache_dir, rid):
    p = os.path.join(cache_dir, f"{rid}.npz")
    return np.load(p) if os.path.exists(p) else None


def cmd_train(args):
    proto = json.load(open(args.eval_ids)) if args.eval_ids else {}
    excl = set(proto.get("tune_dev") or []) | set(proto.get("heldout") or [])
    items = [it for it in load_jsonl(args.train_file) if it.id not in excl]
    X, y = [], []
    for it in items:
        z = load_cached(args.cache_dir, it.id)
        if z is None:
            continue
        for sp in it.labels:
            f = span_feature(z, sp["start"], sp["end"])
            if f is None or sp.get("label") not in CATS:
                continue
            X.append(f)
            y.append(CATS.index(sp["label"]))
    X = torch.tensor(np.stack(X))
    y = torch.tensor(y)
    counts = torch.bincount(y, minlength=len(CATS)).float()
    print(f"[data] {len(y)} gold spans; per class "
          f"{dict(zip(CATS, counts.int().tolist()))}", flush=True)
    w = counts.sum() / counts.clamp(min=1) / len(CATS)

    torch.manual_seed(args.seed)
    perm = torch.randperm(len(y))
    n_dev = max(200, len(y) // 10)
    dv, tr = perm[:n_dev], perm[n_dev:]
    clf = TypeClf(X.shape[1])
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=w)
    best_acc, best_state = -1.0, None
    for ep in range(args.epochs):
        clf.train()
        for i in range(0, len(tr), 256):
            b = tr[i:i + 256]
            opt.zero_grad()
            ce(clf(X[b]), y[b]).backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            pred = clf(X[dv]).argmax(1)
            acc = float((pred == y[dv]).float().mean())
            # balanced accuracy: mean per-class recall (majority guessing scores 0.2)
            bacc = float(np.mean([((pred == c) & (y[dv] == c)).sum() /
                                  max(1, (y[dv] == c).sum()) for c in range(len(CATS))]))
        print(f"[ep {ep + 1}] dev acc={acc:.3f} balanced={bacc:.3f}", flush=True)
        if bacc > best_acc:
            best_acc, best_state = bacc, {k: v.clone() for k, v in clf.state_dict().items()}
    torch.save({"state": best_state, "dim": X.shape[1], "cats": CATS,
                "balanced_acc": best_acc}, args.out)
    print(f"saved -> {args.out} (best balanced acc {best_acc:.3f}; "
          f"majority-class guessing = {1/len(CATS):.2f})")


def cmd_apply(args):
    ck = torch.load(args.clf, map_location="cpu")
    clf = TypeClf(ck["dim"])
    clf.load_state_dict(ck["state"])
    clf.eval()
    n_rel, n_kept, out_rows = 0, 0, []
    for ln in open(args.pred, encoding="utf-8"):
        d = json.loads(ln)
        z = load_cached(args.cache_dir, d["id"])
        for sp in d.get("labels", []):
            f = span_feature(z, sp["start"], sp["end"]) if z is not None else None
            if f is None:
                n_kept += 1
                continue
            with torch.no_grad():
                sp["label"] = CATS[int(clf(torch.tensor(f)[None]).argmax())]
            n_rel += 1
        out_rows.append({"id": d["id"], "labels": d.get("labels", [])})
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"relabeled {n_rel} spans ({n_kept} kept old label, no features) -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--cache_dir", required=True)
    t.add_argument("--train_file", required=True)
    t.add_argument("--eval_ids", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--seed", type=int, default=13)
    a = sub.add_parser("apply")
    a.add_argument("--clf", required=True)
    a.add_argument("--cache_dir", required=True)
    a.add_argument("--pred", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    (cmd_train if args.cmd == "train" else cmd_apply)(args)


if __name__ == "__main__":
    main()
