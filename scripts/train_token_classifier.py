"""Fine-tune pilot: multimodal token classifier for hallucination spans (CUDA).

Architecture (deliberately small & fast, to detect signal in ~tens of minutes):
  - Text: XLM-RoBERTa-base token encoder over "question [SEP] answer".
  - Image: frozen SigLIP vision tower -> pooled embedding -> linear projection ->
    injected as a virtual prefix token (via inputs_embeds).
  - Head: linear per-token logit, trained with BCE against SOFT labels = the
    per-character annotator probability (max over chars in the token). So the
    model learns calibrated probabilities natively.

Data: SHROOM-visions labeled train (en by default), split by image (same seed/logic
as shroom.split) into train/dev. Eval maps token probs back to characters and scores
span_iou (best threshold on dev), ROC-AUC, PR-AUC and calibration vs the trivial
baselines — printed at the end.

Run `--no_image` for the vision-ablation control: if scores match the image run,
the model is not using the image.

Example (single A100):
  python scripts/train_token_classifier.py \
      --train_file ../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl \
      --image_dir ../Shroom-Vision/images --epochs 3 --out_dir results/ft_pilot
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shroom.data import load_jsonl                      # noqa: E402
from shroom.split import group_split_by_image           # noqa: E402
from shroom.metrics import gold_char_probs, char_iou, pearson  # noqa: E402

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score     # noqa: E402

TEXT_MODEL = "xlm-roberta-base"
VISION_MODEL = "google/siglip-base-patch16-224"


# --------------------------------------------------------------------------- data
class SpanDataset(Dataset):
    def __init__(self, items, tokenizer, image_embeds, max_len=384):
        self.items = items
        self.tok = tokenizer
        self.embeds = image_embeds  # image_name -> np.array or None
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def encode(self, it):
        # question + answer; labels only over answer chars
        q, a = it.prompt, it.response
        enc = self.tok(q, a, truncation="only_second", max_length=self.max_len,
                       return_offsets_mapping=True, return_tensors="pt")
        offsets = enc["offset_mapping"][0].tolist()
        seq_ids = enc.sequence_ids(0)
        cprobs = gold_char_probs(it.labels, len(a))
        labels, mask = [], []
        for k, (s, e) in enumerate(offsets):
            if seq_ids[k] != 1 or e <= s:          # not an answer token
                labels.append(0.0); mask.append(0.0)
            else:
                labels.append(max(cprobs[s:e]) if e <= len(cprobs) else 0.0)
                mask.append(1.0)
        emb = self.embeds.get(it.image_name)
        return (enc["input_ids"][0], enc["attention_mask"][0],
                torch.tensor(labels), torch.tensor(mask),
                torch.tensor(emb, dtype=torch.float32) if emb is not None else None,
                it, offsets, seq_ids)

    def __getitem__(self, i):
        return self.encode(self.items[i])


def collate(batch, img_dim):
    maxlen = max(x[0].shape[0] for x in batch)
    def pad(t, val):
        return torch.cat([t, torch.full((maxlen - t.shape[0],), val, dtype=t.dtype)])
    ids = torch.stack([pad(b[0], 1) for b in batch])          # 1 = xlmr pad id
    att = torch.stack([pad(b[1], 0) for b in batch])
    lab = torch.stack([pad(b[2], 0.0) for b in batch])
    msk = torch.stack([pad(b[3], 0.0) for b in batch])
    img = torch.stack([b[4] if b[4] is not None else torch.zeros(img_dim) for b in batch])
    metas = [(b[5], b[6], b[7]) for b in batch]
    return ids, att, lab, msk, img, metas


# --------------------------------------------------------------------------- model
class TokenClassifier(nn.Module):
    def __init__(self, img_dim, use_image=True):
        super().__init__()
        self.text = AutoModel.from_pretrained(TEXT_MODEL)
        h = self.text.config.hidden_size
        self.img_proj = nn.Linear(img_dim, h)
        self.use_image = use_image
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, ids, att, img):
        embeds = self.text.embeddings(input_ids=ids)
        if self.use_image:
            prefix = self.img_proj(img).unsqueeze(1)          # [B,1,H] virtual token
            embeds = torch.cat([prefix, embeds[:, 1:]], dim=1)  # replace <s> position
        out = self.text(inputs_embeds=embeds, attention_mask=att).last_hidden_state
        return self.head(out).squeeze(-1)                      # [B,L] logits


# --------------------------------------------------------------------------- image embeds
@torch.no_grad()
def compute_image_embeds(names, image_dir, device, cache_path):
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True).item()
        if all(n in z for n in names):
            print(f"[img] cache hit: {cache_path} ({len(z)} embeds)")
            return z
    print(f"[img] embedding {len(names)} images with {VISION_MODEL} ...", flush=True)
    proc = AutoImageProcessor.from_pretrained(VISION_MODEL)
    vm = AutoModel.from_pretrained(VISION_MODEL).vision_model.to(device).eval()
    out = {}
    batch, keys = [], []
    def flush():
        if not batch:
            return
        px = proc(images=batch, return_tensors="pt").to(device)
        emb = vm(**px).pooler_output.float().cpu().numpy()
        for k, e in zip(keys, emb):
            out[k] = e
        batch.clear(); keys.clear()
    for n in names:
        p = os.path.join(image_dir, n)
        if not os.path.exists(p):
            out[n] = None; continue
        try:
            batch.append(Image.open(p).convert("RGB")); keys.append(n)
        except Exception:
            out[n] = None; continue
        if len(batch) == 32:
            flush()
    flush()
    np.save(cache_path, out)      # noqa: allow_pickle default True for dict
    print(f"[img] saved cache -> {cache_path}")
    return out


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def predict_char_probs(model, ds, loader, device):
    model.eval()
    per_item = {}
    for ids, att, lab, msk, img, metas in loader:
        logits = model(ids.to(device), att.to(device), img.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        for b, (it, offsets, seq_ids) in enumerate(metas):
            cp = np.zeros(len(it.response))
            for k, (s, e) in enumerate(offsets):
                if k < probs.shape[1] and seq_ids[k] == 1 and e > s and e <= len(cp):
                    cp[s:e] = np.maximum(cp[s:e], probs[b, k])
            per_item[it.id] = (it, cp)
    return per_item


def evaluate(per_item, taus):
    ids = list(per_item)
    gold_all, pred_all = [], []
    for i in ids:
        it, cp = per_item[i]
        gold_all += gold_char_probs(it.labels, len(it.response))
        pred_all += cp.tolist()
    gb = [1 if g > 0 else 0 for g in gold_all]
    roc = roc_auc_score(gb, pred_all) if len(set(gb)) > 1 else float("nan")
    pr = average_precision_score(gb, pred_all)
    cal = pearson(gold_all, pred_all)
    floor = np.mean([1.0 if not per_item[i][0].labels else 0.0 for i in ids])
    best = (0.5, -1)
    for t in taus:
        ious = []
        for i in ids:
            it, cp = per_item[i]
            spans = probs_to_spans(cp, t)
            ious.append(char_iou(it.labels, spans, len(it.response)))
        m = float(np.mean(ious))
        if m > best[1]:
            best = (t, m)
    return dict(roc_auc=roc, pr_auc=pr, calib=cal, span_iou=best[1], tau=best[0],
                floor=float(floor))


def probs_to_spans(cp, tau):
    spans, i = [], 0
    while i < len(cp):
        if cp[i] >= tau:
            j = i
            while j < len(cp) and cp[j] >= tau:
                j += 1
            spans.append({"start": i, "end": j, "prob": float(cp[i:j].mean()), "label": "other"})
            i = j
        else:
            i += 1
    return spans


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", default="results/ft_pilot")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no_image", action="store_true", help="vision-ablation control")
    ap.add_argument("--max_train", type=int, default=None, help="cap train items (quick runs)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}  torch={torch.__version__}", flush=True)

    items = load_jsonl(args.train_file)
    train_ids, dev_ids = group_split_by_image(items, dev_frac=args.dev_frac, seed=args.seed)
    tr = [it for it in items if it.id in set(train_ids)]
    dv = [it for it in items if it.id in set(dev_ids)]
    if args.max_train:
        tr = tr[:args.max_train]
    print(f"[data] train={len(tr)}  dev={len(dv)}  (image-grouped split, seed={args.seed})")

    names = sorted({it.image_name for it in tr + dv})
    cache = os.path.join(args.out_dir, "img_embeds.npy")
    embeds = compute_image_embeds(names, args.image_dir, device, cache)
    img_dim = next(v for v in embeds.values() if v is not None).shape[0]

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    tr_ds = SpanDataset(tr, tok, embeds, args.max_len)
    dv_ds = SpanDataset(dv, tok, embeds, args.max_len)
    coll = lambda b: collate(b, img_dim)  # noqa: E731
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=coll)
    dv_dl = DataLoader(dv_ds, batch_size=args.batch, shuffle=False, collate_fn=coll)

    model = TokenClassifier(img_dim, use_image=not args.no_image).to(device)
    opt = torch.optim.AdamW([
        {"params": model.text.parameters(), "lr": args.lr},
        {"params": list(model.img_proj.parameters()) + list(model.head.parameters()),
         "lr": 1e-3},
    ])
    bce = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = nb = 0
        for ids, att, lab, msk, img, _ in tr_dl:
            ids, att = ids.to(device), att.to(device)
            lab, msk, img = lab.to(device), msk.to(device), img.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(ids, att, img)
                loss = (bce(logits, lab) * msk).sum() / msk.sum().clamp(min=1)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        per_item = predict_char_probs(model, dv_ds, dv_dl, device)
        m = evaluate(per_item, taus)
        print(f"[ep {ep}] loss={tot/nb:.4f}  dev: span_iou={m['span_iou']:.4f} (tau={m['tau']}) "
              f"floor={m['floor']:.4f}  roc_auc={m['roc_auc']:.3f}  pr_auc={m['pr_auc']:.3f}  "
              f"calib={m['calib']:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    # final: write predictions + summary (durable)
    per_item = predict_char_probs(model, dv_ds, dv_dl, device)
    m = evaluate(per_item, taus)
    tag = "no_image" if args.no_image else "with_image"
    pred_path = os.path.join(args.out_dir, f"dev_pred_{tag}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, (it, cp) in per_item.items():
            f.write(json.dumps({"id": i, "language": it.language, "response": it.response,
                                "pred_labels": probs_to_spans(cp, m["tau"]),
                                "char_probs": [round(float(x), 3) for x in cp]},
                               ensure_ascii=False) + "\n")
    summary = {"variant": tag, "train_items": len(tr), "dev_items": len(dv),
               "epochs": args.epochs, "metrics": m, "minutes": round((time.time()-t0)/60, 1)}
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{tag}.pt"))
    print("\n=== FINAL (dev) ===")
    print(json.dumps(summary, indent=2))
    print(f"\npredictions -> {pred_path}")
    print("SIGNAL CHECK: span_iou must beat the floor; compare --no_image vs default to "
          "verify the image contributes.")


if __name__ == "__main__":
    main()
