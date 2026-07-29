"""Score predictions with the ORGANIZERS' scorer (vendored: scripts/official/official_scorer.py).

Uses the official score_cor (per-item Spearman, mean), score_cor_lbl and score_iou.
Accepts our internal prediction files (`pred_labels`) or official ones (`labels`), and
can emit an official-format submission file. Adds paired image-cluster bootstrap CIs.

  python scripts/official/eval_official.py --gold splits/dev.en.jsonl \
      --eval_ids splits/en.eval_protocol.json \
      --pred v3-BIO=results/v3_mac/dev_pred_*.jsonl [--pred ...] \
      [--write_official out.jsonl]
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from official_scorer import score_cor, score_cor_lbl, score_iou  # noqa: E402


def load_gold(path, keep_ids=None):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if keep_ids and d["id"] not in keep_ids:
            continue
        out[d["id"]] = {"id": d["id"], "labels": d.get("labels") or [],
                        "text_len": len(d["response"]), "image": d.get("image_name", "")}
    return out


def load_pred(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        spans = d.get("labels", d.get("pred_labels")) or []
        clean = []
        for s in spans:                       # official checker demands exact types
            clean.append({"start": int(s["start"]), "end": int(s["end"]),
                          "prob": float(s.get("prob", 1.0)),
                          "label": str(s.get("label", "other"))})
        out[d["id"]] = {"id": d["id"], "labels": clean}
    return out


def per_item(gold, pred):
    rows = {}
    for i, g in gold.items():
        p = pred.get(i, {"id": i, "labels": []})
        rows[i] = {"cor": float(score_cor(g, p)), "cor_lbl": float(score_cor_lbl(g, p)),
                   "iou": float(score_iou(g, p)), "image": g["image"],
                   "clean": not g["labels"]}
    return rows


def agg(rows):
    v = list(rows.values())
    return {"Cor": np.mean([r["cor"] for r in v]),
            "Cor_lbl": np.mean([r["cor_lbl"] for r in v]),
            "IoU": np.mean([r["iou"] for r in v]),
            "n": len(v)}


def cluster_ci(rows_a, rows_b=None, key="iou", n_boot=2000, seed=0):
    ids = list(rows_a)
    by_img = {}
    for i in ids:
        by_img.setdefault(rows_a[i]["image"], []).append(i)
    imgs = sorted(by_img)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sel = [i for im in (rng.choice(imgs) for _ in imgs) for i in by_img[im]]
        a = np.mean([rows_a[i][key] for i in sel])
        vals.append(a - np.mean([rows_b[i][key] for i in sel]) if rows_b else a)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--eval_ids", default=None)
    ap.add_argument("--pred", action="append", required=True, help="name=path")
    ap.add_argument("--write_official", default=None, help="name:outpath — official format")
    args = ap.parse_args()

    keep = None
    if args.eval_ids:
        prot = json.load(open(args.eval_ids))
        keep = set(prot.get("tune_dev") or prot.get("tune_dev200"))
    gold = load_gold(args.gold, keep)

    # trivial baselines
    systems = [("predict-nothing", {i: {"id": i, "labels": []} for i in gold}),
               ("predict-everything",
                {i: {"id": i, "labels": [{"start": 0, "end": g["text_len"], "prob": 1.0,
                                          "label": "other"}]} for i, g in gold.items()})]
    for spec in args.pred:
        name, path = spec.split("=", 1)
        systems.append((name, load_pred(path)))

    print(f"OFFICIAL SCORER — gold={args.gold} n={len(gold)}"
          + (f" (eval_ids: {len(keep)})" if keep else ""))
    print(f"{'system':28}{'IoU':>8}{'Cor':>8}{'Cor_lbl':>9}   {'IoU 95% CI (image-cluster)'}")
    all_rows = {}
    for name, pred in systems:
        rows = per_item(gold, pred)
        all_rows[name] = rows
        a = agg(rows)
        lo, hi = cluster_ci(rows)
        print(f"{name:28}{a['IoU']:>8.4f}{a['Cor']:>8.4f}{a['Cor_lbl']:>9.4f}   [{lo:.4f}, {hi:.4f}]")

    base = "predict-nothing"
    print(f"\npaired deltas vs {base} (image-cluster bootstrap):")
    for name in all_rows:
        if name == base:
            continue
        for key, lbl in (("iou", "IoU"), ("cor", "Cor")):
            d = np.mean([all_rows[name][i][key] for i in gold]) - \
                np.mean([all_rows[base][i][key] for i in gold])
            lo, hi = cluster_ci(all_rows[name], all_rows[base], key)
            sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "ns"
            print(f"  {name:26} {lbl:8} {d:+.4f}  [{lo:+.4f},{hi:+.4f}]  {sig}")

    if args.write_official:
        nm, outp = args.write_official.split(":", 1)
        pred = dict(systems)[nm]
        with open(outp, "w", encoding="utf-8") as f:
            for i in sorted(gold):
                f.write(json.dumps({"id": i, "labels": pred.get(i, {}).get("labels", [])},
                                   ensure_ascii=False) + "\n")
        print(f"\nofficial-format submission -> {outp}")


if __name__ == "__main__":
    main()
