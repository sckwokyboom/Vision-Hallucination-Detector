"""Offline (CPU, seconds) analysis of the raw dumps written by eval_ablation.py.

Reads <raw_dir>/<condition>.raw.jsonl (+ manifest.json) and the gold JSONL, decodes
with decode_spans('bio') exactly like the trainer, applies the submission
post-processing (trim edges -> min span length -> merge gap), scores with the vendored
official scorer, and prints three things:

  1. conditions      correct / shuffled / no_image at the FIXED submitted thresholds,
                     raw and post-processed, plus an oracle re-sweep of (tau, g_thr) per
                     condition (upper bound), with paired image-cluster bootstrap CIs
                     of the IoU delta vs `correct`
  2. postproc grid   min_len x merge_gap on `correct` at fixed thresholds — the
                     sensitivity the reviewers asked for (min_len 1 vs 3 on zh)
  3. transfer        (optional) the score under thresholds + post-processing fixed on
                     ANOTHER track (e.g. en's tau=0, g=0.5, min_len=3, gap=2) vs the
                     track's own — how much per-language tuning on dev bought

  python scripts/official/ablation_sweeps.py --raw_dir results/ablation/zh_dev \
      --gold splits/dev.zh.jsonl --min_len 3 --merge_gap 2 --transfer 0.0,0.5,3,2
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "connector"))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
import eval_official                                                  # noqa: E402
from postprocess_submission import postprocess_rows                   # noqa: E402
from train_connector import decode_spans, CATS                        # noqa: E402
from shroom.data import load_jsonl                                    # noqa: E402

TAUS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
G_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load_raw(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["id"]] = d
    return out


def decode_all(raw, items, tau, g_thr):
    """Trainer-identical decode at one operating point -> {id: [spans]}."""
    out = {}
    for it in items:
        r = raw.get(it.id)
        if not r or not r["p"]:
            out[it.id] = []
            continue
        p = np.asarray(r["p"], dtype=float)
        bt = np.asarray(r["bio"], dtype=int)
        t = np.zeros((len(p), len(CATS)), dtype=float)
        t[np.arange(len(p)), np.asarray(r["targ"], dtype=int)] = 1.0
        out[it.id] = decode_spans("bio", p, t, float(r["g"]), tau, g_thr, bt, None,
                                  resp_len=len(it.response))
    return out


def postproc(spans_by_id, texts, min_len, merge_gap):
    rows = [{"id": i, "labels": s} for i, s in spans_by_id.items()]
    rows = postprocess_rows(rows, texts, trim_edges=True, trim_symbols=False,
                            min_len=min_len, min_prob=0.0, label_policy="keep",
                            default_label="invention", merge_gap=merge_gap,
                            keep_ids=None, snap="off")
    return {r["id"]: r for r in rows}


def score(gold, spans_by_id):
    pred = {i: {"id": i, "labels": s} for i, s in spans_by_id.items()}
    per = eval_official.per_item(gold, pred)
    a = eval_official.agg(per)
    empty = float(np.mean([0.0 if pred[i]["labels"] else 1.0 for i in gold]))
    dirty = [r["iou"] for r in per.values() if not r["clean"]]
    clean_ok = [0.0 if pred[i]["labels"] else 1.0 for i, r in per.items() if r["clean"]]
    return per, dict(IoU=a["IoU"], Cor=a["Cor"], empty=empty,
                     dirty_iou=float(np.mean(dirty)) if dirty else float("nan"),
                     clean_ok=float(np.mean(clean_ok)) if clean_ok else float("nan"))


def iou_only(gold, spans_by_id):
    from official_scorer import score_iou
    return float(np.mean([score_iou(g, {"id": i, "labels": spans_by_id.get(i, [])})
                          for i, g in gold.items()]))


def resweep(raw, items, gold, texts, min_len, merge_gap):
    best = (-1.0, None, None)
    for g_thr in G_GRID:
        for tau in TAUS:
            sp = postproc(decode_all(raw, items, tau, g_thr), texts, min_len, merge_gap)
            v = iou_only(gold, {i: r["labels"] for i, r in sp.items()})
            if v > best[0]:
                best = (v, tau, g_thr)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--tau", type=float, default=None, help="default: manifest's")
    ap.add_argument("--g_thr", type=float, default=None, help="default: manifest's")
    ap.add_argument("--min_len", type=int, default=3, help="submitted min span length")
    ap.add_argument("--merge_gap", type=int, default=2, help="submitted merge gap")
    ap.add_argument("--min_len_grid", default="1,2,3,4,5")
    ap.add_argument("--merge_gap_grid", default="0,1,2,3")
    ap.add_argument("--transfer", default=None,
                    help="tau,g_thr,min_len,merge_gap fixed on another track (e.g. 0.0,0.5,3,2)")
    ap.add_argument("--no_resweep", action="store_true")
    ap.add_argument("--out", default=None, help="JSON with every number printed")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.raw_dir, "manifest.json")))
    tau = args.tau if args.tau is not None else float(man["tau"])
    g_thr = args.g_thr if args.g_thr is not None else float(man["g_thr"])
    ids = [json.loads(l)["id"] for l in open(os.path.join(args.raw_dir, "correct.raw.jsonl"),
                                             encoding="utf-8") if l.strip()]
    keep = set(ids)
    items = [it for it in load_jsonl(args.gold) if it.id in keep]
    texts = {it.id: it.response for it in items}
    gold = {it.id: {"id": it.id, "labels": it.labels, "text_len": len(it.response),
                    "image": it.image_name} for it in items}
    floor = float(np.mean([1.0 if not it.labels else 0.0 for it in items]))
    report = {"raw_dir": args.raw_dir, "gold": args.gold, "n": len(items), "floor": floor,
              "fixed": {"tau": tau, "g_thr": g_thr, "min_len": args.min_len,
                        "merge_gap": args.merge_gap}}
    print(f"== {args.raw_dir}: n={len(items)} floor(predict-nothing)={floor:.4f} "
          f"fixed tau={tau} g_thr={g_thr} min_len={args.min_len} merge_gap={args.merge_gap}")

    # 1. conditions ------------------------------------------------------------------
    conds = [c for c in ("correct", "shuffled", "no_image")
             if os.path.exists(os.path.join(args.raw_dir, f"{c}.raw.jsonl"))]
    raws = {c: load_raw(os.path.join(args.raw_dir, f"{c}.raw.jsonl")) for c in conds}
    per_ref = {}
    print("\n[1] conditions (fixed thresholds; 'raw' = trainer decode, 'pp' = submission "
          "post-processing; 'resweep' = oracle tau/g_thr on this split, with pp)")
    print(f"{'condition':10}{'IoU raw':>9}{'IoU pp':>8}{'Cor pp':>8}{'empty':>7}"
          f"{'dirty':>7}{'clnOK':>7}{'dIoU(pp) vs correct [95% CI]':>30}"
          f"{'  resweep IoU (tau,g)':>22}")
    report["conditions"] = {}
    for c in conds:
        dec = decode_all(raws[c], items, tau, g_thr)
        per_raw, m_raw = score(gold, dec)
        pp = postproc(dec, texts, args.min_len, args.merge_gap)
        per, m_pp = score(gold, {i: r["labels"] for i, r in pp.items()})
        row = {"raw": m_raw, "pp": m_pp}
        ci = ""
        if c == "correct":
            per_ref = {"raw": per_raw, "pp": per}
        elif per_ref:
            for kind, per_k, m_k in (("raw", per_raw, m_raw), ("pp", per, m_pp)):
                d = m_k["IoU"] - report["conditions"]["correct"][kind]["IoU"]
                lo, hi = eval_official.cluster_ci(per_k, per_ref[kind], key="iou")
                dc = m_k["Cor"] - report["conditions"]["correct"][kind]["Cor"]
                clo, chi = eval_official.cluster_ci(per_k, per_ref[kind], key="cor")
                row[f"delta_vs_correct_{kind}"] = {"iou": d, "iou_ci95": [lo, hi],
                                                   "cor": dc, "cor_ci95": [clo, chi]}
            d, (lo, hi) = row["delta_vs_correct_pp"]["iou"], row["delta_vs_correct_pp"]["iou_ci95"]
            ci = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        rs = ""
        if not args.no_resweep:
            v, bt, bg = resweep(raws[c], items, gold, texts, args.min_len, args.merge_gap)
            row["resweep"] = {"IoU": v, "tau": bt, "g_thr": bg}
            rs = f"{v:.4f} ({bt},{bg})"
        report["conditions"][c] = row
        print(f"{c:10}{m_raw['IoU']:>9.4f}{m_pp['IoU']:>8.4f}{m_pp['Cor']:>8.4f}"
              f"{m_pp['empty']:>7.2f}{m_pp['dirty_iou']:>7.3f}{m_pp['clean_ok']:>7.2f}"
              f"{ci:>30}{rs:>22}")

    # 2. post-processing grid on `correct` ------------------------------------------
    if "correct" in raws:
        dec = decode_all(raws["correct"], items, tau, g_thr)
        ml_grid = [int(x) for x in args.min_len_grid.split(",")]
        mg_grid = [int(x) for x in args.merge_gap_grid.split(",")]
        print(f"\n[2] post-processing grid on correct (IoU / Cor), rows=min_len, cols=merge_gap")
        print(f"{'min_len':>8}" + "".join(f"{('gap=' + str(g)):>18}" for g in mg_grid))
        report["postproc_grid"] = {}
        for ml in ml_grid:
            line = f"{ml:>8}"
            for mg in mg_grid:
                pp = postproc(dec, texts, ml, mg)
                _, m = score(gold, {i: r["labels"] for i, r in pp.items()})
                report["postproc_grid"][f"min_len={ml},merge_gap={mg}"] = m
                line += f"{m['IoU']:>9.4f}/{m['Cor']:.4f}"
            print(line)

    # 3. threshold transfer ------------------------------------------------------------
    if args.transfer and "correct" in raws:
        t_tau, t_g, t_ml, t_mg = args.transfer.split(",")
        t_tau, t_g, t_ml, t_mg = float(t_tau), float(t_g), int(t_ml), int(t_mg)
        own = postproc(decode_all(raws["correct"], items, tau, g_thr), texts,
                       args.min_len, args.merge_gap)
        per_own, m_own = score(gold, {i: r["labels"] for i, r in own.items()})
        tr = postproc(decode_all(raws["correct"], items, t_tau, t_g), texts, t_ml, t_mg)
        per_tr, m_tr = score(gold, {i: r["labels"] for i, r in tr.items()})
        lo, hi = eval_official.cluster_ci(per_tr, per_own, key="iou")
        report["transfer"] = {"own": m_own, "transferred": m_tr,
                              "transfer_setting": [t_tau, t_g, t_ml, t_mg],
                              "delta_iou": {"delta": m_tr["IoU"] - m_own["IoU"], "ci95": [lo, hi]}}
        print(f"\n[3] threshold transfer: own (tau={tau}, g={g_thr}, min_len={args.min_len}, "
              f"gap={args.merge_gap}) vs fixed-elsewhere (tau={t_tau}, g={t_g}, "
              f"min_len={t_ml}, gap={t_mg})")
        print(f"{'own':14}IoU={m_own['IoU']:.4f} Cor={m_own['Cor']:.4f} empty={m_own['empty']:.2f}")
        print(f"{'transferred':14}IoU={m_tr['IoU']:.4f} Cor={m_tr['Cor']:.4f} empty={m_tr['empty']:.2f}"
              f"   dIoU={m_tr['IoU'] - m_own['IoU']:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
