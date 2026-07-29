"""One scorer, one eval set, one table.

Every system (starter baseline, zero-shot prompting, trained readouts) is re-scored here
from its submission JSONL, so the numbers in the output table are directly comparable.

Two span_iou columns, because they answer different questions:
  span_iou      — char-IoU of the spans the system would actually SUBMIT (`pred_labels`).
                  This is the official metric. Systems whose decoder tuned tau/gate on this
                  same set are marked `tuned-here` in the notes; treat them as optimistic.
  span_iou@thr  — char-IoU at the best single threshold swept over `char_probs`
                  (same sweep as scripts/official_metrics.py). Decoder-independent, so it
                  compares the underlying per-character signal rather than the decode rule.

Threshold-free metrics (roc_auc / pr_auc / cor) all read `char_probs`:
  cor           — Pearson(gold per-char prob, predicted per-char prob), raw probabilities.
  cor_sub       — same, but probabilities are zeroed on items the system decoded to []
                  (what a gated system actually submits).

Also reports the paired bootstrap CI of mean per-item IoU against predict-nothing —
the only honest way to say "beats the floor" on ~200 items.

Usage:
  python scripts/unified_table.py --gold splits/tune202.en.jsonl \
      --pred "name=path.jsonl" [--pred ...] --out results/FINAL_TABLE.md
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl                                  # noqa: E402
from shroom.predict import CATEGORIES                               # noqa: E402
from shroom.metrics import _char_set, gold_char_probs, pearson, char_iou   # noqa: E402

BOOT = 2000
SEED = 99


def load_by_id(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["id"]] = r
    return d


def safe_roc(y, s):
    y = np.asarray(y)
    return float("nan") if y.min() == y.max() else float(roc_auc_score(y, s))


def cat_prob_array(spans, rl):
    arr = {c: np.zeros(rl) for c in CATEGORIES}
    for s in spans:
        c = s.get("label", "other")
        c = c if c in arr else "other"
        a = max(0, min(int(s["start"]), rl))
        b = max(0, min(int(s["end"]), rl))
        arr[c][a:b] = np.maximum(arr[c][a:b], float(s.get("prob", 1.0)))
    return arr


def score(gold, get_pred):
    gbin, pprob, gsoft, psub = [], [], [], []
    cg = {c: [] for c in CATEGORIES}
    cp = {c: [] for c in CATEGORIES}
    per_item_iou, per_item_thr = [], []
    n_spans, n_empty = 0, 0

    for i, it in gold.items():
        rl = len(it.response)
        spans, probs = get_pred(i)
        probs = np.asarray((list(probs) + [0.0] * rl)[:rl], dtype=float)
        n_spans += len(spans)
        n_empty += 0 if spans else 1

        per_item_iou.append(char_iou(it.labels, spans, rl))
        per_item_thr.append((_char_set(it.labels, rl), rl, probs))

        gset = _char_set(it.labels, rl)
        gb = np.zeros(rl)
        for k in gset:
            gb[k] = 1.0
        gbin.extend(gb.tolist())
        pprob.extend(probs.tolist())
        gsoft.extend(gold_char_probs(it.labels, rl))
        psub.extend((probs if spans else np.zeros(rl)).tolist())

        pcat = cat_prob_array(spans, rl)
        for c in CATEGORIES:
            gc = np.zeros(rl)
            for k in _char_set([s for s in it.labels if s.get("label") == c], rl):
                gc[k] = 1.0
            cg[c].extend(gc.tolist())
            cp[c].extend(pcat[c].tolist())

    gbin = np.asarray(gbin)
    pprob = np.asarray(pprob)
    best_iou, best_thr = -1.0, 0.0
    for thr in np.linspace(0.05, 1.0, 20):
        ious = [1.0 if not g and not (p >= thr).any() else
                (0.0 if (not g or not (p >= thr).any()) else
                 len(g & set(np.nonzero(p >= thr)[0].tolist())) /
                 len(g | set(np.nonzero(p >= thr)[0].tolist())))
                for g, rl, p in per_item_thr]
        m = float(np.mean(ious))
        if m > best_iou:
            best_iou, best_thr = m, float(thr)

    class_roc = {c: safe_roc(cg[c], cp[c]) for c in CATEGORIES}
    return dict(
        span_iou=float(np.mean(per_item_iou)),
        span_iou_thr=best_iou, best_thr=best_thr,
        roc_auc=safe_roc(gbin, pprob),
        pr_auc=float(average_precision_score(gbin, pprob)) if gbin.max() > 0 else float("nan"),
        cor=pearson(gsoft, pprob), cor_sub=pearson(gsoft, psub),
        class_macro=float(np.nanmean(list(class_roc.values()))),
        class_roc=class_roc,
        avg_spans=n_spans / max(1, len(gold)),
        pct_empty=n_empty / max(1, len(gold)),
        _ious=per_item_iou,
    )


def boot_ci(a, b, n=BOOT, seed=SEED):
    """Paired bootstrap over items of mean(a) - mean(b)."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--eval_ids", default=None,
                    help="json file with {'tune_dev': [...]} — filters --gold down to that "
                         "split. Use this instead of shipping a materialised gold subset: "
                         "--gold splits/dev.en.jsonl --eval_ids splits/en.eval_protocol.json")
    ap.add_argument("--pred", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--note", action="append", default=[], help="name=note (repeatable)")
    ap.add_argument("--out", default=None, help="write a markdown table here")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    gold = {it.id: it for it in load_jsonl(args.gold)}
    if args.eval_ids:
        prot = json.load(open(args.eval_ids))
        keep = set(prot.get("tune_dev") or prot.get("tune_dev200") or [])
        if not keep:
            raise SystemExit(f"{args.eval_ids}: no 'tune_dev' key")
        missing = keep - set(gold)
        if missing:
            raise SystemExit(f"--gold {args.gold} is missing {len(missing)} of the "
                             f"{len(keep)} eval ids (e.g. {sorted(missing)[:3]})")
        gold = {i: it for i, it in gold.items() if i in keep}
    notes = dict(s.split("=", 1) for s in args.note)

    systems = [
        ("predict-nothing", lambda i: ([], [0.0] * len(gold[i].response))),
        ("predict-everything",
         lambda i: ([{"start": 0, "end": len(gold[i].response), "label": "other"}],
                    [1.0] * len(gold[i].response))),
    ]
    for spec in args.pred:
        name, path = spec.rsplit("=", 1)     # a system name may contain '='; a path may not
        d = load_by_id(path)
        missing = set(gold) - set(d)
        if missing:
            raise SystemExit(f"{name}: {len(missing)} gold ids missing from {path} "
                             f"(e.g. {sorted(missing)[:3]}) — not the same eval set")

        def mk(dd):
            return lambda i: (dd[i].get("pred_labels") or [],
                              dd[i].get("char_probs") or [0.0] * len(gold[i].response))
        systems.append((name, mk(d)))

    res = [(n, score(gold, f)) for n, f in systems]
    floor = dict(res)["predict-nothing"]
    for _, m in res:
        m["d_floor"], m["ci_lo"], m["ci_hi"] = boot_ci(m["_ious"], floor["_ious"])

    hdr = (f"{'system':34}{'span_iou':>9}{'vs floor (95% CI)':>26}{'@thr':>7}{'thr':>6}"
           f"{'roc':>7}{'pr':>7}{'cor':>7}{'corS':>7}{'clsM':>7}{'spans':>7}{'%[]':>6}")
    lines = [f"n = {len(gold)} items   gold file = {args.gold}", "", hdr, "-" * len(hdr)]
    for name, m in res:
        ci = f"{m['d_floor']:+.3f} [{m['ci_lo']:+.3f},{m['ci_hi']:+.3f}]"
        lines.append(f"{name:34}{m['span_iou']:>9.4f}{ci:>26}{m['span_iou_thr']:>7.3f}"
                     f"{m['best_thr']:>6.2f}{m['roc_auc']:>7.3f}{m['pr_auc']:>7.3f}"
                     f"{m['cor']:>7.3f}{m['cor_sub']:>7.3f}{m['class_macro']:>7.3f}"
                     f"{m['avg_spans']:>7.2f}{m['pct_empty']*100:>6.0f}")
    print("\n".join(lines))

    if args.out:
        md = [f"n = **{len(gold)}** items (`{args.gold}`), predict-nothing floor = "
              f"**{floor['span_iou']:.4f}**", "",
              "| system | span_iou | Δ vs floor (95% CI) | span_iou@thr | thr | roc_auc | "
              "pr_auc | cor | cor_sub | class_macro | spans/item | % empty | notes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, m in res:
            sig = "**" if m["ci_lo"] > 0 else ""      # bold = significantly ABOVE the floor
            md.append(
                f"| {name} | {sig}{m['span_iou']:.4f}{sig} | "
                f"{m['d_floor']:+.3f} [{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}] | "
                f"{m['span_iou_thr']:.3f} | {m['best_thr']:.2f} | {m['roc_auc']:.3f} | "
                f"{m['pr_auc']:.3f} | {m['cor']:.3f} | {m['cor_sub']:.3f} | "
                f"{m['class_macro']:.3f} | {m['avg_spans']:.2f} | {m['pct_empty']*100:.0f}% | "
                f"{notes.get(name, '')} |")
        md += ["", "Per-category one-vs-rest ROC AUC", "",
               "| system | " + " | ".join(CATEGORIES) + " | macro |",
               "|---" * (len(CATEGORIES) + 2) + "|"]
        for name, m in res:
            md.append(f"| {name} | " + " | ".join(f"{m['class_roc'][c]:.3f}" for c in CATEGORIES)
                      + f" | {m['class_macro']:.3f} |")
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        print(f"\n[md] -> {args.out}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({n: {k: v for k, v in m.items() if not k.startswith("_")}
                       for n, m in res}, f, indent=2)
        print(f"[json] -> {args.json_out}")


if __name__ == "__main__":
    main()
