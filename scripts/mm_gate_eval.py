"""Summary table for the multimodal few-shot gate predictions (mm_gate_gemma4.py output).

Reports: sample class balance, the binary gate (does it separate clean/dirty) with
precision/recall/F1/specificity/MCC vs always-YES/always-NO baselines, and the
per-type hallucination-category classification (multi-label P/R/F1 + macro/micro +
exact-set match) on the dirty items.
"""
import argparse
import json
import math

CATS = ["invention", "mischaracterization", "OCR", "miscounting", "other"]


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    rows = load(args.pred)
    n = len(rows)

    dirty = [r for r in rows if r["gold_dirty"]]
    clean = [r for r in rows if not r["gold_dirty"]]
    parse_err = sum(1 for r in rows if r["pred_dirty"] is None)

    print(f"=== Multimodal few-shot GATE — Gemma 4 12B — n={n} ===")
    print(f"class balance: dirty={len(dirty)} ({100*len(dirty)/n:.0f}%)  clean={len(clean)} "
          f"({100*len(clean)/n:.0f}%)   parse_errors={parse_err}")

    # ---- binary gate ----
    def binmetrics(pred_fn):
        TP = FP = FN = TN = 0
        for r in rows:
            p = pred_fn(r); g = r["gold_dirty"]
            if p and g: TP += 1
            elif p and not g: FP += 1
            elif not p and g: FN += 1
            else: TN += 1
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        spec = TN / (TN + FP) if TN + FP else 0.0
        den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = ((TP * TN - FP * FN) / den) if den else 0.0
        return TP, FP, FN, TN, prec, rec, f1, spec, (rec + spec) / 2, mcc

    print("\n--- BINARY GATE (does it separate clean vs dirty?) ---")
    hdr = f"{'system':22}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}{'prec':>7}{'rec':>7}{'F1':>7}{'spec':>7}{'balAcc':>8}{'MCC':>7}"
    print(hdr)
    def show(name, fn):
        TP, FP, FN, TN, p, r, f, s, b, m = binmetrics(fn)
        print(f"{name:22}{TP:>4}{FP:>4}{FN:>4}{TN:>4}{p:>7.3f}{r:>7.3f}{f:>7.3f}{s:>7.3f}{b:>8.3f}{m:>7.3f}")
    show("gemma4 gate", lambda r: bool(r["pred_dirty"]))
    show("baseline always-YES", lambda r: True)
    show("baseline always-NO", lambda r: False)

    # ---- type classification (multi-label) over gold-dirty items ----
    print("\n--- CATEGORY CLASSIFICATION (multi-label, on the 79 dirty items) ---")
    print(f"{'type':22}{'gold':>6}{'pred':>6}{'TP':>4}{'prec':>7}{'rec':>7}{'F1':>7}")
    f1s = []
    micro_tp = micro_fp = micro_fn = 0
    for c in CATS:
        g = sum(1 for r in dirty if c in r["gold_types"])
        pr = sum(1 for r in dirty if c in (r["pred_types"] or []))
        tp = sum(1 for r in dirty if c in r["gold_types"] and c in (r["pred_types"] or []))
        prec = tp / pr if pr else 0.0
        rec = tp / g if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        micro_tp += tp; micro_fp += (pr - tp); micro_fn += (g - tp)
        print(f"{c:22}{g:>6}{pr:>6}{tp:>4}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}")
    mprec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    mrec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    microf1 = 2 * mprec * mrec / (mprec + mrec) if (mprec + mrec) else 0.0
    exact = sum(1 for r in dirty if set(r["pred_types"] or []) == set(r["gold_types"])) / max(1, len(dirty))
    print(f"\n  macro-F1={sum(f1s)/len(f1s):.3f}   micro-F1={microf1:.3f}   "
          f"exact-set-match={exact:.3f}  (of dirty items)")

    # on truly-dirty items, how often gate at least fired (caught SOMETHING)
    caught = sum(1 for r in dirty if r["pred_dirty"]) / max(1, len(dirty))
    protected = sum(1 for r in clean if not r["pred_dirty"]) / max(1, len(clean))
    print(f"\n  dirty caught (recall of 'is there any hallucination'): {caught:.3f}")
    print(f"  clean protected (specificity):                          {protected:.3f}")


if __name__ == "__main__":
    main()
