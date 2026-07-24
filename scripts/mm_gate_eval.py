"""Honest summary for the multimodal-gate predictions (mm_gate_gemma4.py output).

Binary gate with Wilson 95% CIs, NPV/FNR, Fisher exact p and a bootstrap MCC CI;
per-type category metrics computed over ALL items (a type predicted on a CLEAN item
is a false positive). Handles no-classify variants (no pred_types) gracefully.
"""
import argparse
import json
import math
import random

CATS = ["invention", "mischaracterization", "OCR", "miscounting", "other"]


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fisher_p(TP, FP, FN, TN):
    try:
        from scipy.stats import fisher_exact
        return fisher_exact([[TP, FP], [FN, TN]])[1]
    except Exception:
        return float("nan")


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def binmetrics(rows, pred_fn):
    TP = FP = FN = TN = 0
    for r in rows:
        p = pred_fn(r); g = r["gold_dirty"]
        if p and g: TP += 1
        elif p and not g: FP += 1
        elif not p and g: FN += 1
        else: TN += 1
    return TP, FP, FN, TN


def mcc(TP, FP, FN, TN):
    den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return ((TP * TN - FP * FN) / den) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", default="gate")
    args = ap.parse_args()
    rows = load(args.pred)
    n = len(rows)
    dirty = [r for r in rows if r["gold_dirty"]]
    clean = [r for r in rows if not r["gold_dirty"]]
    parse_err = sum(1 for r in rows if r["pred_dirty"] is None)
    print(f"=== {args.name} — n={n}  dirty={len(dirty)} clean={len(clean)}  parse_err={parse_err} ===")

    TP, FP, FN, TN = binmetrics(rows, lambda r: bool(r["pred_dirty"]))
    prec, pl, ph = wilson(TP, TP + FP)
    rec, rl, rh = wilson(TP, TP + FN)
    spec, sl, sh = wilson(TN, TN + FP)
    npv, nl, nh = wilson(TN, TN + FN)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    m = mcc(TP, FP, FN, TN)
    # bootstrap MCC CI
    random.seed(0)
    boots = []
    for _ in range(2000):
        s = [random.choice(rows) for _ in rows]
        a, b, c, d = binmetrics(s, lambda r: bool(r["pred_dirty"]))
        boots.append(mcc(a, b, c, d))
    boots.sort(); mlo, mhi = boots[50], boots[1949]
    print(f"  confusion: TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"  precision(YES) = {prec:.3f}  [{pl:.3f},{ph:.3f}]   (base rate dirty={len(dirty)/n:.3f})")
    print(f"  recall/sens    = {rec:.3f}  [{rl:.3f},{rh:.3f}]   FNR={1-rec:.3f}")
    print(f"  specificity    = {spec:.3f}  [{sl:.3f},{sh:.3f}]")
    print(f"  NPV (NO clean) = {npv:.3f}  [{nl:.3f},{nh:.3f}]   -> NO branch is {1-npv:.0%} dirty")
    print(f"  F1={f1:.3f}  balAcc={(rec+spec)/2:.3f}  MCC={m:.3f} [{mlo:.3f},{mhi:.3f}]  Fisher p={fisher_p(TP,FP,FN,TN):.3f}")

    if any(r.get("pred_types") for r in rows):
        print("\n  CATEGORY (per-type over ALL items; a type on a clean item = FP):")
        print(f"  {'type':20}{'gold':>5}{'pred':>5}{'TP':>4}{'prec':>7}{'rec':>7}{'F1':>7}")
        f1s = []; mtp = mfp = mfn = 0
        for c in CATS:
            g = sum(1 for r in rows if c in r["gold_types"])
            pr = sum(1 for r in rows if c in (r["pred_types"] or []))
            tp = sum(1 for r in rows if c in r["gold_types"] and c in (r["pred_types"] or []))
            p2 = tp / pr if pr else 0.0; r2 = tp / g if g else 0.0
            ff = 2 * p2 * r2 / (p2 + r2) if p2 + r2 else 0.0
            f1s.append(ff); mtp += tp; mfp += pr - tp; mfn += g - tp
            print(f"  {c:20}{g:>5}{pr:>5}{tp:>4}{p2:>7.3f}{r2:>7.3f}{ff:>7.3f}")
        mp = mtp / (mtp + mfp) if mtp + mfp else 0; mr = mtp / (mtp + mfn) if mtp + mfn else 0
        micro = 2 * mp * mr / (mp + mr) if mp + mr else 0
        exact = sum(1 for r in rows if set(r["pred_types"] or []) == set(r["gold_types"])) / n
        print(f"  macro-F1={sum(f1s)/len(f1s):.3f}  micro-F1={micro:.3f}  exact-set-match(all {n})={exact:.3f}")


if __name__ == "__main__":
    main()
