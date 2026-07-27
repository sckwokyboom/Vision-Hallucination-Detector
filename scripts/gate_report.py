"""Full metric report for scored gate files (mm_gate_score.py output).

Per condition: greedy (argmax YES/NO) confusion + precision/recall/specificity/NPV/F1/
balAcc/MCC with Wilson 95% CIs (bootstrap CI for MCC) and Fisher exact p; plus threshold-free
ROC-AUC (bootstrap CI), PR-AUC, and specificity@recall {0.80, 0.90, 0.95}. Ends with a
compact comparison table. Pass several --pred name=path.
"""
import argparse
import json
import math
import random

from sklearn.metrics import roc_auc_score, average_precision_score


def load(path):
    g, s, am = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        g.append(1 if d["gold_dirty"] else 0)
        s.append(d["score"]); am.append(1 if d.get("argmax_yes") else 0)
    return g, s, am


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n; z = 1.96; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def conf(g, am):
    TP = sum(1 for gi, a in zip(g, am) if a and gi)
    FP = sum(1 for gi, a in zip(g, am) if a and not gi)
    FN = sum(1 for gi, a in zip(g, am) if not a and gi)
    TN = sum(1 for gi, a in zip(g, am) if not a and not gi)
    return TP, FP, FN, TN


def mcc(TP, FP, FN, TN):
    den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return (TP * TN - FP * FN) / den if den else 0.0


def fisher(TP, FP, FN, TN):
    try:
        from scipy.stats import fisher_exact
        return fisher_exact([[TP, FP], [FN, TN]])[1]
    except Exception:
        return float("nan")


def spec_at(g, s, mr):
    P = sum(g); N = len(g) - P; best = 0.0
    for thr in sorted(set(s), reverse=True):
        tp = sum(1 for gi, si in zip(g, s) if si >= thr and gi)
        fp = sum(1 for gi, si in zip(g, s) if si >= thr and not gi)
        if P and tp / P >= mr:
            best = max(best, (N - fp) / N if N else 0)
    return best


def boot_ci(fn, g, s, n=2000, seed=0):
    rng = random.Random(seed); vals = []
    idx = list(range(len(g)))
    for _ in range(n):
        b = [rng.choice(idx) for _ in idx]
        gg = [g[j] for j in b]
        if len(set(gg)) < 2:
            continue
        vals.append(fn(gg, [s[j] for j in b]))
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="name=path")
    args = ap.parse_args()
    summary = []
    for spec in args.pred:
        name, path = spec.split("=", 1)
        g, s, am = load(path)
        n = len(g); P = sum(g)
        TP, FP, FN, TN = conf(g, am)
        prec = wilson(TP, TP + FP); rec = wilson(TP, TP + FN)
        spc = wilson(TN, TN + FP); npv = wilson(TN, TN + FN)
        f1 = 2 * prec[0] * rec[0] / (prec[0] + rec[0]) if prec[0] + rec[0] else 0
        m = mcc(TP, FP, FN, TN)
        mlo, mhi = boot_ci(lambda gg, ss: mcc(*conf(gg, [1 if x >= 0 else 0 for x in ss])), g, s)
        roc = roc_auc_score(g, s); rlo, rhi = boot_ci(roc_auc_score, g, s)
        prauc = average_precision_score(g, s)
        print(f"\n================  {name}  (n={n}, dirty={P}, clean={n-P})  ================")
        print("  GREEDY (argmax YES/NO):")
        print(f"    confusion   TP={TP}  FP={FP}  FN={FN}  TN={TN}")
        print(f"    precision   {prec[0]:.3f}  [{prec[1]:.3f}, {prec[2]:.3f}]")
        print(f"    recall      {rec[0]:.3f}  [{rec[1]:.3f}, {rec[2]:.3f}]   (FNR={1-rec[0]:.3f})")
        print(f"    specificity {spc[0]:.3f}  [{spc[1]:.3f}, {spc[2]:.3f}]")
        print(f"    NPV         {npv[0]:.3f}  [{npv[1]:.3f}, {npv[2]:.3f}]   (NO branch {1-npv[0]:.0%} dirty)")
        print(f"    F1={f1:.3f}  balAcc={(rec[0]+spc[0])/2:.3f}  MCC={m:.3f} [{mlo:.3f}, {mhi:.3f}]  "
              f"Fisher p={fisher(TP,FP,FN,TN):.3f}")
        print("  THRESHOLD-FREE (score):")
        print(f"    ROC-AUC     {roc:.3f}  [{rlo:.3f}, {rhi:.3f}]")
        print(f"    PR-AUC      {prauc:.3f}   (prevalence baseline {P/n:.3f})")
        print(f"    spec@recall  .80={spec_at(g,s,.80):.3f}   .90={spec_at(g,s,.90):.3f}   "
              f".95={spec_at(g,s,.95):.3f}")
        summary.append((name, roc, prauc, m, spc[0], npv[0], spec_at(g, s, .80), spec_at(g, s, .90)))

    print("\n================  SUMMARY  ================")
    print(f"{'condition':22}{'ROC':>7}{'PR':>7}{'MCC':>7}{'spec':>7}{'NPV':>7}{'sp@.8':>7}{'sp@.9':>7}")
    for r in summary:
        print(f"{r[0]:22}{r[1]:>7.3f}{r[2]:>7.3f}{r[3]:>+7.3f}{r[4]:>7.3f}{r[5]:>7.3f}{r[6]:>7.3f}{r[7]:>7.3f}")


if __name__ == "__main__":
    main()
