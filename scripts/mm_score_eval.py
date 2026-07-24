"""Evaluate continuous-score gate files (mm_gate_score.py output).

Threshold-free ROC-AUC / PR-AUC on score=logP(YES)-logP(NO), plus the operating
metric specificity@recall>=target (sweep the threshold), and the greedy argmax point
for reference. Pass several --pred name=path to compare conditions/controls.
"""
import argparse
import json

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


def spec_at_recall(gold, score, min_rec):
    P = sum(gold); N = len(gold) - P
    best = 0.0
    for thr in sorted(set(score), reverse=True):
        tp = sum(1 for gi, si in zip(gold, score) if si >= thr and gi)
        fp = sum(1 for gi, si in zip(gold, score) if si >= thr and not gi)
        rec = tp / P if P else 0
        spec = (N - fp) / N if N else 0
        if rec >= min_rec and spec > best:
            best = spec
    return best


def greedy(gold, am):
    TP = sum(1 for g, a in zip(gold, am) if a and g)
    FP = sum(1 for g, a in zip(gold, am) if a and not g)
    FN = sum(1 for g, a in zip(gold, am) if not a and g)
    TN = sum(1 for g, a in zip(gold, am) if not a and not g)
    import math
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    spec = TN / (TN + FP) if TN + FP else 0
    den = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / den if den else 0
    return prec, rec, spec, mcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="name=path")
    args = ap.parse_args()
    print(f"{'condition':26}{'ROC-AUC':>9}{'PR-AUC':>8}{'sp@rec.9':>9}{'sp@rec.8':>9}"
          f"{'  |greedy prec/rec/spec/MCC'}")
    for spec in args.pred:
        name, path = spec.split("=", 1)
        g, s, am = load(path)
        roc = roc_auc_score(g, s) if len(set(g)) > 1 else float("nan")
        pr = average_precision_score(g, s) if sum(g) else float("nan")
        s9 = spec_at_recall(g, s, 0.90); s8 = spec_at_recall(g, s, 0.80)
        gp = greedy(g, am)
        print(f"{name:26}{roc:>9.3f}{pr:>8.3f}{s9:>9.3f}{s8:>9.3f}"
              f"   {gp[0]:.2f}/{gp[1]:.2f}/{gp[2]:.2f}/{gp[3]:+.2f}")
    print("\nnote: ROC-AUC 0.5 = no signal. A monotonic drop mm > shuffle_perm > textonly means the "
          "correct image contributes signal (visual); a textonly value >0.5 is the linguistic floor.")


if __name__ == "__main__":
    main()
