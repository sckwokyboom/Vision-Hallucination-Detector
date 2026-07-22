"""Diagnose the 'locator' (extraction) pass on items that actually contain
hallucinations (gold non-empty). Reports character-level recall/precision and
span-level detection recall (did it overlap each gold span at all), plus a
per-category and per-span-length breakdown so we can see WHERE it fails.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.data import load_jsonl                     # noqa: E402
from shroom.metrics import _char_set                   # noqa: E402
from shroom.predict import CATEGORIES                  # noqa: E402


def load_by_id(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line); d[r["id"]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--all", action="store_true", help="include clean items too (default: dirty only)")
    args = ap.parse_args()

    gold = {it.id: it for it in load_jsonl(args.gold)}
    pred = load_by_id(args.pred)
    ids = [i for i in gold if i in pred and (args.all or gold[i].labels)]

    tot_g = tot_p = tot_i = 0
    fired = 0
    span_total = span_hit = 0
    missed = partial = full = 0
    cat_g = defaultdict(int); cat_hit = defaultdict(int)
    len_buckets = {"1-5": [0, 0], "6-20": [0, 0], "21-60": [0, 0], "60+": [0, 0]}  # [hit, total]

    for i in ids:
        it = gold[i]; rl = len(it.response)
        pchars = _char_set(pred[i].get("pred_labels", []), rl)
        gchars = _char_set(it.labels, rl)
        if pchars:
            fired += 1
        tot_g += len(gchars); tot_p += len(pchars); tot_i += len(gchars & pchars)
        for sp in it.labels:
            a = max(0, min(int(sp["start"]), rl)); b = max(0, min(int(sp["end"]), rl))
            schars = set(range(a, b))
            if not schars:
                continue
            ov = len(schars & pchars)
            span_total += 1
            if ov == 0:
                missed += 1
            elif ov == len(schars):
                full += 1; span_hit += 1
            else:
                partial += 1; span_hit += 1
            c = sp.get("label", "other")
            cat_g[c] += len(schars); cat_hit[c] += ov
            L = len(schars)
            key = "1-5" if L <= 5 else "6-20" if L <= 20 else "21-60" if L <= 60 else "60+"
            len_buckets[key][1] += 1
            len_buckets[key][0] += (1 if ov > 0 else 0)

    n = len(ids)
    print(f"locator on {'ALL' if args.all else 'DIRTY'} items: n={n}")
    print(f"  item fire-rate:           {fired}/{n} = {fired/n:.2f}")
    print(f"  char-level recall:        {tot_i/tot_g:.3f}   (of gold hallucination chars caught)")
    print(f"  char-level precision:     {tot_i/tot_p if tot_p else 0:.3f}   (of flagged chars that are correct)")
    print(f"  SPAN detection recall:    {span_hit}/{span_total} = {span_hit/span_total:.3f}   "
          f"(gold spans touched at all)")
    print(f"  gold-span outcome:        missed={missed}  partial={partial}  full={full}")
    print(f"\n  per-category char recall (detection of that category's chars):")
    for c in CATEGORIES:
        if cat_g[c]:
            print(f"    {c:20} {cat_hit[c]/cat_g[c]:.3f}   (gold chars={cat_g[c]})")
    print(f"\n  span detection recall by gold-span length:")
    for k, (h, t) in len_buckets.items():
        if t:
            print(f"    len {k:6} {h}/{t} = {h/t:.3f}")


if __name__ == "__main__":
    main()
