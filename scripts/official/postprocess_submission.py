"""Post-process SHROOM Vision prediction JSONL files.

The A2 BIO decoder can emit many tiny low-confidence islands, and older A2
checkpoints may serialize type labels from an untrained head. This script keeps
the raw prediction file intact and writes a cleaned official-format JSONL.

Examples:
  python scripts/official/postprocess_submission.py \
    --pred results/final/current_a2/tune_predictions.jsonl \
    --gold splits/dev.en.jsonl \
    --eval_ids splits/en.eval_protocol.json \
    --sweep

  python scripts/official/postprocess_submission.py \
    --pred results/final/submission/submission_a2_en.jsonl \
    --items ../Shroom-Vision/distrib/shroom-vision.test.en.unlabeled.jsonl \
    --out results/final/submission/submission_a2_en.cleaned.jsonl \
    --trim_edges --min_len 4 --min_prob 0.20 \
    --label_policy constant --default_label invention
"""
import argparse
import json
import math
import os
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

VALID_LABELS = {"invention", "mischaracterization", "OCR", "miscounting", "other"}
DEFAULT_LABEL = "invention"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_texts(path):
    if not path:
        return {}
    texts = {}
    for row in load_jsonl(path):
        if isinstance(row.get("response"), str):
            texts[row["id"]] = row["response"]
        elif isinstance(row.get("text_len"), int):
            texts[row["id"]] = " " * int(row["text_len"])
    return texts


def load_eval_ids(path):
    if not path:
        return None
    protocol = json.load(open(path, encoding="utf-8"))
    ids = protocol.get("tune_dev") or protocol.get("tune_dev200")
    return set(ids) if ids else None


def span_prob(span):
    try:
        value = float(span.get("prob", 1.0))
    except (TypeError, ValueError):
        value = 0.0
    return value if math.isfinite(value) else 0.0


def should_trim(ch, trim_symbols):
    if ch.isspace():
        return True
    category = unicodedata.category(ch)
    if category.startswith("P") or category.startswith("Z"):
        return True
    return trim_symbols and category.startswith("S")


def clean_one_span(span, text, *, trim_edges, trim_symbols, min_len, min_prob,
                   label_policy, default_label):
    try:
        start = int(span.get("start", 0))
        end = int(span.get("end", 0))
    except (TypeError, ValueError):
        return None

    if text is not None:
        n = len(text)
        start = max(0, min(start, n))
        end = max(0, min(end, n))
    if start >= end:
        return None

    prob = span_prob(span)
    label = str(span.get("label", "other"))
    if label_policy == "constant":
        label = default_label
    elif label not in VALID_LABELS:
        label = "other"

    if trim_edges and text is not None:
        while start < end and should_trim(text[start], trim_symbols):
            start += 1
        while start < end and should_trim(text[end - 1], trim_symbols):
            end -= 1

    if start >= end:
        return None
    if end - start < min_len:
        return None
    if prob < min_prob:
        return None
    return {"start": start, "end": end, "prob": float(prob), "label": label}


def merge_spans(spans, merge_gap):
    if merge_gap < 0:
        return spans
    out = []
    for span in sorted(spans, key=lambda sp: (sp["start"], sp["end"], sp["label"])):
        if (out and out[-1]["label"] == span["label"]
                and span["start"] <= out[-1]["end"] + merge_gap):
            left_len = out[-1]["end"] - out[-1]["start"]
            right_len = span["end"] - span["start"]
            total = max(1, left_len + right_len)
            out[-1]["end"] = max(out[-1]["end"], span["end"])
            out[-1]["prob"] = float((out[-1]["prob"] * left_len + span["prob"] * right_len) / total)
        else:
            out.append(dict(span))
    return out


def postprocess_rows(rows, texts, *, trim_edges, trim_symbols, min_len, min_prob,
                     label_policy, default_label, merge_gap, keep_ids=None):
    out = []
    for row in rows:
        item_id = row["id"]
        if keep_ids is not None and item_id not in keep_ids:
            continue
        text = texts.get(item_id)
        labels = row.get("labels", row.get("pred_labels")) or []
        spans = []
        for span in labels:
            cleaned = clean_one_span(
                span,
                text,
                trim_edges=trim_edges,
                trim_symbols=trim_symbols,
                min_len=min_len,
                min_prob=min_prob,
                label_policy=label_policy,
                default_label=default_label,
            )
            if cleaned:
                spans.append(cleaned)
        out.append({"id": item_id, "labels": merge_spans(spans, merge_gap)})
    return out


def summarize(rows):
    n_items = len(rows)
    n_nonempty = sum(1 for row in rows if row["labels"])
    labels = Counter()
    lengths = []
    probs = []
    for row in rows:
        for span in row["labels"]:
            labels[span["label"]] += 1
            lengths.append(span["end"] - span["start"])
            probs.append(float(span.get("prob", 1.0)))
    span_count = len(lengths)
    return {
        "items": n_items,
        "nonempty": n_nonempty,
        "empty": n_items - n_nonempty,
        "spans": span_count,
        "mean_spans_nonempty": span_count / max(1, n_nonempty),
        "mean_len": statistics.fmean(lengths) if lengths else 0.0,
        "median_len": statistics.median(lengths) if lengths else 0.0,
        "mean_prob": statistics.fmean(probs) if probs else 0.0,
        "labels": dict(labels),
    }


def print_summary(title, stats):
    print(f"{title}:")
    print(
        f"  items={stats['items']} nonempty={stats['nonempty']} empty={stats['empty']} "
        f"spans={stats['spans']} mean_spans_nonempty={stats['mean_spans_nonempty']:.2f}"
    )
    print(
        f"  mean_len={stats['mean_len']:.2f} median_len={stats['median_len']:.1f} "
        f"mean_prob={stats['mean_prob']:.3f}"
    )
    print(f"  labels={stats['labels']}")


def write_jsonl(rows, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_float_grid(value):
    return [float(part) for part in value.split(",") if part.strip()]


def parse_int_grid(value):
    return [int(part) for part in value.split(",") if part.strip()]


def parse_policy_grid(value):
    policies = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw == "keep":
            policies.append(("keep", DEFAULT_LABEL))
        elif raw.startswith("constant:"):
            label = raw.split(":", 1)[1]
            if label not in VALID_LABELS:
                raise SystemExit(f"invalid label in --label_policy_grid: {label}")
            policies.append(("constant", label))
        else:
            raise SystemExit(f"invalid --label_policy_grid entry: {raw}")
    return policies


def evaluate_rows(gold_path, eval_ids_path, rows):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_official  # noqa: WPS433

    gold = eval_official.load_gold(gold_path, load_eval_ids(eval_ids_path))
    pred = {row["id"]: row for row in rows}
    per_item = eval_official.per_item(gold, pred)
    return eval_official.agg(per_item)


def run_sweep(args, pred_rows, texts):
    if not args.gold:
        raise SystemExit("--sweep requires --gold")
    keep_ids = load_eval_ids(args.eval_ids)
    candidates = []
    for min_len in parse_int_grid(args.min_len_grid):
        for min_prob in parse_float_grid(args.min_prob_grid):
            for label_policy, default_label in parse_policy_grid(args.label_policy_grid):
                rows = postprocess_rows(
                    pred_rows,
                    texts,
                    trim_edges=args.trim_edges,
                    trim_symbols=args.trim_symbols,
                    min_len=min_len,
                    min_prob=min_prob,
                    label_policy=label_policy,
                    default_label=default_label,
                    merge_gap=args.merge_gap,
                    keep_ids=keep_ids,
                )
                metrics = evaluate_rows(args.gold, args.eval_ids, rows)
                candidates.append({
                    "min_len": min_len,
                    "min_prob": min_prob,
                    "label_policy": label_policy,
                    "default_label": default_label,
                    "IoU": metrics["IoU"],
                    "Cor": metrics["Cor"],
                    "Cor_lbl": metrics["Cor_lbl"],
                    "rows": rows,
                    "summary": summarize(rows),
                })

    candidates.sort(key=lambda row: (row["IoU"], row["Cor"], row["Cor_lbl"]), reverse=True)
    print("Top postprocess settings by tune IoU:")
    print(f"{'rank':>4} {'min_len':>7} {'min_prob':>8} {'label':>20} {'IoU':>8} {'Cor':>8} {'Cor_lbl':>8} {'spans':>7} {'nonempty':>8}")
    for rank, row in enumerate(candidates[:args.top], 1):
        label = row["label_policy"] if row["label_policy"] == "keep" else f"constant:{row['default_label']}"
        print(
            f"{rank:>4} {row['min_len']:>7} {row['min_prob']:>8.2f} {label:>20} "
            f"{row['IoU']:>8.4f} {row['Cor']:>8.4f} {row['Cor_lbl']:>8.4f} "
            f"{row['summary']['spans']:>7} {row['summary']['nonempty']:>8}"
        )
    if args.out:
        best = candidates[0]
        best_rows = postprocess_rows(
            pred_rows,
            texts,
            trim_edges=args.trim_edges,
            trim_symbols=args.trim_symbols,
            min_len=best["min_len"],
            min_prob=best["min_prob"],
            label_policy=best["label_policy"],
            default_label=best["default_label"],
            merge_gap=args.merge_gap,
            keep_ids=None,
        )
        write_jsonl(best_rows, args.out)
        label = best["label_policy"] if best["label_policy"] == "keep" else f"constant:{best['default_label']}"
        print(
            f"\nwrote best sweep output -> {args.out} "
            f"(min_len={best['min_len']} min_prob={best['min_prob']:.2f} label={label})"
        )


def main():
    parser = argparse.ArgumentParser(description="Clean SHROOM Vision prediction spans.")
    parser.add_argument("--pred", required=True, help="Input prediction JSONL with labels or pred_labels.")
    parser.add_argument("--out", default=None, help="Output official-format JSONL.")
    parser.add_argument("--items", default=None, help="JSONL with id and response for test/dev cleanup.")
    parser.add_argument("--gold", default=None, help="Gold JSONL; also used as item text source.")
    parser.add_argument("--eval_ids", default=None, help="Eval protocol JSON for tune subset.")
    parser.add_argument("--trim_edges", action="store_true", help="Trim whitespace and punctuation at span edges.")
    parser.add_argument("--trim_symbols", action="store_true", help="Also trim Unicode symbol characters at span edges.")
    parser.add_argument("--min_len", type=int, default=1, help="Minimum span length after trimming.")
    parser.add_argument("--min_prob", type=float, default=0.0, help="Minimum span probability.")
    parser.add_argument("--merge_gap", type=int, default=0, help="Merge same-label spans separated by at most this many chars; -1 disables.")
    parser.add_argument("--label_policy", choices=["keep", "constant"], default="keep")
    parser.add_argument("--default_label", choices=sorted(VALID_LABELS), default=DEFAULT_LABEL)
    parser.add_argument("--sweep", action="store_true", help="Grid-search cleanup settings on --gold.")
    parser.add_argument("--min_len_grid", default="1,2,3,4,5,8")
    parser.add_argument("--min_prob_grid", default="0,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50")
    parser.add_argument("--label_policy_grid", default="keep,constant:invention,constant:other")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if args.min_len < 1:
        raise SystemExit("--min_len must be >= 1")
    if args.min_prob < 0.0:
        raise SystemExit("--min_prob must be >= 0")

    pred_rows = load_jsonl(args.pred)
    texts = {}
    texts.update(load_texts(args.gold))
    texts.update(load_texts(args.items))

    raw_rows = postprocess_rows(
        pred_rows,
        texts,
        trim_edges=False,
        trim_symbols=False,
        min_len=1,
        min_prob=0.0,
        label_policy="keep",
        default_label=args.default_label,
        merge_gap=-1,
    )
    print_summary("Raw", summarize(raw_rows))

    if args.sweep:
        run_sweep(args, pred_rows, texts)
        return

    if not args.out:
        raise SystemExit("--out is required unless --sweep is used")

    rows = postprocess_rows(
        pred_rows,
        texts,
        trim_edges=args.trim_edges,
        trim_symbols=args.trim_symbols,
        min_len=args.min_len,
        min_prob=args.min_prob,
        label_policy=args.label_policy,
        default_label=args.default_label,
        merge_gap=args.merge_gap,
    )
    print_summary("Cleaned", summarize(rows))
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
