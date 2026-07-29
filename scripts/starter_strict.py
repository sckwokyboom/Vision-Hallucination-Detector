"""Re-derive the starter baseline's spans with the starter's OWN alignment.

`mlx_predict.py --prompt_style original` reproduces the starter prompt, but the spans
go through our forgiving aligner (exact -> normalized -> fuzzy). The organizers' script
(`label_with_gemma.py: phrases_to_spans`) uses a plain `response.find(phrase)` and keeps
only the FIRST occurrence. This rewrites a --save_raw prediction file with that exact
rule, so the table can show both (strict = faithful starter, fuzzy = generous upper bound).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shroom.predict import parse_output          # noqa: E402  (same JSON tolerance)


def phrases_to_spans_starter(phrases, response):
    """Verbatim port of label_with_gemma.phrases_to_spans."""
    spans = []
    for entry in phrases:
        if not isinstance(entry, dict):
            continue
        phrase = entry.get("phrase", "")
        idx = response.find(phrase)
        if idx == -1 or not phrase:
            continue
        spans.append({"start": idx, "end": idx + len(phrase),
                      "label": entry.get("label", "unknown")})
    spans.sort(key=lambda s: s["start"])
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction jsonl written with --save_raw")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n, n_parse_err, n_nonempty = 0, 0, 0
    with open(args.pred, encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            resp = rec["response"]
            raws = rec.get("raw_samples") or []
            n += 1
            hits = [0] * len(resp)
            valid = 0
            merged = []
            for raw in raws:
                parsed = parse_output(raw)
                if parsed is None:
                    n_parse_err += 1
                    continue
                valid += 1
                spans = phrases_to_spans_starter(parsed, resp)
                merged.extend(spans)
                for sp in spans:
                    for i in range(sp["start"], min(sp["end"], len(resp))):
                        hits[i] = 1                       # per-sample coverage
            denom = valid or 1
            # with n_samples=1 this is 0/1; kept general for self-consistency runs
            probs = [h / denom for h in hits]
            for sp in merged:
                sp["prob"] = 1.0 / denom
            if merged:
                n_nonempty += 1
            out.write(json.dumps({"id": rec["id"], "language": rec.get("language"),
                                  "response": resp, "pred_labels": merged,
                                  "char_probs": probs}, ensure_ascii=False) + "\n")
    print(f"[strict] {n} items -> {args.output}  (unparseable outputs: {n_parse_err}, "
          f"items with >=1 span: {n_nonempty})")


if __name__ == "__main__":
    main()
