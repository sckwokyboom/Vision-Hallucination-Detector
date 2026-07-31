"""Build an HTML viewer for SHROOM Vision prediction files.

Works for both unlabeled test metadata and labeled dev/gold files. The prediction
file may be official-format (`labels`) or internal-format (`pred_labels`).

Example:
  python scripts/official/inspect_submission.py \
    --items ../Shroom-Vision/distrib/shroom-vision.test.en.unlabeled.jsonl \
    --pred results/final/submission/submission_a2_en.jsonl \
    --image_dir ../Shroom-Vision/images \
    --out results/final/submission/inspect_a2_en.html
"""
import argparse
import html
import json
import math
import os
from pathlib import Path

VALID_LABELS = ["invention", "mischaracterization", "OCR", "miscounting", "other"]
COLORS = {
    "invention": "#d92d20",
    "mischaracterization": "#e66f00",
    "OCR": "#7a3db8",
    "miscounting": "#0969da",
    "other": "#6e7781",
}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def span_key(span):
    return int(span.get("start", 0)), int(span.get("end", 0)), span.get("label", "other")


def clean_spans(spans, n):
    out = []
    for sp in spans or []:
        a = max(0, min(int(sp.get("start", 0)), n))
        b = max(0, min(int(sp.get("end", 0)), n))
        if not a < b:
            continue
        p = float(sp.get("prob", 1.0))
        if not math.isfinite(p):
            p = 0.0
        label = str(sp.get("label", "other"))
        if label not in VALID_LABELS:
            label = "other"
        out.append({"start": a, "end": b, "prob": p, "label": label})
    return sorted(out, key=span_key)


def char_arrays(spans, n):
    prob = [0.0] * n
    label = [None] * n
    for sp in spans:
        a, b = int(sp["start"]), int(sp["end"])
        p = float(sp.get("prob", 1.0))
        for i in range(a, b):
            if p >= prob[i]:
                prob[i] = p
                label[i] = sp["label"]
    return prob, label


def render_text(text, spans):
    prob, label = char_arrays(spans, len(text))
    chunks = []
    i = 0
    while i < len(text):
        j = i + 1
        while j < len(text) and label[j] == label[i] and abs(prob[j] - prob[i]) < 1e-9:
            j += 1
        seg = html.escape(text[i:j])
        if label[i] is None:
            chunks.append(seg)
        else:
            color = COLORS[label[i]]
            alpha = 0.22 + 0.58 * max(0.0, min(1.0, prob[i]))
            chunks.append(
                f'<mark style="--c:{color};--a:{alpha:.3f}" '
                f'title="{html.escape(label[i])} p={prob[i]:.3f}">{seg}</mark>'
            )
        i = j
    return "".join(chunks).replace("\n", "<br>")


def img_src(path):
    return Path(path).resolve().as_uri()


def item_stats(spans, n):
    covered = set()
    by_label = {label: 0 for label in VALID_LABELS}
    for sp in spans:
        by_label[sp["label"]] += 1
        covered.update(range(sp["start"], sp["end"]))
    mean_prob = sum(float(sp["prob"]) for sp in spans) / max(1, len(spans))
    return {
        "n_spans": len(spans),
        "covered_chars": len(covered),
        "coverage": len(covered) / max(1, n),
        "mean_prob": mean_prob,
        "by_label": by_label,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="test/dev JSONL with id, prompt, response, image_name")
    ap.add_argument("--pred", required=True, help="official labels or internal pred_labels JSONL")
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="SHROOM Vision submission inspection")
    ap.add_argument("--limit", type=int, default=None, help="optional first-N cap for a smaller HTML")
    args = ap.parse_args()

    items = {row["id"]: row for row in load_jsonl(args.items)}
    pred_rows = load_jsonl(args.pred)
    rows = []
    totals = {
        "items": 0,
        "empty": 0,
        "spans": 0,
        "covered_chars": 0,
        "chars": 0,
        "labels": {label: 0 for label in VALID_LABELS},
    }
    missing_meta = []
    for pr in pred_rows:
        it = items.get(pr["id"])
        if it is None:
            missing_meta.append(pr["id"])
            continue
        response = it.get("response", "")
        spans = clean_spans(pr.get("labels", pr.get("pred_labels")) or [], len(response))
        st = item_stats(spans, len(response))
        totals["items"] += 1
        totals["empty"] += (1 if not spans else 0)
        totals["spans"] += st["n_spans"]
        totals["covered_chars"] += st["covered_chars"]
        totals["chars"] += len(response)
        for label, count in st["by_label"].items():
            totals["labels"][label] += count
        rows.append((it, spans, st))
    if args.limit:
        rows = rows[:args.limit]

    cards = []
    for it, spans, st in rows:
        image_path = os.path.join(args.image_dir, it.get("image_name", ""))
        image_html = (
            f'<img src="{html.escape(img_src(image_path))}" loading="lazy" alt="">'
            if os.path.exists(image_path)
            else '<div class="no-img">Image missing</div>'
        )
        labels = ",".join(label for label, count in st["by_label"].items() if count)
        labels = labels or "none"
        table_rows = "".join(
            f"<tr><td>{idx + 1}</td><td>{sp['start']}</td><td>{sp['end']}</td>"
            f"<td>{sp['end'] - sp['start']}</td><td><span class=\"pill\" style=\"--c:{COLORS[sp['label']]}\">{html.escape(sp['label'])}</span></td>"
            f"<td>{sp['prob']:.3f}</td><td>{html.escape(it.get('response', '')[sp['start']:sp['end']])}</td></tr>"
            for idx, sp in enumerate(spans)
        ) or '<tr><td colspan="7" class="muted">No predicted spans</td></tr>'
        cards.append(f"""
<section class="item" data-empty="{str(not spans).lower()}" data-labels="{html.escape(labels)}"
  data-spans="{st['n_spans']}" data-coverage="{st['coverage']:.6f}" data-prob="{st['mean_prob']:.6f}">
  <div class="item-head">
    <div><strong>{html.escape(it['id'])}</strong><span>{st['n_spans']} spans</span><span>{st['coverage']:.1%} chars</span><span>p {st['mean_prob']:.3f}</span></div>
    <a href="{html.escape(img_src(image_path))}" target="_blank" rel="noreferrer">Open image</a>
  </div>
  <div class="grid">
    <div class="image-box">{image_html}</div>
    <div class="content">
      <div class="prompt">{html.escape(it.get('prompt', ''))}</div>
      <div class="response">{render_text(it.get('response', ''), spans)}</div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>start</th><th>end</th><th>len</th><th>label</th><th>prob</th><th>text</th></tr></thead>
          <tbody>{table_rows}</tbody>
        </table>
      </div>
    </div>
  </div>
</section>""")

    label_rows = "".join(
        f"<tr><td><span class=\"pill\" style=\"--c:{COLORS[label]}\">{label}</span></td>"
        f"<td>{count}</td><td>{count / max(1, totals['spans']):.1%}</td></tr>"
        for label, count in totals["labels"].items()
    )
    coverage = totals["covered_chars"] / max(1, totals["chars"])
    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(args.title)}</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: light-dark(#f6f8fa, #0d1117);
  --fg: light-dark(#1f2328, #e6edf3);
  --muted: light-dark(#59636e, #8b949e);
  --line: light-dark(#d0d7de, #30363d);
  --panel: light-dark(#ffffff, #161b22);
  --soft: light-dark(#f6f8fa, #0d1117);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--fg); }}
.top {{ position: sticky; top: 0; z-index: 5; background: color-mix(in srgb, var(--panel) 96%, transparent); border-bottom: 1px solid var(--line); padding: 12px 16px; }}
.title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
h1 {{ font-size: 18px; margin: 0; font-weight: 500; }}
.summary {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 8px; margin-top: 10px; }}
.metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: var(--soft); }}
.metric b {{ display: block; font-size: 18px; font-weight: 500; }}
.controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 10px; }}
button, select, input {{ border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--fg); padding: 6px 8px; font: inherit; }}
input {{ min-width: 240px; }}
.legend {{ display: flex; gap: 8px; flex-wrap: wrap; color: var(--muted); }}
.pill {{ --c: #6e7781; display: inline-flex; align-items: center; gap: 4px; border: 1px solid color-mix(in srgb, var(--c) 45%, var(--line)); color: var(--fg); background: color-mix(in srgb, var(--c) 18%, transparent); border-radius: 999px; padding: 1px 7px; white-space: nowrap; }}
main {{ padding: 12px 16px 28px; }}
.item {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin: 12px 0; overflow: hidden; }}
.item-head {{ display: flex; justify-content: space-between; gap: 10px; padding: 10px 12px; border-bottom: 1px solid var(--line); }}
.item-head div {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.item-head span, .item-head a, .muted {{ color: var(--muted); }}
.grid {{ display: grid; grid-template-columns: minmax(220px, 380px) minmax(0, 1fr); gap: 12px; padding: 12px; }}
.image-box img {{ width: 100%; max-height: 420px; object-fit: contain; border-radius: 6px; background: var(--soft); }}
.no-img {{ min-height: 160px; display: grid; place-items: center; border: 1px dashed var(--line); color: var(--muted); border-radius: 6px; }}
.prompt {{ font-weight: 500; margin-bottom: 8px; }}
.response {{ white-space: normal; border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: var(--soft); }}
mark {{ background: color-mix(in srgb, var(--c) calc(var(--a) * 100%), transparent); color: inherit; border-radius: 3px; padding: 0 1px; }}
.table-wrap {{ overflow-x: auto; margin-top: 10px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 6px; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 500; }}
@media (max-width: 820px) {{
  .summary {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
  .grid {{ grid-template-columns: 1fr; }}
  input {{ min-width: 160px; flex: 1; }}
}}
</style>
</head>
<body>
<div class="top">
  <div class="title"><h1>{html.escape(args.title)}</h1><div class="muted">{html.escape(os.path.basename(args.pred))}</div></div>
  <div class="summary">
    <div class="metric"><b>{totals['items']}</b>Rows with metadata</div>
    <div class="metric"><b>{totals['empty']}</b>Empty predictions</div>
    <div class="metric"><b>{totals['spans']}</b>Total spans</div>
    <div class="metric"><b>{coverage:.1%}</b>Predicted char coverage</div>
    <div class="metric"><b>{len(missing_meta)}</b>Rows missing metadata</div>
  </div>
  <div class="controls">
    <button data-filter="all">All</button>
    <button data-filter="nonempty">Non-empty</button>
    <button data-filter="empty">Empty</button>
    <select id="labelFilter"><option value="all">All labels</option>{"".join(f'<option value="{label}">{label}</option>' for label in VALID_LABELS)}</select>
    <input id="search" placeholder="Search id, prompt, response">
    <button id="sortCoverage">Sort coverage</button>
    <button id="sortSpans">Sort spans</button>
  </div>
  <div class="legend">{"".join(f'<span class="pill" style="--c:{COLORS[label]}">{label}: {totals["labels"][label]}</span>' for label in VALID_LABELS)}</div>
</div>
<main id="items">
{"".join(cards)}
</main>
<script>
const itemsRoot = document.getElementById('items');
const cards = Array.from(document.querySelectorAll('.item'));
let mode = 'all';
function applyFilters() {{
  const label = document.getElementById('labelFilter').value;
  const query = document.getElementById('search').value.trim().toLowerCase();
  for (const card of cards) {{
    const emptyOk = mode === 'all' || (mode === 'empty' && card.dataset.empty === 'true') || (mode === 'nonempty' && card.dataset.empty === 'false');
    const labelOk = label === 'all' || card.dataset.labels.split(',').includes(label);
    const textOk = !query || card.textContent.toLowerCase().includes(query);
    card.style.display = emptyOk && labelOk && textOk ? '' : 'none';
  }}
}}
document.querySelectorAll('button[data-filter]').forEach(btn => btn.addEventListener('click', () => {{ mode = btn.dataset.filter; applyFilters(); }}));
document.getElementById('labelFilter').addEventListener('change', applyFilters);
document.getElementById('search').addEventListener('input', applyFilters);
document.getElementById('sortCoverage').addEventListener('click', () => {{
  cards.sort((a, b) => Number(b.dataset.coverage) - Number(a.dataset.coverage)).forEach(card => itemsRoot.appendChild(card));
}});
document.getElementById('sortSpans').addEventListener('click', () => {{
  cards.sort((a, b) => Number(b.dataset.spans) - Number(a.dataset.spans)).forEach(card => itemsRoot.appendChild(card));
}});
</script>
</body>
</html>
"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
