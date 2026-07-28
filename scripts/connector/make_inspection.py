"""Build a self-contained HTML inspection report: per dev sample — embedded image,
question, answer rendered twice (GOLD spans vs MODEL spans; color = type, opacity =
probability), per-item IoU, filters and sorting. Usage:

  python scripts/connector/make_inspection.py \
      --gold splits/dev.en.jsonl --pred results/fullscale/dev_pred_<tag>.jsonl \
      --image_dir ../Shroom-Vision/images --out results/fullscale/inspect.html
"""
import argparse
import base64
import html
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shroom.data import load_jsonl                      # noqa: E402
from shroom.metrics import char_iou, gold_char_probs    # noqa: E402
from PIL import Image                                   # noqa: E402

COLORS = {"invention": "#e5484d", "mischaracterization": "#f76b15", "OCR": "#8e4ec6",
          "miscounting": "#0090ff", "other": "#8d8d8d"}


def img_b64(path, max_side=380):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def char_arrays(spans, n):
    prob = [0.0] * n
    lab = [None] * n
    for sp in spans:
        a, b = max(0, int(sp["start"])), min(n, int(sp["end"]))
        p = float(sp.get("prob", 1.0))
        for i in range(a, b):
            if p >= prob[i]:
                prob[i] = p
                lab[i] = sp.get("label", "other")
    return prob, lab


def render_text(text, spans):
    prob, lab = char_arrays(spans, len(text))
    out, i = [], 0
    while i < len(text):
        j = i
        while j < len(text) and lab[j] == lab[i] and abs(prob[j] - prob[i]) < 1e-9:
            j += 1
        seg = html.escape(text[i:j]).replace("\n", "&#10;")
        if lab[i] is None:
            out.append(seg)
        else:
            c = COLORS.get(lab[i], "#8d8d8d")
            alpha = 0.25 + 0.55 * min(1.0, prob[i])
            out.append(f'<mark style="background:{c}; --a:{alpha:.2f}" '
                       f'title="{lab[i]} p={prob[i]:.2f}">{seg}</mark>')
        i = j
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Gold vs Model — manual inspection")
    args = ap.parse_args()

    gold = {it.id: it for it in load_jsonl(args.gold)}
    preds = {}
    for line in open(args.pred, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            preds[d["id"]] = d

    cards = []
    stats = {"n": 0, "clean": 0, "iou_sum": 0.0}
    for pid, pr in preds.items():
        it = gold.get(pid)
        if it is None:
            continue
        n = len(it.response)
        pspans = pr.get("pred_labels", [])
        iou = char_iou(it.labels, pspans, n)
        is_clean = not it.labels
        stats["n"] += 1
        stats["clean"] += is_clean
        stats["iou_sum"] += iou
        try:
            b64 = img_b64(os.path.join(args.image_dir, it.image_name))
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" loading="lazy">'
        except Exception:
            img_tag = '<div class="noimg">no image</div>'
        gold_html = render_text(it.response, it.labels)
        pred_html = render_text(it.response, pspans)
        badge = "clean" if is_clean else "dirty"
        cards.append(f"""
<div class="card" data-iou="{iou:.4f}" data-kind="{badge}">
 <div class="head"><b>{pid}</b> <span class="tag {badge}">{badge}</span>
   <span class="iou">IoU {iou:.3f}</span>
   <span class="cnt">gold spans: {len(it.labels)} | model spans: {len(pspans)}</span></div>
 <div class="row">
  <div class="imgbox">{img_tag}</div>
  <div class="txt">
    <div class="q">Q: {html.escape(it.prompt)}</div>
    <div class="lbl">GOLD</div><div class="ans">{gold_html}</div>
    <div class="lbl">MODEL</div><div class="ans">{pred_html}</div>
  </div>
 </div>
</div>""")

    legend = " ".join(f'<span class="leg"><mark style="background:{c};--a:.6">&nbsp;&nbsp;</mark> {t}</span>'
                      for t, c in COLORS.items())
    mean_iou = stats["iou_sum"] / max(1, stats["n"])
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>
 body{{font:14px/1.45 -apple-system,system-ui,sans-serif;margin:16px;background:#fafafa;color:#111}}
 .card{{background:#fff;border:1px solid #ddd;border-radius:10px;margin:14px 0;padding:12px}}
 .row{{display:flex;gap:14px}} .imgbox{{flex:0 0 380px}} .imgbox img{{max-width:380px;border-radius:6px}}
 .txt{{flex:1;min-width:0}} .q{{color:#444;margin-bottom:8px;font-weight:600}}
 .lbl{{font-size:11px;letter-spacing:.08em;color:#888;margin-top:8px}}
 .ans{{white-space:pre-wrap;border:1px solid #eee;border-radius:6px;padding:8px;background:#fcfcfc}}
 mark{{color:inherit;border-radius:3px;padding:0 1px;background-color:color-mix(in srgb, currentColor 0%, transparent);
      background: color-mix(in srgb, var(--mc, #f00) calc(var(--a)*100%), transparent)}}
 mark[style*="background"]{{background: color-mix(in srgb, currentColor 0%, transparent)}}
 .tag{{padding:1px 8px;border-radius:10px;font-size:12px}} .tag.clean{{background:#d3f2d9}} .tag.dirty{{background:#ffd9d9}}
 .iou{{margin-left:10px;font-weight:600}} .cnt{{margin-left:10px;color:#777;font-size:12px}}
 .leg{{margin-right:14px}} .bar{{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:5;border-bottom:1px solid #eee}}
 button{{margin-right:6px}}
</style></head><body>
<div class="bar"><b>{html.escape(args.title)}</b> — n={stats['n']} (clean {stats['clean']}), mean IoU {mean_iou:.3f}
 &nbsp; {legend}<br>
 <button onclick="sortBy(1)">IoU ↑</button><button onclick="sortBy(-1)">IoU ↓</button>
 <button onclick="filt('all')">all</button><button onclick="filt('dirty')">dirty only</button>
 <button onclick="filt('clean')">clean only</button>
 <i>насыщенность подсветки = вероятность; наведи на спан — тип и p</i></div>
<div id="cards">{''.join(cards)}</div>
<script>
function sortBy(d){{const c=[...document.querySelectorAll('.card')];
 c.sort((a,b)=>d*(parseFloat(a.dataset.iou)-parseFloat(b.dataset.iou)));
 const p=document.getElementById('cards'); c.forEach(x=>p.appendChild(x));}}
function filt(k){{document.querySelectorAll('.card').forEach(x=>{{
 x.style.display=(k==='all'||x.dataset.kind===k)?'':'none';}});}}
</script></body></html>"""
    # fix mark background handling: inline style sets background directly; alpha via opacity trick
    page = page.replace('<mark style="background:', '<mark style="--mc:').replace(
        "mark[style*=\"background\"]{background: color-mix(in srgb, currentColor 0%, transparent)}",
        "mark{background: color-mix(in srgb, var(--mc) calc(var(--a)*100%), transparent) !important}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB, {stats['n']} samples)")


if __name__ == "__main__":
    main()
