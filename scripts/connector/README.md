# Task-specific connector over frozen Gemma 4 12B

Implementation of the connector spec: a small trainable cross-attention module over a
**frozen** Gemma 4 12B that, for every character of a candidate answer, predicts hard-span
membership `q_c`, soft hallucination probability `p_c`, five type probabilities `p_{c,k}`,
and an auxiliary clean/dirty gate. Targets the official character-level IoU, Cor, Cor-lbl.

## Design

**Two stages, so ablations are cheap:**

1. `extract_features.py` — one frozen forward pass per item; caches to `.npz`:
   `V` (hidden states at visual-token positions) and `H` (hidden states of the **review
   copy** of the answer — the prompt repeats the answer so every review token attends to
   the full original), at layers `{24,32,40,48}` (learnable softmax mix at train time).
   Prompt: `<image> Question: … Candidate answer: … Review token by token: …`
2. `train_connector.py` — trains on the cache. Architecture (`model.py`):
   `Q=Lin(H), K/V=Lin(V)` → 2–4 cross-attention blocks (residual+LN, 8 heads, dropout 0.1)
   → fusion MLP over `[H, C, H−C, H⊙C]` → char scatter via tokenizer offsets + in-token
   position embedding + 1D-CNN refiner → four heads.

**Loss** (per-example averaged, then batch): `λ1·BCE(p_c, y_c)` soft probs + `λ2·soft-Jaccard(q_c, m_c)`
(IoU surrogate) + `λ3·pairwise ranking` (Cor surrogate) + `λ4·BCE types` + `λ5·BCE gate`.
Gold: `y_c` = annotator prob of covering spans; `y_{c,k}` per type.

**Postprocessing:** dev threshold sweep on `q_c` → drop below-threshold chars → argmax type
→ merge same-type neighbours → official JSONL `{"start","end","prob","label"}`.

## Run (single A100)

```bash
bash run_connector.sh ../Shroom-Vision google/gemma-4-12B-it
```

Runs: probe (validates token alignment) → subset cache (1200 items, quick signal) →
grid: linear readout baseline / main connector (+ shuffled-image eval) / text-only.
Stage 2 takes minutes per variant on cached features; rerun stage 1b without
`--max_items` for the full cache afterwards.

## Experiment grid & success criteria

| variant | purpose |
|---|---|
| `--arch linear` | frozen readout baseline |
| `--arch connector` | main method |
| `--no_image` | text-only control |
| `--eval_shuffle` | shuffled-image grounding check |
| `--processor_kwargs` / `--max_side` | resolution ablation (visual token count — verify via probe) |

Success = connector beats the linear readout on span_iou / calib / calib_lbl; correct-image
beats text-only and shuffled; span_iou clears the predict-nothing floor; gains hold on the
image-grouped held-out split (and across seeds / paired bootstrap).

## Notes / knowns

- Visual-token count depends on the HF processor settings; the probe prints it. The
  280-vs-1120 ablation maps to processor/pan-scan configuration (`--processor_kwargs`).
- Cache size: ~[visual_tokens × 4 layers × hidden × 2B] per item; with 4 cached layers and
  ~1120 tokens this is tens of MB/item — use the subset cache first, or cache fewer layers
  (`--layers 32`).
- The token↔position alignment between the offsets tokenizer pass and the processor pass is
  handled via a length shift; the probe prints it (`shift=`) — verify it is constant.
