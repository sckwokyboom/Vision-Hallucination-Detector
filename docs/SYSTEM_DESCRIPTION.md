# System Description — SHROOM-visions 2026 submission `gemma4-lora-char-span-decoder(-refine)`

Character-level hallucination-span detection in image-conditioned VLM answers, four
language tracks (en / fr / it / zh). One submission file per language; all design
decisions validated with the organizers' official scorer on image-disjoint dev splits.

## 1. Overview

The system reads (image, question, answer) with a **LoRA-adapted Gemma-4-12B-it**
backbone and predicts, per character of the answer: a hallucination probability, span
boundaries, and a hallucination type. It does **not generate text**: the answer is
embedded in a fixed prompt template, the backbone is run once, and a small trained
decoder maps intermediate hidden states to character-level predictions. Span types are
assigned by a **separate class-balanced classifier** applied as post-processing.

Per-language checkpoints:

| track | training data | model |
|---|---|---|
| en | en only (3,440 items) | LoRA, 5 epochs, best epoch 4 |
| fr | fr only (3,410) | LoRA, 3 epochs, best epoch 3 |
| it | it only (3,417) | LoRA, 3 epochs, best epoch 3 |
| zh (v3) | all 4 languages (12,874) | multilingual LoRA, 3 epochs, best epoch 2 |

The multilingual model helps only zh (+0.032 IoU / +0.042 Cor on its dev over the
zh-only model) — the track with the fewest hallucinated training answers benefits most
from 4x data; en/fr/it kept their dedicated models (no measured gain).

## 2. Input encoding

Fixed non-generative template, single user turn with the image:

```
Question: {question}
Candidate answer: {answer}
Review token by token:
{answer}
```

The second ("review") copy of the answer ends the prompt; its token positions are the
feature-extraction sites. Alignment from tokens back to character offsets uses the
fast tokenizer's offset mapping (two earlier bugs mattered here: chat templates rstrip
message content — 54/3799 en answers end in newlines; and decode-and-search offset
reconstruction silently corrupts non-ASCII text — 32 fr/it/zh items. Both fixed and
regression-tested; alignment verified exact on all ~20k items of all 8 files).
Images are resized to max side 768 (the processor caps visual tokens at 280 anyway).

## 3. Backbone and LoRA

- **google/gemma-4-12B-it**, bf16, frozen. 48 transformer layers, hidden size 3840,
  unified decoder-only architecture (no separate vision encoder).
- **LoRA** (hand-rolled): rank 16, alpha 32, dropout 0.05, fp32 adapters, on the
  q/k/v/o attention projections of layers 24–47. That is 92 (not 96) modules:
  Gemma-4 sets `attention_k_eq_v`, so its four full-attention layers in that range
  have no v_proj. 10.7 M trainable adapter parameters.
- Layer choice covers every feature tap (see below); adapting only the top 8 layers
  would leave half the tapped layers frozen by construction.

## 4. Decoder (~14.6 M parameters)

Hidden states are taken from layers **{24, 32, 40, 47}** at the review-copy token
positions (`H ∈ R^{T x 4 x 3840}`), then:

1. **LayerMix** — learnable softmax mix over the 4 layers;
2. Linear projection to d=768;
3. token→character scatter (each character inherits its token's vector) + a 32-d
   in-token position embedding;
4. **char refiner**: two Conv1d (k=5) + GELU, then a residual **BiGRU**;
5. heads: soft-span logit (q), per-character probability (p), **BIO** tags (3-way),
   an answer-level abstention **gate** (on mean-pooled features), and a type head
   (superseded at inference by the external classifier).

Decoding: BIO tags → candidate spans; **two-signal abstention** — the answer is
declared clean if the gate is below g_thr OR the mean of the top-5% character
probabilities is below tau. (tau, g_thr) are selected per language by an official-
scorer sweep on that language's dev split. Span probability = mean p over the span.

## 5. Training objective

Per-example-averaged sum (v3 recipe):

```
L = 1.0 * BCE(p, y_soft)            # y_soft = official SUM of annotator span probs, clipped to [0,1]
  + 1.0 * Tversky(q, y>0; a=0.7, b=0.3)
  + 0.5 * pairwise ranking (hallucinated chars above clean chars)
  + 0.5 * CE(BIO, tags; class weights [O,B,I]=[1,4,2])
  + 0.2 * BCE(gate, has_hallucination)
  + 0.3 * MSE(sigmoid(gate), top-5%-mean of sigmoid(p))   # gate consistency
```

The training target uses the official scorer's aggregation semantics (overlapping
annotator spans SUM). No type loss during locator training — joint type training
repeatedly degraded localization, which motivated the standalone classifier.

Optimization: batch 1 with gradient accumulation 16 (live backbone forward per item);
AdamW, lr 3e-4 (decoder) / 1e-4 (LoRA), grad clip 1.0; epoch 1 is a decoder-only
warm-up with LoRA frozen; seed 13; best checkpoint by dev IoU. One epoch over 3.4k
items ≈ 12–15 min on a single H100-80GB (≈ 48 min for the multilingual 12.9k).

## 6. Span-type classifier (the "-refine" step)

Trained separately per language on **gold** spans of the training split: features are
the layer-averaged frozen hidden state meaned over the span's tokens (3840) plus
length and position scalars; model is LayerNorm → Linear(3843,256) → GELU → Dropout
→ Linear(256,5); class-balanced CE (invention/mischaracterization are 84% of spans);
best checkpoint by balanced accuracy (en: 0.739 on 11,489 spans; 0.70–0.74 across
languages). At inference it relabels the decoder's predicted spans — spans and
probabilities stay byte-identical, so IoU and Cor are unchanged by construction.

## 7. Post-processing

Swept on dev with the official scorer: trim edge whitespace/punctuation, drop spans
shorter than 3 characters, merge same-label spans separated by ≤2 characters.
Measured-neutral and therefore NOT applied: snapping span edges to word boundaries
(raw output crosses word boundaries 10x more often than gold, but char-IoU is
indifferent — expansion adds as many wrong characters as it repairs).

## 8. Evaluation protocol

- **Image-disjoint splits** everywhere (items sharing an image never straddle a
  split; hash-bucketed by image name). For multilingual training, a hard filter
  additionally drops any item of any language sharing an image with an eval item
  (121/142 en-eval images also occur in the fr training set).
- en: frozen tune-202 for all decisions + a **single-shot held-out-157** evaluation
  of the final pipeline before submission (IoU 0.3850 [0.319–0.451], Cor 0.4267 —
  confirming the tune numbers transfer).
- Everything is scored with the organizers' vendored scorer; paired image-cluster
  bootstrap CIs against the predict-nothing floor.

## 9. Results

Test leaderboard (v1 = constant labels, v2 = classifier labels; spans identical):

| track | Cor+Lbl v1 → v2 | Cor v1 → v2 |
|---|---|---|
| EN | 0.1993 → **0.3129** | 0.4477 → **0.4491** |
| FR | 0.2360 → **0.3181** | 0.4323 → **0.4378** |
| IT | 0.2480 → **0.3373** | 0.4892 → **0.4926** |
| ZH | 0.3072 → **0.3759** | 0.4833 → **0.4882** |

Dev (official scorer; floor = predict-nothing):

| track | floor | IoU | Cor | Cor_lbl |
|---|---:|---:|---:|---:|
| en (tune-202; held-out in parens) | 0.213 | 0.4221 (0.3850) | 0.4529 (0.4267) | 0.3348 |
| fr | 0.297 | 0.4513 | 0.4393 | 0.3534 |
| it | 0.289 | 0.4335 | 0.4696 | 0.3595 |
| zh, zh-only model | 0.355 | 0.4703 | 0.4868 | 0.4180 |
| **zh, multilingual model (v3)** | 0.355 | **0.5121** | **0.5389** | 0.3639* |

\* before relabeling. All IoU/Cor deltas vs floor significant (image-cluster bootstrap).

English progression during development (tune-202, official): predict-nothing 0.2129 →
starter baseline 0.2005 → zero-shot claim-gate hybrid 0.273 → frozen-feature decoder
0.37 → +LoRA 0.4151 → +post-processing 0.4225.

## 10. Findings that shaped the design (incl. negative results)

1. **Decoder architecture saturates**: six decode-head families (threshold/hysteresis,
   peak, BIO, DETR-style set prediction, segment-ranker+DP, semi-Markov CRF) converge
   once features are good. BIO kept for simplicity.
2. **Direct visual-token access is useless**: giving the decoder cross-attention over
   visual-token hidden states adds nothing — frozen (ReZero gate never opens;
   eval-time image shuffling changes predictions by float-noise only) or LoRA-adapted.
   Everything the decoder needs reaches it through the answer-token states.
3. **Backbone adaptation is the main lever**: LoRA +0.039 IoU, isolated from
   continued-training confounds by a frozen-adapter control run through the identical
   pipeline.
4. **Splitting clean/dirty objectives** (dirty-only locator + separate gate) matches
   LoRA's localization gain on frozen features but overlaps with it — no additive win.
5. **Ensembles fail here**: char-probability averaging destroys the structured
   BIO+gate decode (−0.04); span-level union/intersection/fill variants all lose to
   the single best system (systems too correlated, r≈0.73).
6. **More data helps only where capacity adapts and data is scarce**: 4x multilingual
   data moved frozen-feature training by 0.000 and en/fr/it LoRA within noise, but
   gave zh (the track with the fewest dirty answers) +0.03/+0.04.
7. **Type information lives in the frozen features** (balanced acc 0.74) — it was the
   joint training that kept the type head dead, not the representation.
8. Quantization is a non-issue for feature extraction (nf4 ≈ int8 ≈ bf16, paired).

## 11. Reproducibility

Code: https://github.com/sckwokyboom/Vision-Hallucination-Detector — extraction,
training, evaluation (vendored official scorer + format checker), post-processing,
classifiers, and the experiment log (results/STATUS.md) with per-run manifests,
fixed seeds and submission SHA-256 hashes. Only the official SHROOM-visions 2026
training data and images were used; no external or synthetic data, no RAG, no
text generation.
