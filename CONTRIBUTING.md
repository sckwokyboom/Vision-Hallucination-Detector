# Contributing

Thanks for your interest! This project is a research codebase for hallucination-span
detection in image-conditioned VLM outputs. It is organised so that the **deterministic
core is fully unit-tested and model-free**, while the **model-specific experiments live in
`scripts/`** behind small, swappable interfaces.

## Setup

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements/core.txt      # core + tests (no model, runs anywhere)
pip install -r requirements/mlx.txt        # optional: Apple-Silicon VLM inference (MLX)
pip install -r requirements/hf.txt         # optional: GPU/transformers inference
python -m pytest                           # 39 tests, ~0.3s, no model or GPU needed
```

See [`data/README.md`](data/README.md) to obtain the dataset.

## Ground rules

- **Keep the core deterministic and tested.** Anything in `shroom/` (data, split, align,
  aggregate, metrics, predict-parse) must stay model-free and covered by a unit test. Model
  calls belong in `scripts/` or `shroom/backends/`.
- **Add a test with every change to `shroom/`.** Use the `MockBackend` to test end-to-end
  logic without a model (see `tests/test_predict.py`).
- **Evaluate honestly.** New systems must be scored with `scripts/official_metrics.py`
  (span-IoU, roc_auc, pr_auc, calibration, per-class ROC) and compared against the trivial
  baselines. Report confidence intervals for small samples; prefer threshold-free metrics
  (ROC-AUC, specificity@recall) over single greedy points.
- **Don't commit dataset content.** No response text, gold labels, images, or prediction
  logs — they fall under the dataset's terms. Put result *tables* (numbers) in `docs/`.

## Adding things

- **A new inference backend** → implement `shroom.backends.base.VLMBackend.generate(...)`
  and register it in `shroom/run_predict.py::make_backend`.
- **A new detection method** → add a driver in `scripts/`, emit predictions in the standard
  schema (`pred_labels` = list of `{start,end,prob,label}`, plus optional per-char `char_probs`),
  and score with `scripts/official_metrics.py`.
- **A new metric** → add it to `shroom/metrics.py` with a unit test.

## Pull requests

Small, focused PRs with a clear message. Run `python -m pytest` before submitting. If your
change affects results, include the metric table it produces (numbers only).
