# SHROOM-visions Baseline Implementation Plan

**Goal:** Build a model-agnostic evaluation harness (both official metrics, per-language, image-grouped dev split) and a zero-shot Qwen2.5-VL baseline for SHROOM-visions 2026 hallucination-span detection.

**Architecture:** A pure-Python core (data / split / align / aggregate / metrics / prompt-parse) is fully unit-tested via a `MockBackend`, so the whole predict→eval loop runs deterministically without a model or GPU. Real VLM backends (HF transformers on A100, MLX on Mac) are thin adapters behind one `generate()` interface, validated by smoke commands. Self-consistency over N samples yields per-character probabilities: raw frequency feeds the calibration metric; a threshold τ turns it into binary spans for the IoU metric.

**Tech Stack:** Python 3, pytest, numpy, Pillow, PyYAML, tqdm (core); transformers + torch (HF backend); mlx-vlm (Mac backend). Spec: `docs/superpowers/specs/2026-07-21-shroom-visions-baseline-design.md`.

**Working dir:** `vision-detector/` on branch `baseline-pipeline`. Data lives in `../Shroom-Vision/distrib/`; images in `../Shroom-Vision/shroom-visions-images.tar.gz`.

---

## File Structure

```
vision-detector/
  shroom/
    __init__.py
    data.py            # Item dataclass, load_jsonl
    metrics.py         # char_iou, gold_char_probs, pearson/spearman, calibration, trivial baselines
    split.py           # group_split_by_image (deterministic, per image)
    align.py           # phrase_to_spans (exact / normalized / fuzzy)
    predict.py         # CATEGORIES, build_prompt, parse_output, predict_item
    aggregate.py       # aggregate N samples -> per-char prob + spans
    run_predict.py     # CLI: file -> predictions jsonl
    run_eval.py        # CLI: predictions vs gold -> per-language report
    backends/
      __init__.py
      base.py          # VLMBackend ABC, MockBackend
      hf_backend.py    # transformers / Qwen2.5-VL (A100)
      mlx_backend.py   # mlx-vlm / Qwen2.5-VL (Mac)
  configs/
    base.yaml
  scripts/
    extract_images.sh  # untar images once
  tests/
    __init__.py
    conftest.py
    fixtures/mini.train.jsonl
    test_data.py test_metrics.py test_split.py test_align.py
    test_parse.py test_aggregate.py test_predict.py test_eval.py test_backends.py
  requirements-core.txt
  requirements-hf.txt
  requirements-mlx.txt
  pyproject.toml
```

Existing `label_with_gemma.py`, `evaluate.py`, `run_cluster.sh` are kept untouched.

---

## Task 1: Project scaffolding & test environment

**Files:**
- Create: `shroom/__init__.py`, `shroom/backends/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/fixtures/mini.train.jsonl`, `requirements-core.txt`, `pyproject.toml`

- [ ] **Step 1: Create package dirs and requirements**

`requirements-core.txt`:
```
numpy>=1.26
Pillow>=10.0
PyYAML>=6.0
tqdm>=4.66
pytest>=8.0
```

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Create empty `shroom/__init__.py`, `shroom/backends/__init__.py`, `tests/__init__.py`.

- [ ] **Step 2: Create the shared test fixture**

`tests/fixtures/mini.train.jsonl` (3 items; offsets verified against the response strings):
```
{"id":"t-1","language":"en","split":"train","prompt":"Does this desk have legs?","image_name":"desk.jpg","response":"No, it has a hidden bracket on the ceiling.","labels":[{"start":13,"end":19,"prob":0.3333,"label":"mischaracterization"},{"start":35,"end":43,"prob":0.6667,"label":"mischaracterization"}]}
{"id":"t-2","language":"en","split":"train","prompt":"How many cats?","image_name":"cat.jpg","response":"There are three cats.","labels":[{"start":10,"end":15,"prob":1.0,"label":"miscounting"}]}
{"id":"t-3","language":"en","split":"train","prompt":"What color?","image_name":"desk.jpg","response":"It is blue.","labels":[]}
```

`tests/conftest.py`:
```python
import os, pytest
from PIL import Image

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mini.train.jsonl")

@pytest.fixture
def mini_path():
    return FIXTURE

@pytest.fixture
def dummy_image():
    return Image.new("RGB", (8, 8), (127, 127, 127))
```

- [ ] **Step 3: Create venv and install**

Run:
```bash
cd vision-detector
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-core.txt
```
Expected: installs succeed. (If numpy/Pillow wheels are unavailable for your Python, recreate the venv with Python 3.12: `python3.12 -m venv .venv`.)

- [ ] **Step 4: Verify pytest collects nothing yet (sanity)**

Run: `python -m pytest`
Expected: `no tests ran` (exit code 5) — confirms pytest works.

- [ ] **Step 5: Commit**
```bash
git add shroom tests requirements-core.txt pyproject.toml
git commit -m "chore: scaffold shroom package, test env, mini fixture"
```

---

## Task 2: Data loading (`shroom/data.py`)

**Files:**
- Create: `shroom/data.py`, `tests/test_data.py`

- [ ] **Step 1: Write the failing test**

`tests/test_data.py`:
```python
from shroom.data import load_jsonl

def test_load_jsonl(mini_path):
    items = load_jsonl(mini_path)
    assert len(items) == 3
    it = items[0]
    assert it.id == "t-1"
    assert it.language == "en"
    assert len(it.labels) == 2
    # offsets index into response and return the exact substring
    sp = it.labels[0]
    assert it.response[sp["start"]:sp["end"]] == "hidden"
    # unlabeled item yields empty labels list
    assert items[2].labels == []

def test_load_unlabeled_missing_key(tmp_path):
    p = tmp_path / "u.jsonl"
    p.write_text('{"id":"x","language":"en","prompt":"q","image_name":"i.jpg","response":"r"}\n', encoding="utf-8")
    items = load_jsonl(str(p))
    assert items[0].labels == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shroom.data'`.

- [ ] **Step 3: Write the implementation**

`shroom/data.py`:
```python
import json
from dataclasses import dataclass, field


@dataclass
class Item:
    id: str
    language: str
    prompt: str
    image_name: str
    response: str
    labels: list = field(default_factory=list)  # list[dict]: start, end, prob, label
    split: str = ""


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(Item(
                id=d["id"],
                language=d.get("language", ""),
                prompt=d.get("prompt", ""),
                image_name=d.get("image_name", ""),
                response=d.get("response", ""),
                labels=d.get("labels") or [],
                split=d.get("split", ""),
            ))
    return items
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**
```bash
git add shroom/data.py tests/test_data.py
git commit -m "feat: data.py load_jsonl + Item"
```

---

## Task 3: Metrics — char-IoU & gold per-char probs (`shroom/metrics.py`)

**Files:**
- Create: `shroom/metrics.py`, `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
from shroom.metrics import char_iou, gold_char_probs

G = [{"start": 13, "end": 19, "prob": 0.3333, "label": "mischaracterization"},
     {"start": 35, "end": 43, "prob": 0.6667, "label": "mischaracterization"}]  # 6 + 8 = 14 chars
RESP_LEN = 43

def test_iou_empty_empty_is_one():
    assert char_iou([], [], RESP_LEN) == 1.0

def test_iou_gold_nonempty_pred_empty_is_zero():
    assert char_iou(G, [], RESP_LEN) == 0.0

def test_iou_exact_match_is_one():
    assert char_iou(G, G, RESP_LEN) == 1.0

def test_iou_partial():
    pred = [{"start": 13, "end": 19}]  # only "hidden": 6 chars, all inside gold
    # inter=6, union=14
    assert abs(char_iou(G, pred, RESP_LEN) - 6/14) < 1e-9

def test_gold_char_probs_uses_max_over_covering_spans():
    probs = gold_char_probs(G, RESP_LEN)
    assert probs[13] == 0.3333 and probs[18] == 0.3333
    assert probs[35] == 0.6667
    assert probs[0] == 0.0
    assert len(probs) == RESP_LEN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shroom.metrics'`.

- [ ] **Step 3: Write the implementation**

`shroom/metrics.py`:
```python
def _char_set(spans, resp_len):
    s = set()
    for sp in spans:
        a = max(0, min(int(sp["start"]), resp_len))
        b = max(0, min(int(sp["end"]), resp_len))
        s.update(range(a, b))
    return s


def char_iou(gold_spans, pred_spans, resp_len):
    g = _char_set(gold_spans, resp_len)
    p = _char_set(pred_spans, resp_len)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    return len(g & p) / len(g | p)


def gold_char_probs(spans, resp_len):
    probs = [0.0] * resp_len
    for sp in spans:
        a = max(0, min(int(sp["start"]), resp_len))
        b = max(0, min(int(sp["end"]), resp_len))
        pr = float(sp.get("prob", 1.0))
        for i in range(a, b):
            if pr > probs[i]:
                probs[i] = pr
    return probs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**
```bash
git add shroom/metrics.py tests/test_metrics.py
git commit -m "feat: metrics char_iou + gold_char_probs"
```

---

## Task 4: Metrics — calibration correlation & trivial baselines (`shroom/metrics.py`)

**Files:**
- Modify: `shroom/metrics.py`
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test (append)**

Append to `tests/test_metrics.py`:
```python
from shroom.metrics import pearson, spearman, calibration, trivial_baselines
from shroom.data import load_jsonl

def test_pearson_perfect_and_degenerate():
    assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
    assert pearson([1, 1, 1], [2, 4, 6]) == 0.0     # zero variance -> 0
    assert pearson([1.0], [2.0]) == 0.0              # too few points -> 0

def test_spearman_monotonic_is_one():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 99]) - 1.0) < 1e-9

def test_calibration_returns_both():
    out = calibration([0.0, 0.5, 1.0], [0.1, 0.4, 0.9])
    assert set(out) == {"pearson", "spearman"}
    assert out["pearson"] > 0.9

def test_trivial_baselines(mini_path):
    items = load_jsonl(mini_path)
    tb = trivial_baselines(items)
    # predict-nothing IoU = fraction of clean items = 1/3
    assert abs(tb["predict_nothing_iou"] - 1/3) < 1e-9
    # predict-all: t-1 -> 14/43, t-2 -> 5/21, t-3 -> 0 ; mean of the three
    exp = (14/43 + 5/21 + 0.0) / 3
    assert abs(tb["predict_all_iou"] - exp) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'pearson'`.

- [ ] **Step 3: Write the implementation (append)**

Append to `shroom/metrics.py`:
```python
import numpy as np


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=int)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.flatnonzero(obs), len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    return pearson(_rankdata(x), _rankdata(y))


def calibration(gold_probs, pred_probs):
    return {"pearson": pearson(gold_probs, pred_probs),
            "spearman": spearman(gold_probs, pred_probs)}


def trivial_baselines(items):
    """IoU of the two degenerate systems: predict nothing / predict everything."""
    nothing, allh = [], []
    for it in items:
        rl = len(it.response)
        nothing.append(char_iou(it.labels, [], rl))
        full = [{"start": 0, "end": rl}]
        allh.append(char_iou(it.labels, full, rl))
    n = max(1, len(items))
    return {"predict_nothing_iou": sum(nothing) / n,
            "predict_all_iou": sum(allh) / n}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**
```bash
git add shroom/metrics.py tests/test_metrics.py
git commit -m "feat: calibration correlation + trivial baselines"
```

---

## Task 5: Group split by image (`shroom/split.py`)

**Files:**
- Create: `shroom/split.py`, `tests/test_split.py`

- [ ] **Step 1: Write the failing test**

`tests/test_split.py`:
```python
from shroom.data import load_jsonl
from shroom.split import group_split_by_image

def test_split_is_deterministic(mini_path):
    items = load_jsonl(mini_path)
    a = group_split_by_image(items, dev_frac=0.5, seed=13)
    b = group_split_by_image(items, dev_frac=0.5, seed=13)
    assert a == b

def test_no_image_spans_both_splits(mini_path):
    items = load_jsonl(mini_path)
    train_ids, dev_ids = group_split_by_image(items, dev_frac=0.5, seed=13)
    id2img = {it.id: it.image_name for it in items}
    train_imgs = {id2img[i] for i in train_ids}
    dev_imgs = {id2img[i] for i in dev_ids}
    assert train_imgs.isdisjoint(dev_imgs)      # desk.jpg (t-1,t-3) never split apart

def test_split_covers_all_items(mini_path):
    items = load_jsonl(mini_path)
    train_ids, dev_ids = group_split_by_image(items, dev_frac=0.5, seed=13)
    assert set(train_ids) | set(dev_ids) == {it.id for it in items}
    assert set(train_ids).isdisjoint(dev_ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shroom.split'`.

- [ ] **Step 3: Write the implementation**

`shroom/split.py`:
```python
import hashlib


def _bucket(image_name, seed):
    h = hashlib.md5(f"{seed}:{image_name}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def group_split_by_image(items, dev_frac=0.1, seed=13):
    """Assign every item to train/dev by hashing its image_name.

    All items sharing an image land in the same split (no leakage). Deterministic.
    Returns (train_ids, dev_ids).
    """
    train_ids, dev_ids = [], []
    for it in items:
        if _bucket(it.image_name, seed) < dev_frac:
            dev_ids.append(it.id)
        else:
            train_ids.append(it.id)
    return train_ids, dev_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_split.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**
```bash
git add shroom/split.py tests/test_split.py
git commit -m "feat: deterministic image-grouped dev split"
```

---

## Task 6: Phrase → char-span alignment (`shroom/align.py`)

**Files:**
- Create: `shroom/align.py`, `tests/test_align.py`

- [ ] **Step 1: Write the failing test**

`tests/test_align.py`:
```python
from shroom.align import phrase_to_spans

R = "No, it has a hidden bracket on the ceiling."

def test_exact_single():
    assert phrase_to_spans("hidden", R) == [(13, 19)]

def test_all_occurrences():
    r = "cat cat"
    assert phrase_to_spans("cat", r) == [(0, 3), (4, 7)]

def test_normalized_markdown():
    r = "It is a **hidden** bracket."
    # phrase without the asterisks still locates the region (asterisks included in span)
    spans = phrase_to_spans("hidden", r)
    assert len(spans) == 1
    s, e = spans[0]
    assert "hidden" in r[s:e]

def test_normalized_whitespace():
    r = "on the   ceiling"
    assert phrase_to_spans("on the ceiling", r)  # collapsed whitespace matches

def test_fuzzy_typo():
    spans = phrase_to_spans("celing", R)          # missing an 'l'
    assert len(spans) == 1 and R[spans[0][0]:spans[0][1]].startswith("ceil")

def test_no_match_returns_empty():
    assert phrase_to_spans("elephant", R) == []

def test_empty_phrase():
    assert phrase_to_spans("   ", R) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_align.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shroom.align'`.

- [ ] **Step 3: Write the implementation**

`shroom/align.py`:
```python
import re
import difflib

_MD_CHARS = set("*_`#")


def _normalize(s):
    """Lowercase, drop markdown chars, collapse whitespace.

    Returns (norm_str, idx_map) where idx_map[i] is the original index of norm char i.
    """
    out, idx_map, prev_space = [], [], False
    for i, ch in enumerate(s):
        if ch in _MD_CHARS:
            continue
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx_map.append(i)
            prev_space = False
    return "".join(out), idx_map


def _find_all(haystack, needle):
    spans, start = [], haystack.find(needle)
    while start != -1:
        spans.append((start, start + len(needle)))
        start = haystack.find(needle, start + 1)
    return spans


def phrase_to_spans(phrase, response, fuzzy_ratio=0.8):
    """Locate `phrase` inside `response`, returning list of (start, end) char spans.

    Tries exact (all occurrences), then whitespace/markdown-normalized, then fuzzy.
    Returns [] if nothing acceptable is found.
    """
    phrase = phrase.strip()
    if not phrase:
        return []

    # 1. exact
    spans = _find_all(response, phrase)
    if spans:
        return spans

    # 2. normalized
    nresp, rmap = _normalize(response)
    nphr, _ = _normalize(phrase)
    if nphr:
        for s, e in _find_all(nresp, nphr):
            spans.append((rmap[s], rmap[e - 1] + 1))
    if spans:
        return spans

    # 3. fuzzy (contiguous best-match window in normalized space)
    if nphr:
        sm = difflib.SequenceMatcher(None, nresp, nphr, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
        matched = sum(b.size for b in blocks)
        if blocks and matched >= fuzzy_ratio * len(nphr):
            first = blocks[0].a
            last = blocks[-1].a + blocks[-1].size - 1
            return [(rmap[first], rmap[last] + 1)]
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_align.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**
```bash
git add shroom/align.py tests/test_align.py
git commit -m "feat: robust phrase->span alignment (exact/normalized/fuzzy)"
```

---

## Task 7: Backend interface + MockBackend (`shroom/backends/base.py`)

**Files:**
- Create: `shroom/backends/base.py`, `tests/test_backends.py`

- [ ] **Step 1: Write the failing test**

`tests/test_backends.py`:
```python
from shroom.backends.base import VLMBackend, MockBackend

def test_mock_list_script(dummy_image):
    be = MockBackend([["a", "b", "c"], ["x"]])
    assert be.generate(dummy_image, "prompt-1", n=3) == ["a", "b", "c"]
    assert be.generate(dummy_image, "prompt-2", n=1) == ["x"]
    assert be.calls[0][0] == "prompt-1"

def test_mock_callable_script(dummy_image):
    be = MockBackend(lambda text, n: ["[]"] * n)
    assert be.generate(dummy_image, "p", n=2) == ["[]", "[]"]

def test_is_subclass():
    assert issubclass(MockBackend, VLMBackend)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shroom.backends.base'`.

- [ ] **Step 3: Write the implementation**

`shroom/backends/base.py`:
```python
from abc import ABC, abstractmethod


class VLMBackend(ABC):
    @abstractmethod
    def generate(self, image, text, n=1, temperature=0.5):
        """Return a list of `n` raw model output strings for (image, text)."""
        raise NotImplementedError


class MockBackend(VLMBackend):
    """Deterministic backend for tests / harness dry-runs.

    script: either a callable(text, n) -> list[str], or a list where each
    generate() call returns the next element (itself a list[str]).
    """

    def __init__(self, script):
        self._script = script
        self._i = 0
        self.calls = []

    def generate(self, image, text, n=1, temperature=0.5):
        self.calls.append((text, n))
        if callable(self._script):
            return self._script(text, n)
        out = self._script[self._i]
        self._i += 1
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backends.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**
```bash
git add shroom/backends/base.py tests/test_backends.py
git commit -m "feat: VLMBackend interface + MockBackend"
```

---

## Task 8: Prompt building & output parsing (`shroom/predict.py`)

**Files:**
- Create: `shroom/predict.py`, `tests/test_parse.py`

- [ ] **Step 1: Write the failing test**

`tests/test_parse.py`:
```python
from shroom.predict import build_prompt, parse_output, CATEGORIES

def test_categories_are_five():
    assert set(CATEGORIES) == {"invention", "mischaracterization", "OCR", "miscounting", "other"}

def test_build_prompt_mentions_all_categories_and_texts():
    p = build_prompt("How many cats?", "There are three cats.")
    for c in CATEGORIES:
        assert c in p
    assert "How many cats?" in p and "There are three cats." in p

def test_parse_clean_json():
    out = parse_output('[{"phrase":"three","label":"miscounting"}]')
    assert out == [{"phrase": "three", "label": "miscounting"}]

def test_parse_code_fenced():
    raw = "```json\n[{\"phrase\":\"x\",\"label\":\"invention\"}]\n```"
    assert parse_output(raw) == [{"phrase": "x", "label": "invention"}]

def test_parse_embedded_in_prose():
    raw = 'Sure! Here: [{"phrase":"y","label":"OCR"}] done.'
    assert parse_output(raw) == [{"phrase": "y", "label": "OCR"}]

def test_parse_unknown_label_becomes_other():
    assert parse_output('[{"phrase":"z","label":"weird"}]') == [{"phrase": "z", "label": "other"}]

def test_parse_empty_list():
    assert parse_output("[]") == []

def test_parse_garbage_returns_none():
    assert parse_output("not json at all") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shroom.predict'`.

- [ ] **Step 3: Write the implementation**

`shroom/predict.py`:
```python
import json
import re

CATEGORIES = ["invention", "mischaracterization", "OCR", "miscounting", "other"]

PROMPT_TEMPLATE = (
    "Look at the image. A user asked a question about it and a model produced the answer below.\n"
    "Find every span of the answer that is a hallucination — content NOT supported by the image.\n"
    "Classify each span into exactly one category:\n"
    "  - invention: entities, objects, properties or events not present in the image\n"
    "  - mischaracterization: incorrect description of content that IS visible\n"
    "  - OCR: misreading of text visible in the image\n"
    "  - miscounting: wrong quantity of visible items\n"
    "  - other: a hallucination that fits none of the above\n\n"
    "Output ONLY a JSON array. Each element: "
    '{"phrase": "<exact substring copied verbatim from the answer>", "label": "<category>"}. '
    "Copy the phrase EXACTLY, character for character. "
    "If the answer is fully correct, output []."
)


def build_prompt(prompt, response):
    return (f"{PROMPT_TEMPLATE}\n\n"
            f'Question: "{prompt}"\n'
            f'Answer: "{response}"\n'
            "Output:")


def _coerce(parsed):
    if not isinstance(parsed, list):
        return None
    out = []
    for e in parsed:
        if isinstance(e, dict) and "phrase" in e:
            label = e.get("label", "other")
            if label not in CATEGORIES:
                label = "other"
            out.append({"phrase": str(e["phrase"]), "label": label})
    return out


def parse_output(raw):
    """Parse a raw model output into [{phrase, label}], or None if unparseable."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```")).strip()
    try:
        return _coerce(json.loads(raw))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        return _coerce(json.loads(m.group(0)))
    except json.JSONDecodeError:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_parse.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**
```bash
git add shroom/predict.py tests/test_parse.py
git commit -m "feat: 5-category prompt + robust output parser"
```

---

## Task 9: Aggregate N samples → per-char prob + spans (`shroom/aggregate.py`)

**Files:**
- Create: `shroom/aggregate.py`, `tests/test_aggregate.py`

- [ ] **Step 1: Write the failing test**

`tests/test_aggregate.py`:
```python
from shroom.aggregate import aggregate

R = "No, it has a hidden bracket on the ceiling."  # "hidden" = chars 13..18

def _samples():
    return [
        [{"phrase": "hidden", "label": "mischaracterization"}],
        [{"phrase": "hidden", "label": "mischaracterization"}],
        [],  # parsed-but-empty counts as a valid vote of "no hallucination here"
    ]

def test_per_char_prob_frequency():
    spans, prob = aggregate(_samples(), R, tau=0.5)
    assert abs(prob[13] - 2/3) < 1e-9      # covered in 2 of 3 valid samples
    assert prob[0] == 0.0
    assert len(prob) == len(R)

def test_span_built_above_threshold():
    spans, _ = aggregate(_samples(), R, tau=0.5)
    assert len(spans) == 1
    s = spans[0]
    assert (s["start"], s["end"]) == (13, 19)
    assert s["label"] == "mischaracterization"
    assert abs(s["prob"] - 2/3) < 1e-3

def test_threshold_suppresses_low_prob():
    spans, _ = aggregate(_samples(), R, tau=0.7)   # 0.667 < 0.7
    assert spans == []

def test_none_samples_ignored_in_denominator():
    # 2 valid samples both mark "hidden" -> prob 1.0; the None is skipped
    samples = [[{"phrase": "hidden", "label": "invention"}],
               [{"phrase": "hidden", "label": "invention"}],
               None]
    spans, prob = aggregate(samples, R, tau=0.5)
    assert abs(prob[13] - 1.0) < 1e-9

def test_all_none_yields_nothing():
    spans, prob = aggregate([None, None], R, tau=0.5)
    assert spans == [] and set(prob) == {0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shroom.aggregate'`.

- [ ] **Step 3: Write the implementation**

`shroom/aggregate.py`:
```python
from collections import Counter
from .align import phrase_to_spans


def aggregate(samples, response, tau=0.5):
    """Turn N parsed samples into (spans, per_char_prob).

    samples: list where each element is either a list of {phrase, label} dicts,
             or None (unparseable — excluded from the denominator).
    per_char_prob[i] = fraction of *valid* (non-None) samples whose spans cover char i.
    spans: contiguous runs of per_char_prob >= tau, with mean prob and majority label,
           as {start, end, prob, label} dicts (gold schema).
    """
    resp_len = len(response)
    hits = [0] * resp_len
    label_votes = [Counter() for _ in range(resp_len)]
    valid = 0

    for parsed in samples:
        if parsed is None:
            continue
        valid += 1
        covered = {}
        for entry in parsed:
            for a, b in phrase_to_spans(entry["phrase"], response):
                for i in range(a, b):
                    if i not in covered:
                        covered[i] = entry["label"]
        for i, label in covered.items():
            hits[i] += 1
            label_votes[i][label] += 1

    denom = valid if valid > 0 else 1
    per_char_prob = [hits[i] / denom for i in range(resp_len)]

    spans, i = [], 0
    while i < resp_len:
        if per_char_prob[i] >= tau:
            j = i
            while j < resp_len and per_char_prob[j] >= tau:
                j += 1
            run_prob = sum(per_char_prob[i:j]) / (j - i)
            votes = Counter()
            for k in range(i, j):
                votes.update(label_votes[k])
            label = votes.most_common(1)[0][0] if votes else "other"
            spans.append({"start": i, "end": j,
                          "prob": round(run_prob, 4), "label": label})
            i = j
        else:
            i += 1
    return spans, per_char_prob
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aggregate.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**
```bash
git add shroom/aggregate.py tests/test_aggregate.py
git commit -m "feat: self-consistency aggregation -> per-char prob + spans"
```

---

## Task 10: Predict driver + CLI (`shroom/predict.py`, `shroom/run_predict.py`)

**Files:**
- Modify: `shroom/predict.py` (add `predict_item`)
- Create: `shroom/run_predict.py`
- Modify: `tests/test_predict.py` (new file)

- [ ] **Step 1: Write the failing test**

`tests/test_predict.py`:
```python
from shroom.data import load_jsonl
from shroom.predict import predict_item
from shroom.backends.base import MockBackend

def test_predict_item_end_to_end(mini_path, dummy_image):
    items = load_jsonl(mini_path)
    it = items[0]  # response has "hidden" at 13..18
    raws = ['[{"phrase":"hidden","label":"mischaracterization"}]',
            '[{"phrase":"hidden","label":"mischaracterization"}]',
            '[]']
    be = MockBackend([raws])           # one generate() call returns these 3 samples
    spans, per_char = predict_item(be, it, dummy_image, n=3, temperature=0.5, tau=0.5)
    assert len(spans) == 1
    assert (spans[0]["start"], spans[0]["end"]) == (13, 19)
    assert len(per_char) == len(it.response)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predict.py -v`
Expected: FAIL — `ImportError: cannot import name 'predict_item'`.

- [ ] **Step 3: Write `predict_item` (append to `shroom/predict.py`)**

```python
from .aggregate import aggregate


def predict_item(backend, item, image, n=5, temperature=0.5, tau=0.5):
    """Run one item through the VLM backend and aggregate into (spans, per_char_prob)."""
    text = build_prompt(item.prompt, item.response)
    raws = backend.generate(image, text, n=n, temperature=temperature)
    samples = [parse_output(r) for r in raws]
    return aggregate(samples, item.response, tau=tau)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_predict.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Write the CLI**

`shroom/run_predict.py`:
```python
import argparse
import json
import os

import yaml
from PIL import Image
from tqdm import tqdm

from .data import load_jsonl
from .predict import predict_item


def make_backend(cfg):
    kind = cfg.get("backend", "mock")
    if kind == "hf":
        from .backends.hf_backend import HFBackend
        return HFBackend(model_id=cfg["model_id"], max_pixels=cfg.get("max_pixels", 1024 * 1024))
    if kind == "mlx":
        from .backends.mlx_backend import MLXBackend
        return MLXBackend(model_id=cfg["model_id"])
    raise ValueError(f"Unknown/unsupported backend for prediction: {kind!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    backend = make_backend(cfg)
    items = load_jsonl(args.input)
    if args.max_samples:
        items = items[:args.max_samples]

    image_dir_real = os.path.realpath(args.image_dir) + os.sep
    with open(args.output, "w", encoding="utf-8") as out:
        for it in tqdm(items, desc="predict"):
            rec = {"id": it.id, "language": it.language, "response": it.response}
            img_path = os.path.realpath(os.path.join(args.image_dir, it.image_name))
            if not img_path.startswith(image_dir_real) or not os.path.exists(img_path):
                rec["pred_labels"] = []
                rec["char_probs"] = []
                rec["error"] = "image_missing"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            image = Image.open(img_path).convert("RGB")
            spans, per_char = predict_item(
                backend, it, image,
                n=cfg.get("n_samples", 5),
                temperature=cfg.get("temperature", 0.5),
                tau=cfg.get("tau", 0.5),
            )
            rec["pred_labels"] = spans
            rec["char_probs"] = [round(p, 3) for p in per_char]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**
```bash
git add shroom/predict.py shroom/run_predict.py tests/test_predict.py
git commit -m "feat: predict_item + run_predict CLI"
```

---

## Task 11: Eval driver + per-language report (`shroom/run_eval.py`, `shroom/metrics.py`)

**Files:**
- Modify: `shroom/metrics.py` (add `evaluate`)
- Create: `shroom/run_eval.py`
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

`tests/test_eval.py`:
```python
from shroom.data import load_jsonl
from shroom.metrics import evaluate

def test_evaluate_per_language(mini_path):
    gold = load_jsonl(mini_path)
    # predictions: t-1 exact gold spans, t-2 empty, t-3 empty
    pred_by_id = {
        "t-1": {"pred_labels": [{"start": 13, "end": 19, "prob": 0.7, "label": "mischaracterization"},
                                {"start": 35, "end": 43, "prob": 0.7, "label": "mischaracterization"}]},
        "t-2": {"pred_labels": []},
        "t-3": {"pred_labels": []},
    }
    rep = evaluate(gold, pred_by_id)
    en = rep["en"]
    assert en["n"] == 3
    # IoU: t-1=1.0, t-2=0.0 (gold nonempty, pred empty), t-3=1.0  -> mean 2/3
    assert abs(en["iou"] - 2/3) < 1e-9
    assert abs(en["predict_nothing_iou"] - 1/3) < 1e-9
    assert "pearson" in en and "spearman" in en
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval.py -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate'`.

- [ ] **Step 3: Write `evaluate` (append to `shroom/metrics.py`)**

```python
from collections import defaultdict


def _pred_char_probs(pred_rec, resp_len):
    """Per-char pred prob: use stored char_probs if present, else reconstruct from spans."""
    cp = pred_rec.get("char_probs")
    if cp:
        arr = list(cp)[:resp_len]
        arr += [0.0] * (resp_len - len(arr))
        return arr
    arr = [0.0] * resp_len
    for sp in pred_rec.get("pred_labels", []):
        a = max(0, min(int(sp["start"]), resp_len))
        b = max(0, min(int(sp["end"]), resp_len))
        pr = float(sp.get("prob", 1.0))
        for i in range(a, b):
            if pr > arr[i]:
                arr[i] = pr
    return arr


def evaluate(gold_items, pred_by_id):
    """Per-language report: mean IoU, per-label IoU, calibration, trivial baselines."""
    by_lang = defaultdict(list)
    for it in gold_items:
        by_lang[it.language].append(it)

    report = {}
    for lang, items in by_lang.items():
        ious, gp_all, pp_all = [], [], []
        label_inter = defaultdict(int)
        label_union = defaultdict(int)
        for it in items:
            rl = len(it.response)
            pred = pred_by_id.get(it.id, {"pred_labels": []})
            pred_spans = pred.get("pred_labels", [])
            ious.append(char_iou(it.labels, pred_spans, rl))
            gp_all.extend(gold_char_probs(it.labels, rl))
            pp_all.extend(_pred_char_probs(pred, rl))
            # per-label agnostic IoU accumulation
            for lab in CATEGORIES:
                g = _char_set([s for s in it.labels if s.get("label") == lab], rl)
                p = _char_set([s for s in pred_spans if s.get("label") == lab], rl)
                label_inter[lab] += len(g & p)
                label_union[lab] += len(g | p)
        tb = trivial_baselines(items)
        cal = calibration(gp_all, pp_all)
        per_label = {lab: (label_inter[lab] / label_union[lab] if label_union[lab] else 1.0)
                     for lab in CATEGORIES}
        report[lang] = {
            "n": len(items),
            "iou": sum(ious) / len(ious),
            "per_label_iou": per_label,
            "pearson": cal["pearson"],
            "spearman": cal["spearman"],
            "predict_nothing_iou": tb["predict_nothing_iou"],
            "predict_all_iou": tb["predict_all_iou"],
        }
    return report
```

Note: `evaluate` uses `CATEGORIES`; add `from .predict import CATEGORIES` at the top of `metrics.py`. (This import is safe: `predict.py` does not import `metrics.py` at module load — it imports `aggregate`, which imports `align`. No cycle.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Write the CLI**

`shroom/run_eval.py`:
```python
import argparse
import json

from .data import load_jsonl
from .metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    pred_by_id = {}
    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                pred_by_id[d["id"]] = d

    report = evaluate(gold, pred_by_id)
    for lang in sorted(report):
        r = report[lang]
        print(f"\n=== {lang}  (n={r['n']}) ===")
        print(f"  Char-IoU:            {r['iou']:.4f}")
        print(f"  [baseline nothing]:  {r['predict_nothing_iou']:.4f}")
        print(f"  [baseline all]:      {r['predict_all_iou']:.4f}")
        print(f"  Calibration Pearson: {r['pearson']:.4f}   Spearman: {r['spearman']:.4f}")
        print(f"  Per-label IoU: " +
              "  ".join(f"{k}={v:.3f}" for k, v in r["per_label_iou"].items()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest`
Expected: PASS (all tests green — ~30+).

- [ ] **Step 7: Commit**
```bash
git add shroom/metrics.py shroom/run_eval.py tests/test_eval.py
git commit -m "feat: evaluate() per-language report + run_eval CLI"
```

---

## Task 12: Config + image extraction script (`configs/base.yaml`, `scripts/extract_images.sh`)

**Files:**
- Create: `configs/base.yaml`, `scripts/extract_images.sh`

- [ ] **Step 1: Write the config**

`configs/base.yaml`:
```yaml
# Backend: mock | hf | mlx
backend: hf
model_id: Qwen/Qwen2.5-VL-7B-Instruct
n_samples: 5
temperature: 0.5
tau: 0.5
max_pixels: 1048576   # 1024*1024
dev_frac: 0.1
seed: 13
```

- [ ] **Step 2: Write the extraction script**

`scripts/extract_images.sh`:
```bash
#!/bin/bash
# Extract the image archive once into ../Shroom-Vision/images/
set -eu
ARCHIVE="${1:-../Shroom-Vision/shroom-visions-images.tar.gz}"
DEST="${2:-../Shroom-Vision/images}"
mkdir -p "$DEST"
echo "Extracting $ARCHIVE -> $DEST ..."
tar -xzf "$ARCHIVE" -C "$DEST" --strip-components=0
echo "Done. Image count: $(find "$DEST" -type f | wc -l)"
```

- [ ] **Step 3: Verify the config loads**

Run: `python -c "import yaml; print(yaml.safe_load(open('configs/base.yaml'))['model_id'])"`
Expected: `Qwen/Qwen2.5-VL-7B-Instruct`

- [ ] **Step 4: Commit**
```bash
git add configs/base.yaml scripts/extract_images.sh
git commit -m "chore: base config + image extraction script"
```

---

## Task 13: HF backend for A100 (`shroom/backends/hf_backend.py`)

**Files:**
- Create: `shroom/backends/hf_backend.py`, `requirements-hf.txt`

Not unit-tested (needs GPU + model weights). Provided as a complete adapter, validated by the smoke command in Step 3.

- [ ] **Step 1: Write requirements and implementation**

`requirements-hf.txt`:
```
torch
transformers>=4.49
accelerate
qwen-vl-utils
Pillow
```

`shroom/backends/hf_backend.py`:
```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import VLMBackend


class HFBackend(VLMBackend):
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct",
                 max_pixels=1024 * 1024, dtype=torch.bfloat16):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map="auto", dtype=dtype,
            attn_implementation="sdpa", trust_remote_code=True,
        )
        self.model.eval()
        self.max_pixels = max_pixels

    def generate(self, image, text, n=1, temperature=0.5):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text}]}]
        templ = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=templ, images=image, return_tensors="pt",
            max_pixels=self.max_pixels).to(self.model.device)
        plen = inputs.input_ids.shape[-1]
        with torch.no_grad():
            gen = self.model.generate(
                **inputs, max_new_tokens=512,
                do_sample=temperature > 0, temperature=temperature,
                top_p=0.95, num_return_sequences=n)
        return [self.processor.decode(seq[plen:], skip_special_tokens=True).strip()
                for seq in gen]
```

- [ ] **Step 2: Extract images (once)**

Run: `bash scripts/extract_images.sh`
Expected: prints a non-zero image count under `../Shroom-Vision/images`.

- [ ] **Step 3: Smoke test on the cluster (2 items)**

Run:
```bash
python -m shroom.run_predict --config configs/base.yaml \
  --input ../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl \
  --image_dir ../Shroom-Vision/images \
  --output /tmp/smoke_en.jsonl --max_samples 2
python -m shroom.run_eval --gold ../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl --pred /tmp/smoke_en.jsonl
```
Expected: predictions written for 2 items; eval prints an `en` block with Char-IoU and the two baseline lines. (Numbers are not asserted here — this only proves the model path runs end-to-end.)

- [ ] **Step 4: Commit**
```bash
git add shroom/backends/hf_backend.py requirements-hf.txt
git commit -m "feat: HF (transformers) Qwen2.5-VL backend for A100"
```

---

## Task 14: MLX backend for Mac (`shroom/backends/mlx_backend.py`)

**Files:**
- Create: `shroom/backends/mlx_backend.py`, `requirements-mlx.txt`

Mac-only smoke path. `mlx-vlm`'s helper names have shifted across releases, so pin the version below and adjust the two helper calls if the smoke test reports an import/signature error.

- [ ] **Step 1: Write requirements and implementation**

`requirements-mlx.txt`:
```
mlx-vlm==0.1.12
Pillow
```

`shroom/backends/mlx_backend.py`:
```python
from mlx_vlm import load, generate as mlx_generate
from mlx_vlm.prompt_utils import apply_chat_template

from .base import VLMBackend


class MLXBackend(VLMBackend):
    def __init__(self, model_id="mlx-community/Qwen2.5-VL-3B-Instruct-4bit"):
        self.model, self.processor = load(model_id)
        self.config = self.model.config

    def generate(self, image, text, n=1, temperature=0.5):
        prompt = apply_chat_template(self.processor, self.config, text, num_images=1)
        outs = []
        for _ in range(n):
            out = mlx_generate(
                self.model, self.processor, prompt, image=[image],
                temperature=temperature, max_tokens=512, verbose=False)
            outs.append((out.text if hasattr(out, "text") else str(out)).strip())
        return outs
```

- [ ] **Step 2: Smoke test on Mac (config override to mlx, 5 items)**

Run:
```bash
. .venv/bin/activate
pip install -r requirements-mlx.txt
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/base.yaml"))
c["backend"] = "mlx"; c["model_id"] = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"; c["n_samples"] = 3
yaml.safe_dump(c, open("configs/mlx.yaml", "w"))
PY
bash scripts/extract_images.sh
python -m shroom.run_predict --config configs/mlx.yaml \
  --input ../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl \
  --image_dir ../Shroom-Vision/images \
  --output /tmp/smoke_mlx.jsonl --max_samples 5
python -m shroom.run_eval --gold ../Shroom-Vision/distrib/shroom-vision.train.en.labeled.jsonl --pred /tmp/smoke_mlx.jsonl
```
Expected: predictions for 5 items; eval prints an `en` block. Confirms the full pipeline runs locally on Mac.

- [ ] **Step 3: Commit**
```bash
git add shroom/backends/mlx_backend.py requirements-mlx.txt configs/mlx.yaml
git commit -m "feat: MLX Qwen2.5-VL backend for Mac smoke tests"
```

---

## Task 15: Dev-split materialization, full dev run, README

**Files:**
- Create: `shroom/make_split.py`, `README_baseline.md`

- [ ] **Step 1: Write the split materializer**

`shroom/make_split.py`:
```python
import argparse
import json
import os

import yaml

from .data import load_jsonl
from .split import group_split_by_image

LANGS = ["en", "fr", "it", "zh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distrib", default="../Shroom-Vision/distrib")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out_dir", default="splits")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out_dir, exist_ok=True)
    for lang in LANGS:
        path = os.path.join(args.distrib, f"shroom-vision.train.{lang}.labeled.jsonl")
        items = load_jsonl(path)
        train_ids, dev_ids = group_split_by_image(
            items, dev_frac=cfg.get("dev_frac", 0.1), seed=cfg.get("seed", 13))
        json.dump({"train": train_ids, "dev": dev_ids},
                  open(os.path.join(args.out_dir, f"{lang}.json"), "w"))
        # write the dev gold subset for convenient eval
        dev_set = set(dev_ids)
        with open(os.path.join(args.out_dir, f"dev.{lang}.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                if it.id in dev_set:
                    f.write(json.dumps({
                        "id": it.id, "language": it.language, "prompt": it.prompt,
                        "image_name": it.image_name, "response": it.response,
                        "labels": it.labels}, ensure_ascii=False) + "\n")
        print(f"{lang}: train={len(train_ids)} dev={len(dev_ids)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Materialize splits and check dev sizes**

Run: `python -m shroom.make_split`
Expected: prints four lines like `en: train=~3400 dev=~380` (dev ≈ 10% of each language), and creates `splits/dev.{lang}.jsonl`.

- [ ] **Step 3: Harness dry-run — trivial baselines on real dev (no model)**

Create an empty prediction file and confirm the harness reports the ~0.28 plank on real data:
```bash
python - <<'PY'
import json
src, dst = "splits/dev.en.jsonl", "/tmp/dev_en_nothing.jsonl"
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as o:
    for line in f:
        d = json.loads(line)
        o.write(json.dumps({"id": d["id"], "language": d["language"],
                            "response": d["response"], "pred_labels": []}) + "\n")
print("wrote", dst)
PY
python -m shroom.run_eval --gold splits/dev.en.jsonl --pred /tmp/dev_en_nothing.jsonl
```
Expected: `Char-IoU` equals `[baseline nothing]` (both ≈ 0.25 for en), proving the harness and split are wired to real data. This is Milestone 1 complete.

- [ ] **Step 4: Full dev prediction run on A100** (after Task 13 smoke passes)

Run:
```bash
python -m shroom.run_predict --config configs/base.yaml \
  --input splits/dev.en.jsonl --image_dir ../Shroom-Vision/images \
  --output preds/dev.en.jsonl
python -m shroom.run_eval --gold splits/dev.en.jsonl --pred preds/dev.en.jsonl
```
Expected: `Char-IoU` **clearly above** the `[baseline nothing]` line for `en`. Record the number; this is the baseline result.

- [ ] **Step 5: Write the README and commit**

`README_baseline.md`: a short usage doc covering (1) `python -m shroom.make_split`, (2) `bash scripts/extract_images.sh`, (3) predict + eval commands for cluster (hf) and Mac (mlx), (4) how to sweep `tau` and `n_samples` in `configs/base.yaml`, (5) that `run_eval` prints trivial baselines every time.

```bash
git add shroom/make_split.py README_baseline.md
git commit -m "feat: split materializer + baseline README; harness on real dev"
```

---

## Self-Review

**Spec coverage:**
- §7 both metrics → Tasks 3 (IoU), 4 (calibration + trivial baselines), 11 (per-language `evaluate`). ✓
- §4 Method A (extraction + self-consistency, τ threshold) → Tasks 8 (prompt/parse), 9 (aggregate), 10 (predict). ✓
- §5 backend abstraction (HF + MLX + Mock) → Tasks 7, 13, 14. ✓
- §6 image-grouped dev split → Tasks 5, 15. ✓
- §8 gold-schema output + `char_probs` for calibration → Task 10 (`pred_labels` + `char_probs`), Task 11 (`_pred_char_probs` prefers stored array). ✓
- §9 model/compute (Qwen2.5-VL, resolution) → Tasks 12, 13; `max_pixels` raised to 1024². ✓
- §10 milestones → Task 15 (Step 3 = M1 harness; Task 14 = M2 Mac smoke; Step 4 = M3 dev run). ✓
- §11 image extraction one-time step → Task 12/13. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows full code. README (Task 15 Step 5) is described by explicit contents, not left vague. ✓

**Type consistency:** Spans are plain dicts `{start,end,prob,label}` everywhere (data, align→aggregate, predict output, metrics). `predict_item`/`aggregate` return `(spans, per_char_prob)` consistently. `evaluate` consumes `pred_by_id[id]` as the full record dict (uses `.get("pred_labels")`, `.get("char_probs")`), which matches `run_eval`'s loader and `run_predict`'s writer. `CATEGORIES` defined once in `predict.py`, imported by `metrics.py`. ✓ (Import direction verified acyclic: `metrics`→`predict`→`aggregate`→`align`; `metrics` is not imported by any of those.)

---

## Execution Handoff

Plan complete. After approval, two execution options:
1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.
