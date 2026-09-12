"""Eval-time image controls (eval_ablation.py) and the offline sweeps that read its
dumps (ablation_sweeps.py). No model, no GPU: the dumps are synthesised."""
import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "connector"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "official"))

import eval_ablation as ea        # noqa: E402
import ablation_sweeps as sw      # noqa: E402
from shroom.data import Item      # noqa: E402


def test_derangement_has_no_fixed_points_and_is_a_bijection():
    names = [f"img{i}.jpg" for i in range(7)] + ["img3.jpg", "img0.jpg"]   # duplicates
    m = ea.derangement(names, seed=13)
    assert set(m) == set(names)
    assert sorted(m.values()) == sorted(set(names))
    assert all(k != v for k, v in m.items())
    assert m == ea.derangement(names, seed=13), "must be deterministic"


def test_derangement_two_images_swaps_them():
    assert ea.derangement(["a", "b"], seed=0) == {"a": "b", "b": "a"}


def test_derangement_needs_two_images():
    with pytest.raises(SystemExit):
        ea.derangement(["only"], seed=0)


def _items():
    return [Item(id="d1", language="en", prompt="Q?", image_name="a.jpg",
                 response="The car is red and big.",
                 labels=[{"start": 11, "end": 14, "prob": 1.0, "label": "mischaracterization"}]),
            Item(id="c1", language="en", prompt="Q?", image_name="b.jpg",
                 response="A plain answer.", labels=[])]


def _raw(p_dirty, g_dirty, p_clean, g_clean):
    n1 = len("The car is red and big.")
    bio1 = [0] * n1
    for k in range(11, 14):
        bio1[k] = 1 if k == 11 else 2
    n2 = len("A plain answer.")
    return {"d1": {"id": "d1", "p": [p_dirty] * n1, "g": g_dirty, "bio": bio1,
                   "targ": [1] * n1},
            "c1": {"id": "c1", "p": [p_clean] * n2, "g": g_clean, "bio": [0] * n2,
                   "targ": [0] * n2}}


def test_decode_all_gates_on_g_and_recovers_bio_spans():
    items = _items()
    raw = _raw(0.9, 0.95, 0.05, 0.1)
    out = sw.decode_all(raw, items, tau=0.0, g_thr=0.5)
    assert out["c1"] == []
    assert [(s["start"], s["end"]) for s in out["d1"]] == [(11, 14)]
    assert out["d1"][0]["label"] == "mischaracterization"
    # raise the gate above the dirty item's g -> everything abstains
    out = sw.decode_all(raw, items, tau=0.0, g_thr=0.99)
    assert out["d1"] == [] and out["c1"] == []


def test_postproc_min_len_drops_short_spans():
    items = _items()
    texts = {it.id: it.response for it in items}
    out = sw.decode_all(_raw(0.9, 0.95, 0.05, 0.1), items, 0.0, 0.5)
    assert sw.postproc(out, texts, min_len=3, merge_gap=2)["d1"]["labels"]
    assert sw.postproc(out, texts, min_len=4, merge_gap=2)["d1"]["labels"] == []


def test_score_is_perfect_for_a_perfect_decode():
    items = _items()
    gold = {it.id: {"id": it.id, "labels": it.labels, "text_len": len(it.response),
                    "image": it.image_name} for it in items}
    out = sw.decode_all(_raw(0.9, 0.95, 0.05, 0.1), items, 0.0, 0.5)
    _, m = sw.score(gold, out)
    assert m["IoU"] == pytest.approx(1.0)
    assert m["empty"] == pytest.approx(0.5)


def test_sweeps_end_to_end_on_a_synthetic_dump(tmp_path, capsys):
    items = _items()
    gold_path = tmp_path / "gold.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({"id": it.id, "language": "en", "prompt": it.prompt,
                                "image_name": it.image_name, "response": it.response,
                                "labels": it.labels}) + "\n")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for cond, raw in (("correct", _raw(0.9, 0.95, 0.05, 0.1)),
                      ("shuffled", _raw(0.6, 0.3, 0.05, 0.1))):     # gate closes -> abstain
        with open(raw_dir / f"{cond}.raw.jsonl", "w") as f:
            for r in raw.values():
                f.write(json.dumps(r) + "\n")
    json.dump({"tau": 0.0, "g_thr": 0.5}, open(raw_dir / "manifest.json", "w"))
    sys.argv = ["ablation_sweeps.py", "--raw_dir", str(raw_dir), "--gold", str(gold_path),
                "--transfer", "0.0,0.2,3,2", "--out", str(tmp_path / "rep.json")]
    sw.main()
    rep = json.load(open(tmp_path / "rep.json"))
    assert rep["conditions"]["correct"]["pp"]["IoU"] == pytest.approx(1.0)
    assert rep["conditions"]["shuffled"]["pp"]["IoU"] == pytest.approx(0.5)   # dirty missed
    assert rep["conditions"]["shuffled"]["delta_vs_correct_pp"]["iou"] == pytest.approx(-0.5)
    assert rep["conditions"]["shuffled"]["resweep"]["IoU"] == pytest.approx(1.0)  # g_thr<=0.3 recovers
    assert rep["postproc_grid"]["min_len=4,merge_gap=2"]["IoU"] == pytest.approx(0.5)
    assert rep["transfer"]["transferred"]["IoU"] == pytest.approx(1.0)
