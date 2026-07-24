from shroom.metrics import (char_iou, gold_char_probs, pearson, spearman,
                            calibration, trivial_baselines)
from shroom.data import load_jsonl

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
    assert abs(char_iou(G, pred, RESP_LEN) - 6 / 14) < 1e-9


def test_gold_char_probs_uses_max_over_covering_spans():
    probs = gold_char_probs(G, RESP_LEN)
    assert probs[13] == 0.3333 and probs[18] == 0.3333
    assert probs[35] == 0.6667
    assert probs[0] == 0.0
    assert len(probs) == RESP_LEN


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
    assert abs(tb["predict_nothing_iou"] - 1 / 3) < 1e-9
    # predict-all: t-1 -> 14/43, t-2 -> 5/21, t-3 -> 0 ; mean of the three
    exp = (14 / 43 + 5 / 21 + 0.0) / 3
    assert abs(tb["predict_all_iou"] - exp) < 1e-9
