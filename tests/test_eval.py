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
    assert abs(en["iou"] - 2 / 3) < 1e-9
    assert abs(en["predict_nothing_iou"] - 1 / 3) < 1e-9
    assert "pearson" in en and "spearman" in en
