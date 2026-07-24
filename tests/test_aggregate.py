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
    assert abs(prob[13] - 2 / 3) < 1e-9      # covered in 2 of 3 valid samples
    assert prob[0] == 0.0
    assert len(prob) == len(R)


def test_span_built_above_threshold():
    spans, _ = aggregate(_samples(), R, tau=0.5)
    assert len(spans) == 1
    s = spans[0]
    assert (s["start"], s["end"]) == (13, 19)
    assert s["label"] == "mischaracterization"
    assert abs(s["prob"] - 2 / 3) < 1e-3


def test_threshold_suppresses_low_prob():
    spans, _ = aggregate(_samples(), R, tau=0.7)   # 0.667 < 0.7
    assert spans == []


def test_none_samples_ignored_in_denominator():
    samples = [[{"phrase": "hidden", "label": "invention"}],
               [{"phrase": "hidden", "label": "invention"}],
               None]
    spans, prob = aggregate(samples, R, tau=0.5)
    assert abs(prob[13] - 1.0) < 1e-9


def test_all_none_yields_nothing():
    spans, prob = aggregate([None, None], R, tau=0.5)
    assert spans == [] and set(prob) == {0.0}
