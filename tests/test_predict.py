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
