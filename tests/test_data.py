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
