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
