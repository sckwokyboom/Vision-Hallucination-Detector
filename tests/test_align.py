from shroom.align import phrase_to_spans

R = "No, it has a hidden bracket on the ceiling."


def test_exact_single():
    assert phrase_to_spans("hidden", R) == [(13, 19)]


def test_all_occurrences():
    r = "cat cat"
    assert phrase_to_spans("cat", r) == [(0, 3), (4, 7)]


def test_normalized_markdown():
    r = "It is a **hidden** bracket."
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
