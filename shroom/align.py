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


def phrase_to_spans(phrase, response, fuzzy_ratio=0.8, fuzzy_len_cap=4.0):
    """Locate `phrase` inside `response`, returning list of (start, end) char spans.

    Tries exact (all occurrences), then whitespace/markdown-normalized, then fuzzy.
    A fuzzy match whose window is longer than `fuzzy_len_cap` times the phrase is
    rejected (guards against a paraphrase mapping to a huge span). Returns [] if
    nothing acceptable is found.
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
            if (last - first + 1) <= fuzzy_len_cap * len(nphr):
                return [(rmap[first], rmap[last] + 1)]
    return []
