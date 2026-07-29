"""Review-copy token alignment in the feature extractors.

The extractors embed the response twice — once as the candidate answer, once as a
"review copy" that ends the prompt — then locate the review copy by searching the
prompt's token ids for the tokenisation of the response, and map those tokens back to
character offsets in the response (gold spans are character offsets, so this mapping is
the supervision signal).

Two things have broken here, both regression-tested below:

1. `apply_chat_template` rstrips the message content, so a response with trailing
   whitespace no longer matches the tokens searched for (54/3799 en items).
2. Mapping tokens to characters by decoding each token and str.find()-ing it desyncs on
   any byte-fallback token that splits a multi-byte character, and never resyncs
   (32 fr/it/zh items, offsets up to 1.7x past the end of the response).

The stub tokenizer below is byte-level with a truthful offset mapping, so multi-byte
characters split across tokens exactly as they do in the real tokenizer — no model, no
download, no GPU.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "connector"))

import extract_features as fx  # noqa: E402


class StubTokenizer:
    """Byte-level ids with a real fast-tokenizer-style offset mapping.

    A multi-byte character becomes several ids that all map to its single character
    span — the shape that defeats decode-and-search mapping.
    """

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs = [], []
        for i, ch in enumerate(text):
            for b in ch.encode("utf-8"):
                ids.append(b)
                offs.append((i, i + 1))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


@pytest.fixture
def tok():
    return StubTokenizer()


def templated_ids(tok, prompt, response):
    """What the model actually sees: build_text, then the template's rstrip."""
    return tok.encode(fx.build_text(prompt, response).rstrip())


def locate(tok, prompt, response):
    """The extractor's alignment, lifted out of the model loop."""
    ids = templated_ids(tok, prompt, response)
    body = fx.review_body(response)
    ans_ids, offs = fx.encode_review(tok, body)
    pos, trim = fx.find_subseq(ids, ans_ids), 0
    while pos < 0 and trim < 3:
        trim += 1
        pos = fx.find_subseq(ids, ans_ids[trim:])
    if pos < 0:
        return None
    rt, tok_char = list(range(pos, pos + len(ans_ids[trim:]))), offs[trim:]
    while rt and not body[max(0, tok_char[0][0]):max(0, tok_char[0][1])].strip():
        rt.pop(0)
        tok_char = tok_char[1:]
    return body, rt, [(max(0, a), max(0, b)) for a, b in tok_char]


RESPONSES = [
    "A plain answer.",
    "Ends with one newline.\n",
    "Ends with four newlines.\n\n\n\n",                 # train-en-414
    "  Leading whitespace is kept.",
    "\n\nBoth ends.\n\n",
    "Answer with\n\ninterior blank lines.\n\n",
    "Le chien est noir, pas blanc — les éléments de la scène.",   # accented Latin
    "这张图片中观察到的细节。\n\n",                                    # CJK + trailing ws
    "Mixed 中文 and café ☕ emoji.\n",
]


@pytest.mark.parametrize("response", RESPONSES)
def test_review_copy_is_locatable(tok, response):
    assert locate(tok, "Is this a question?", response) is not None, \
        f"review copy not located for {response!r}"


@pytest.mark.parametrize("response", RESPONSES)
def test_offsets_are_in_range_and_monotone(tok, response):
    """Every token maps to a real character span of the ORIGINAL response."""
    body, rt, tok_char = locate(tok, "Q?", response)
    assert len(rt) == len(tok_char), "token/offset lists must stay in lockstep"
    assert tok_char, "no review tokens"
    for a, b in tok_char:
        assert 0 <= a <= b <= len(response), f"offset ({a},{b}) outside 0..{len(response)}"
    starts = [a for a, _ in tok_char]
    assert starts == sorted(starts), "offsets must be non-decreasing"
    assert tok_char[-1][1] == len(body), "the review span must reach the end of the body"


@pytest.mark.parametrize("response", RESPONSES)
def test_offsets_recover_the_response_text(tok, response):
    """The strongest check: slicing the response by the offsets rebuilds it."""
    body, _, tok_char = locate(tok, "Q?", response)
    covered = "".join(body[a:b] for a, b in tok_char)
    # tokens may repeat a character span (multi-byte splits), so compare the covered set
    assert body[tok_char[0][0]:tok_char[-1][1]] == body.lstrip()
    assert set(covered) <= set(body)


def test_located_span_is_the_last_copy(tok):
    """It must find the review copy at the end, not the candidate-answer copy."""
    response = "Repeated text.\n\n"
    ids = templated_ids(tok, "Q?", response)
    body = fx.review_body(response)
    ans_ids, _ = fx.encode_review(tok, body)
    pos = fx.find_subseq(ids, ans_ids)
    assert pos >= 0
    assert pos + len(ans_ids) == len(ids), "the review copy must end the prompt"


def test_review_body_only_strips_the_tail():
    """Leading whitespace must survive: gold spans are offsets into the raw response,
    so shifting the start would silently move every label."""
    assert fx.review_body("  keep my indent.\n\n") == "  keep my indent."
    assert fx.review_body("nothing to do.") == "nothing to do."


def test_multibyte_offsets_do_not_run_away(tok):
    """Regression for the fr/it/zh corruption: a CJK response whose tokens each split
    into several bytes must not push offsets past the end."""
    response = "这张图片中观察到的细节符合生物学特征。"
    body, rt, tok_char = locate(tok, "Q?", response)
    assert tok_char[-1][1] == len(body) == len(response)
    assert max(b for _, b in tok_char) <= len(response)
