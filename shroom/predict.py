import json
import re

from .aggregate import aggregate

CATEGORIES = ["invention", "mischaracterization", "OCR", "miscounting", "other"]

PROMPT_TEMPLATE = (
    "You are a strict fact-checker. The IMAGE is the ground truth.\n"
    "Below is a QUESTION about the image and an ANSWER produced by another AI model.\n"
    "The answer OFTEN contains hallucinations: content NOT supported by — or contradicting — "
    "the image. Be skeptical: check every claim in the answer against the image and flag each part "
    "the image does not support.\n"
    "Categories:\n"
    "  - invention: objects, properties, details or events not present in the image\n"
    "  - mischaracterization: incorrect description of something that IS visible\n"
    "  - OCR: misreading of text visible in the image\n"
    "  - miscounting: wrong quantity of visible items\n"
    "  - other: a hallucination that fits none of the above\n\n"
    "For each hallucination, copy the offending text VERBATIM from the answer (exact characters, "
    "spelling and punctuation). Prefer the shortest phrase that captures the error, but flagging a "
    "slightly longer span is better than missing a hallucination.\n"
    'Return ONLY a JSON array; each element {"phrase": "<verbatim substring>", "label": "<category>"}. '
    "Only if every claim is fully supported by the image, return []."
)


def build_prompt(prompt, response):
    return (f"{PROMPT_TEMPLATE}\n\n"
            f'Question: "{prompt}"\n'
            f'Answer: "{response}"\n'
            "Output:")


# Original starter prompt (label_with_gemma.py) — 3 of 5 categories, permissive framing.
# Kept for measuring the starter solution on the same benchmark.
LABELING_PROMPT_ORIGINAL = (
    "Look at the image. A user asked a question about it and a model produced the answer below. "
    "Your task: check if the answer contains hallucinations (factual errors, miscounting, "
    "or invented details inconsistent with what's visible in the image).\n\n"
    "Output a JSON array of hallucinated phrases found in the answer. "
    'Each entry: {"phrase": "exact substring from the answer", '
    '"label": "mischaracterization"|"miscounting"|"invention"}. '
    "Quote the phrase EXACTLY as it appears — copy-paste characters. "
    "If the answer is fully correct, output [].\n"
    "Output ONLY the JSON array, nothing else."
)


def build_prompt_original(prompt, response):
    return (f"{LABELING_PROMPT_ORIGINAL}\n\n"
            f'Question: "{prompt}"\n'
            f'Answer: "{response}"\n'
            "Output:")


def _coerce(parsed):
    if not isinstance(parsed, list):
        return None
    out = []
    for e in parsed:
        if isinstance(e, dict) and "phrase" in e:
            label = e.get("label", "other")
            if label not in CATEGORIES:
                label = "other"
            out.append({"phrase": str(e["phrase"]), "label": label})
    return out


def parse_output(raw):
    """Parse a raw model output into [{phrase, label}], or None if unparseable."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```")).strip()
    try:
        return _coerce(json.loads(raw))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        return _coerce(json.loads(m.group(0)))
    except json.JSONDecodeError:
        return None


def predict_item(backend, item, image, n=5, temperature=0.5, tau=0.5):
    """Run one item through the VLM backend and aggregate into (spans, per_char_prob)."""
    text = build_prompt(item.prompt, item.response)
    raws = backend.generate(image, text, n=n, temperature=temperature)
    samples = [parse_output(r) for r in raws]
    return aggregate(samples, item.response, tau=tau)
