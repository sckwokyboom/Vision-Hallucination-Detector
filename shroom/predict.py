import json
import re

from .aggregate import aggregate

CATEGORIES = ["invention", "mischaracterization", "OCR", "miscounting", "other"]

PROMPT_TEMPLATE = (
    "Look at the image. A user asked a question about it and a model produced the answer below.\n"
    "Find every span of the answer that is a hallucination — content NOT supported by the image.\n"
    "Classify each span into exactly one category:\n"
    "  - invention: entities, objects, properties or events not present in the image\n"
    "  - mischaracterization: incorrect description of content that IS visible\n"
    "  - OCR: misreading of text visible in the image\n"
    "  - miscounting: wrong quantity of visible items\n"
    "  - other: a hallucination that fits none of the above\n\n"
    "Output ONLY a JSON array. Each element: "
    '{"phrase": "<exact substring copied verbatim from the answer>", "label": "<category>"}. '
    "Copy the phrase EXACTLY, character for character. "
    "If the answer is fully correct, output []."
)


def build_prompt(prompt, response):
    return (f"{PROMPT_TEMPLATE}\n\n"
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
