import json
from dataclasses import dataclass, field


@dataclass
class Item:
    id: str
    language: str
    prompt: str
    image_name: str
    response: str
    labels: list = field(default_factory=list)  # list[dict]: start, end, prob, label
    split: str = ""


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(Item(
                id=d["id"],
                language=d.get("language", ""),
                prompt=d.get("prompt", ""),
                image_name=d.get("image_name", ""),
                response=d.get("response", ""),
                labels=d.get("labels") or [],
                split=d.get("split", ""),
            ))
    return items
