from collections import Counter
from .align import phrase_to_spans


def aggregate(samples, response, tau=0.5):
    """Turn N parsed samples into (spans, per_char_prob).

    samples: list where each element is either a list of {phrase, label} dicts,
             or None (unparseable — excluded from the denominator).
    per_char_prob[i] = fraction of *valid* (non-None) samples whose spans cover char i.
    spans: contiguous runs of per_char_prob >= tau, with mean prob and majority label,
           as {start, end, prob, label} dicts (gold schema).
    """
    resp_len = len(response)
    hits = [0] * resp_len
    label_votes = [Counter() for _ in range(resp_len)]
    valid = 0

    for parsed in samples:
        if parsed is None:
            continue
        valid += 1
        covered = {}
        for entry in parsed:
            for a, b in phrase_to_spans(entry["phrase"], response):
                for i in range(a, b):
                    if i not in covered:
                        covered[i] = entry["label"]
        for i, label in covered.items():
            hits[i] += 1
            label_votes[i][label] += 1

    denom = valid if valid > 0 else 1
    per_char_prob = [hits[i] / denom for i in range(resp_len)]

    spans, i = [], 0
    while i < resp_len:
        if per_char_prob[i] >= tau:
            j = i
            while j < resp_len and per_char_prob[j] >= tau:
                j += 1
            run_prob = sum(per_char_prob[i:j]) / (j - i)
            votes = Counter()
            for k in range(i, j):
                votes.update(label_votes[k])
            label = votes.most_common(1)[0][0] if votes else "other"
            spans.append({"start": i, "end": j,
                          "prob": round(run_prob, 4), "label": label})
            i = j
        else:
            i += 1
    return spans, per_char_prob
