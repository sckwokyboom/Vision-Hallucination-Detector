import hashlib


def _bucket(image_name, seed):
    h = hashlib.md5(f"{seed}:{image_name}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def group_split_by_image(items, dev_frac=0.1, seed=13):
    """Assign every item to train/dev by hashing its image_name.

    All items sharing an image land in the same split (no leakage). Deterministic.
    Returns (train_ids, dev_ids).
    """
    train_ids, dev_ids = [], []
    for it in items:
        if _bucket(it.image_name, seed) < dev_frac:
            dev_ids.append(it.id)
        else:
            train_ids.append(it.id)
    return train_ids, dev_ids
