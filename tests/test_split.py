from shroom.data import load_jsonl
from shroom.split import group_split_by_image


def test_split_is_deterministic(mini_path):
    items = load_jsonl(mini_path)
    a = group_split_by_image(items, dev_frac=0.5, seed=13)
    b = group_split_by_image(items, dev_frac=0.5, seed=13)
    assert a == b


def test_no_image_spans_both_splits(mini_path):
    items = load_jsonl(mini_path)
    train_ids, dev_ids = group_split_by_image(items, dev_frac=0.5, seed=13)
    id2img = {it.id: it.image_name for it in items}
    train_imgs = {id2img[i] for i in train_ids}
    dev_imgs = {id2img[i] for i in dev_ids}
    assert train_imgs.isdisjoint(dev_imgs)      # desk.jpg (t-1,t-3) never split apart


def test_split_covers_all_items(mini_path):
    items = load_jsonl(mini_path)
    train_ids, dev_ids = group_split_by_image(items, dev_frac=0.5, seed=13)
    assert set(train_ids) | set(dev_ids) == {it.id for it in items}
    assert set(train_ids).isdisjoint(dev_ids)
