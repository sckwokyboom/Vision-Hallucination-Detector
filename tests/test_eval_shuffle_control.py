"""The eval-time shuffled-image control must be able to DETECT V-dependence.

The H100 A1 run reported correct-image and shuffled-image metrics identical to ~1e-8.
Before believing "the model ignores V", the measuring instrument itself needs a
positive and a negative control:

- a connector FORCED to use V (alpha=1) must show materially different probabilities
  under run_eval(shuffle=True);
- a connector with alpha=0 must show bitwise-identical outputs (CPU is deterministic,
  so any difference would expose a bug such as shuffling the wrong tensor).

Synthetic CacheDS-shaped data, stock collate + run_eval — the exact code path the
cluster executed.
"""
import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "connector"))

from model import Connector  # noqa: E402
from train_connector import collate, run_eval  # noqa: E402

L, D = 4, 16


class It:
    def __init__(self, i):
        self.id = f"x{i}"
        self.language = "en"
        self.image_name = f"img{i}.jpg"
        self.prompt = "Q?"
        self.response = "word " * 10
        self.labels = [{"start": 0, "end": 8, "prob": 0.8, "label": "invention"}] \
            if i % 2 == 0 else []


def synth_ds(n=8, T=10, P=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        n_ch = 20
        t2c = np.repeat(np.arange(T), 2).astype(np.int64)[:n_ch]
        ex = dict(
            V=torch.randn(P, L, D, generator=g).numpy(),   # every item: DIFFERENT V
            H=torch.randn(T, L, D, generator=g).numpy(),
            t2c=t2c,
            inpos=np.zeros(n_ch, dtype=np.int64),
            y=np.array(([0.8] * 8 + [0.0] * 12) if i % 2 == 0 else [0.0] * n_ch,
                       dtype=np.float32),
            ytype=np.zeros((n_ch, 5), dtype=np.float32),
            bio=np.zeros(n_ch, dtype=np.int64),
            segs=[], gate=float(i % 2 == 0), item=It(i))
        out.append(ex)
    return out


class DS(torch.utils.data.Dataset):
    def __init__(self, ex):
        self.ex = ex

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def probs(model, dl, shuffle):
    per, m = run_eval(model, dl, "cpu", shuffle=shuffle, decoder="bio")
    p = np.concatenate([v[2] for v in per.values()])
    return p, m


@pytest.fixture
def dl():
    return torch.utils.data.DataLoader(DS(synth_ds()), batch_size=8,
                                       shuffle=False, collate_fn=collate)


def test_open_alpha_makes_shuffle_detectable(dl):
    """Positive control: with the visual branch OPEN, deranging V must change the
    per-char probabilities materially. If this fails, run_eval's shuffle is fake."""
    torch.manual_seed(0)
    model = Connector(D, L, dim=16, blocks=1, arch="connector")
    with torch.no_grad():
        model.alpha.fill_(1.0)
    p_ok, _ = probs(model, dl, shuffle=False)
    p_sh, _ = probs(model, dl, shuffle=True)
    assert np.abs(p_ok - p_sh).max() > 1e-4, \
        "alpha=1 yet shuffled V changed nothing: the control instrument is broken"


def test_closed_alpha_is_bitwise_identical(dl):
    """Negative control: with alpha=0 the shuffle must change NOTHING at all on CPU.
    (The 1e-8 wiggle seen on the H100 is GPU kernel nondeterminism, not V influence —
    this test pins that interpretation: deterministic hardware -> exact equality.)"""
    torch.manual_seed(0)
    model = Connector(D, L, dim=16, blocks=1, arch="connector")   # alpha starts at 0
    p_ok, m_ok = probs(model, dl, shuffle=False)
    p_sh, m_sh = probs(model, dl, shuffle=True)
    assert np.array_equal(p_ok, p_sh), \
        "alpha=0 but outputs differ: something besides alpha*fuse leaks V"
    assert m_ok["span_iou"] == m_sh["span_iou"]


def test_small_alpha_scales_the_gap(dl):
    """The gap must scale with alpha — tiny alpha, tiny gap. This is what makes the
    cluster's 1e-8 gap DIAGNOSTIC of |alpha| rather than of a broken measurement."""
    gaps = {}
    for a in (1e-4, 1e-1):
        torch.manual_seed(0)
        model = Connector(D, L, dim=16, blocks=1, arch="connector")
        with torch.no_grad():
            model.alpha.fill_(a)
        p_ok, _ = probs(model, dl, shuffle=False)
        p_sh, _ = probs(model, dl, shuffle=True)
        gaps[a] = float(np.abs(p_ok - p_sh).max())
    assert gaps[1e-1] > gaps[1e-4] * 10, f"gap must grow with alpha: {gaps}"
    assert gaps[1e-4] < 1e-3, f"tiny alpha must mean tiny gap: {gaps}"
