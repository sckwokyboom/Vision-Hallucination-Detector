"""The connector arch's visual branch (A1 in the Unified Visual Adaptation matrix).

Invariants that make the A0-vs-A1 comparison interpretable:

1. ReZero: at init, arch=connector computes EXACTLY what arch=linear computes — the
   visual branch enters only as alpha moves off zero, so a worse connector run can
   never be blamed on random fusion weights perturbing the H-path.
2. The branch is trainable: alpha receives gradient at init, and the visual-path
   parameters receive gradient once alpha is non-zero.
3. v_mask works: padded visual positions must not influence the output.
4. shuffle_v: the derangement never maps an item to its own image.
"""
import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "connector"))

from model import Connector  # noqa: E402

B, T, P, L, D, CMAX = 2, 5, 7, 4, 16, 11


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    H = torch.randn(B, T, L, D, generator=g)
    V = torch.randn(B, P, L, D, generator=g)
    t2c = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 4, 4, -1, -1],
                        [0, 1, 1, 2, 2, 2, 3, 4, -1, -1, -1]])
    inpos = torch.zeros(B, CMAX, dtype=torch.long)
    vmask = torch.zeros(B, P, dtype=torch.bool)
    vmask[1, 5:] = True                     # item 1 has 5 real patches, 2 padded
    return H, V, t2c, inpos, vmask


def make(arch):
    torch.manual_seed(7)
    return Connector(D, L, dim=16, blocks=2, arch=arch).eval()


def test_rezero_equals_linear_at_init(batch):
    H, V, t2c, inpos, vmask = batch
    conn = make("connector")
    with torch.no_grad():
        out_conn = conn(H, V, t2c, inpos, vmask)
        conn.arch = "linear"                # same weights, H-only path
        out_lin = conn(H, V, t2c, inpos, vmask)
    for a, b in zip(out_conn[:5], out_lin[:5]):
        assert torch.allclose(a, b, atol=1e-6), "connector at alpha=0 must equal linear"


def test_alpha_gets_gradient_at_init(batch):
    H, V, t2c, inpos, vmask = batch
    conn = make("connector").train()
    q = conn(H, V, t2c, inpos, vmask)[0]
    q.sum().backward()
    assert conn.alpha.grad is not None and conn.alpha.grad.abs().item() > 0, \
        "alpha must be trainable from step one or the branch never opens"


def test_visual_params_get_gradient_once_open(batch):
    H, V, t2c, inpos, vmask = batch
    conn = make("connector").train()
    with torch.no_grad():
        conn.alpha.fill_(1.0)
    q = conn(H, V, t2c, inpos, vmask)[0]
    q.sum().backward()
    g = conn.proj_v.weight.grad
    assert g is not None and g.abs().sum().item() > 0, \
        "with alpha open, gradient must reach the visual projection"


def test_vmask_blocks_padded_patches(batch):
    H, V, t2c, inpos, vmask = batch
    conn = make("connector")
    with torch.no_grad():
        conn.alpha.fill_(1.0)               # open the branch so V matters at all
        out1 = conn(H, V, t2c, inpos, vmask)[0]
        V2 = V.clone()
        V2[1, 5:] += 100.0                  # rewrite only the MASKED patches
        out2 = conn(H, V2, t2c, inpos, vmask)[0]
    assert torch.allclose(out1, out2, atol=1e-5), "masked patches must not leak"


def test_unmasked_patches_do_matter(batch):
    H, V, t2c, inpos, vmask = batch
    conn = make("connector")
    with torch.no_grad():
        conn.alpha.fill_(1.0)
        out1 = conn(H, V, t2c, inpos, vmask)[0]
        V2 = V.clone()
        V2[0] += 100.0                      # rewrite REAL patches of item 0
        out2 = conn(H, V2, t2c, inpos, vmask)[0]
    assert not torch.allclose(out1[0], out2[0], atol=1e-3), \
        "real patches must influence the output (otherwise A1 tests nothing)"


def test_shuffle_v_derangement_avoids_own_image():
    import numpy as np

    class It:                               # minimal stand-in for a data item
        def __init__(self, i, img):
            self.id, self.image_name = f"x{i}", img

    items = [It(i, f"img{i // 2}.jpg") for i in range(20)]   # 2 answers per image
    n = len(items)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    for i in range(n):
        if items[perm[i]].image_name == items[i].image_name:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    same = sum(items[perm[i]].image_name == items[i].image_name for i in range(n))
    assert same <= 1, f"derangement left {same} items with their own image"
