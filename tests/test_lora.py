"""lora.py (hand-rolled LoRA) and the live-training mechanics of train_lora.py.

Everything runs on CPU with toy modules — no Gemma, no GPU. What must hold:

- injection wraps exactly the requested q/k/v/o projections of the requested layers;
- a wrapped module computes the base function at init (B = 0);
- gradients reach A and B, and freezing works;
- the LoRA checkpoint round-trips;
- v3_loss is finite, differentiable, and its gate-consistency term reacts to the gate;
- single_batch keeps the autograd graph (the reason collate can't be used live).
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "connector"))

from lora import (LoRALinear, inject_lora, lora_parameters,  # noqa: E402
                  lora_state_dict, load_lora_state, set_lora_training)
from model import Connector  # noqa: E402
import train_lora as tl  # noqa: E402


class Toy(nn.Module):
    """language_model.layers.N.self_attn.{q,k,v,o}_proj — the Gemma naming shape."""

    def __init__(self, n_layers=4, d=8):
        super().__init__()
        blocks = []
        for _ in range(n_layers):
            attn = nn.Module()
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attn, p, nn.Linear(d, d))
            layer = nn.Module()
            layer.self_attn = attn
            blocks.append(layer)
        lm = nn.Module()
        lm.layers = nn.ModuleList(blocks)
        self.language_model = lm


def test_inject_targets_only_requested_layers():
    m = Toy(n_layers=4)
    wrapped = inject_lora(m, from_layer=2)
    assert len(wrapped) == 2 * 4                       # layers 2,3 x q/k/v/o
    assert all(".layers.2." in w or ".layers.3." in w for w in wrapped)
    assert isinstance(m.language_model.layers[3].self_attn.q_proj, LoRALinear)
    assert isinstance(m.language_model.layers[1].self_attn.q_proj, nn.Linear)


def test_inject_range_empty_raises():
    with pytest.raises(ValueError):
        inject_lora(Toy(n_layers=4), from_layer=10)


def test_wrapped_equals_base_at_init():
    base = nn.Linear(8, 8)
    wrapped = LoRALinear(base, r=4).eval()
    x = torch.randn(3, 8)
    assert torch.allclose(wrapped(x), base(x), atol=1e-7), "B=0 must mean no change"


def test_bf16_base_gets_fp32_adapters_on_base_device():
    """The smoke-test crash: adapters born on CPU next to a cuda base. Device must
    follow the base; dtype must be fp32 with cast-in/cast-out around the matmul."""
    base = nn.Linear(8, 8).to(torch.bfloat16)
    w = LoRALinear(base, r=4, dropout=0.0).eval()
    assert w.A.device == base.weight.device and w.B.device == base.weight.device
    assert w.A.dtype == torch.float32
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    y = w(x)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()
    with torch.no_grad():
        w.B.add_(torch.randn_like(w.B))
    assert not torch.allclose(w(x).float(), base(x).float()), \
        "after B moves, the adapter must actually contribute"


def test_gradients_reach_A_and_B_and_base_stays_frozen():
    m = Toy(n_layers=2)
    inject_lora(m, from_layer=1, dropout=0.0)
    x = torch.randn(2, 8)
    y = m.language_model.layers[1].self_attn.q_proj(x)
    y.sum().backward()
    mod = m.language_model.layers[1].self_attn.q_proj
    assert mod.A.grad is not None and mod.B.grad is not None
    assert mod.base.weight.grad is None, "base weights must stay frozen"
    # B.grad is nonzero at init (dL/dB = scale * (A x) outer grad), A.grad is zero
    # until B moves — the standard LoRA asymmetry.
    assert mod.B.grad.abs().sum() > 0


def test_state_dict_roundtrip():
    m1, m2 = Toy(3), Toy(3)
    inject_lora(m1, from_layer=1)
    inject_lora(m2, from_layer=1)
    with torch.no_grad():
        for p in lora_parameters(m1):
            p.add_(torch.randn_like(p))
    n = load_lora_state(m2, lora_state_dict(m1))
    assert n == len(lora_parameters(m1))
    x = torch.randn(2, 8)
    q1 = m1.language_model.layers[1].self_attn.q_proj
    q2 = m2.language_model.layers[1].self_attn.q_proj
    with torch.no_grad():
        q2.base.load_state_dict(q1.base.state_dict())
    q1.eval(), q2.eval()
    assert torch.allclose(q1(x), q2(x), atol=1e-6)


def test_set_lora_training_makes_eval_outputs_deterministic():
    base = nn.Linear(8, 8)
    wrapped = LoRALinear(base, r=4, dropout=0.9)
    with torch.no_grad():
        wrapped.B.normal_()
    model = nn.Sequential(wrapped)
    x = torch.randn(4, 8)

    set_lora_training(model, True)
    assert wrapped.drop.training
    set_lora_training(model, False)
    assert not wrapped.drop.training
    ys = [model(x) for _ in range(3)]
    assert torch.equal(ys[0], ys[1])
    assert torch.equal(ys[1], ys[2])


# ------------------------------------------------------------------- live mechanics
class FakeItem:
    def __init__(self):
        self.id, self.language = "x1", "en"
        self.image_name, self.prompt = "img.jpg", "Q?"
        self.response = "A cat sits on the mat."
        self.labels = [{"start": 2, "end": 5, "prob": 0.7, "label": "invention"}]


def fake_live_example(T=6, P=3, L=4, D=16, requires_grad=True):
    it = FakeItem()
    H = torch.randn(T, L, D, requires_grad=requires_grad)
    V = torch.randn(P, L, D, requires_grad=requires_grad)
    n = len(it.response)
    step = max(1, n // T)
    tok_char = [(i * step, min(n, (i + 1) * step)) for i in range(T)]
    ex = tl.build_example(it, V, H, tok_char, n, max_chars=4000)
    return it, ex


class EvalItem:
    def __init__(self, i, dirty):
        self.id = f"train-en-{i}"
        self.language = "en"
        self.image_name = f"img{i}.jpg"
        self.prompt = "Q?"
        self.response = "abcd efgh ijkl"
        self.labels = ([{"start": 0, "end": 4, "prob": 1.0, "label": "invention"}]
                       if dirty else [])


class DropoutFeatureModel(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.adapter = LoRALinear(nn.Linear(d, d), r=2, dropout=0.85)
        with torch.no_grad():
            self.adapter.B.normal_()

    def forward(self, x):
        return self.adapter(x)


class FakeEvalBackbone:
    def __init__(self, T=7, L=2, D=8):
        self.model = DropoutFeatureModel(D)
        self.x = torch.randn(T, L, D)
        step = 2
        self.tok_char = [(i * step, min(len(EvalItem(0, True).response), (i + 1) * step))
                         for i in range(T)]

    def prepare(self, it, image_dir):
        return {"tok_char": self.tok_char, "answer_len": len(it.response)}

    def features(self, prep, grad=True):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            H = self.model(self.x)
        return H, H[:0]


def test_lora_eval_epoch_is_deterministic_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    bb = FakeEvalBackbone()
    dec = Connector(8, 2, dim=8, blocks=1, arch="linear")
    ck_path = tmp_path / "best.pt"
    torch.save({"model": dec.state_dict(), "lora": lora_state_dict(bb.model)}, ck_path)
    ck = torch.load(ck_path, map_location="cpu")
    dec.load_state_dict(ck["model"])
    load_lora_state(bb.model, ck["lora"])

    items = [EvalItem(1, True), EvalItem(2, False)]
    set_lora_training(bb.model, True)
    dec.train()
    runs = []
    for _ in range(3):
        per, metrics = tl.eval_epoch(bb, dec, items, "", "cpu", max_chars=100, batch=2)
        decoded = {}
        logits = {}
        for i, (it, q, p, t, g, bt, qsp) in per.items():
            decoded[i] = tl.decode_spans("bio", p, t, g, metrics["tau"], metrics["g_thr"],
                                         bt, qsp, resp_len=len(it.response))
            logits[i] = (q.copy(), p.copy(), t.copy(), g, bt.copy())
        runs.append((logits, decoded, metrics))

    for lhs, rhs in zip(runs, runs[1:]):
        for key in lhs[0]:
            assert np.array_equal(lhs[0][key][0], rhs[0][key][0])
            assert np.array_equal(lhs[0][key][1], rhs[0][key][1])
            assert np.array_equal(lhs[0][key][2], rhs[0][key][2])
            assert lhs[0][key][3] == rhs[0][key][3]
            assert np.array_equal(lhs[0][key][4], rhs[0][key][4])
        assert lhs[1] == rhs[1]
        assert lhs[2] == rhs[2]


def test_single_batch_keeps_graph():
    _, ex = fake_live_example()
    H, V, vmask, t2c, inpos, tgt = tl.single_batch(ex, "cpu")
    assert H.requires_grad and V.requires_grad, \
        "live features must keep their graph through build_example/single_batch"
    assert H.shape[0] == 1 and t2c.shape[0] == 1
    assert tgt["valid"].sum() > 0


def test_v3_loss_backward_reaches_features():
    _, ex = fake_live_example()
    H, V, vmask, t2c, inpos, tgt = tl.single_batch(ex, "cpu")
    dec = Connector(16, 4, dim=16, blocks=1, arch="connector")
    outs = dec(H, V, t2c, inpos, vmask)
    loss = tl.v3_loss(outs, tgt)
    assert torch.isfinite(loss)
    loss.backward()
    assert ex["H"].grad is not None and ex["H"].grad.abs().sum() > 0, \
        "gradient must flow through the features into the (LoRA) backbone"


def test_v3_loss_gate_term_reacts():
    """A dirty item (gate=1) must be penalised more when the gate logit says clean."""
    _, ex = fake_live_example()
    H, V, vmask, t2c, inpos, tgt = tl.single_batch(ex, "cpu")
    dec = Connector(16, 4, dim=16, blocks=1, arch="linear")
    with torch.no_grad():
        outs = [o.clone() for o in dec(H, V, t2c, inpos, vmask)]
    lo, hi = list(outs), list(outs)
    lo[3] = torch.full_like(outs[3], -5.0)     # gate logit: "clean"
    hi[3] = torch.full_like(outs[3], +5.0)     # gate logit: "dirty"
    assert tl.v3_loss(hi, tgt) < tl.v3_loss(lo, tgt), \
        "item HAS labels (gate=1): predicting clean must cost more"
