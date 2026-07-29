"""Minimal LoRA for the Unified Visual Adaptation experiment (A2/A3).

Hand-rolled instead of peft on purpose: ~60 lines buys exact control over which
layers are adapted (the tap-layer interaction below), a checkpoint that is just the
A/B matrices, and full testability on CPU with a toy module tree.

Layer targeting matters here more than usual: the decoder reads hidden states tapped
at layers {24, 32, 40, 47}, so LoRA restricted to layers 40+ leaves the 24/32 taps
frozen by construction. The default (from=24) adapts everything the decoder sees;
pass --lora_from 40 for the top-8-only configuration.
"""
import re

import torch
import torch.nn as nn

TARGETS = re.compile(r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj$")


class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B(A(dropout(x))). B starts at zero, so at init the
    wrapped module computes exactly what the frozen base computes."""

    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        # On the base's device (a CPU-born adapter next to a cuda base was the first
        # smoke-test crash), and in fp32 regardless of the base dtype: AdamW moments in
        # bf16 quantize away ~3 mantissa bits, and at r=16 fp32 costs nothing.
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features,
                                          dtype=torch.float32, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r,
                                          dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x):
        lo = self.drop(x).to(self.A.dtype) @ self.A.t() @ self.B.t() * self.scale
        return self.base(x) + lo.to(x.dtype)


def inject_lora(model, from_layer=24, to_layer=10 ** 9, r=16, alpha=32, dropout=0.05):
    """Wrap every q/k/v/o projection of self_attn in layers [from_layer, to_layer].
    Matches by module path, so it works for any prefix (language_model.layers.N...).
    Returns the wrapped module names."""
    wrapped = []
    for name, mod in list(model.named_modules()):
        m = TARGETS.search(name)
        if not m or not isinstance(mod, nn.Linear):
            continue
        if not (from_layer <= int(m.group(1)) <= to_layer):
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1],
                LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
        wrapped.append(name)
    if not wrapped:
        raise ValueError(f"no layers.N.self_attn.[qkvo]_proj found in [{from_layer}, {to_layer}]")
    return wrapped


def lora_parameters(model):
    return [p for mod in model.modules() if isinstance(mod, LoRALinear)
            for p in (mod.A, mod.B)]


def lora_state_dict(model):
    return {n: p for n, p in model.state_dict().items()
            if n.endswith((".A", ".B")) and ".base." not in n.rsplit(".", 1)[0]}


def load_lora_state(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in unexpected]
    if bad:
        raise ValueError(f"unexpected LoRA keys: {bad[:5]}")
    loaded = {k for k in state}
    want = {n for n, _ in model.named_parameters() if n.endswith((".A", ".B"))}
    if not loaded <= want:
        raise ValueError("LoRA state does not match injected modules")
    return len(loaded)
