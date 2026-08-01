"""Fixed-threshold inference for LoRA connector checkpoints.

This is intentionally separate from training/eval sweeps: it restores one checkpoint,
uses frozen thresholds, and writes official-format JSONL with only `id` and `labels`.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shroom.data import load_jsonl  # noqa: E402
from model import Connector  # noqa: E402
from train_connector import build_example, decode_spans  # noqa: E402
from train_lora import LiveBackbone, single_batch  # noqa: E402
from lora import inject_lora, load_lora_state, set_lora_training  # noqa: E402


def _config_value(args, cfg, ck_args, key, default=None):
    val = getattr(args, key, None)
    if val is not None:
        return val
    if key in cfg:
        return cfg[key]
    thresholds = cfg.get("thresholds") or {}
    if key in thresholds:
        return thresholds[key]
    return ck_args.get(key, default)


def _parse_layers(value):
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x]
    return [int(x) for x in value]


def _model_hidden_size(model):
    cfg = model.config
    return cfg.text_config.hidden_size if hasattr(cfg, "text_config") else cfg.hidden_size


def _infer_one(bb, dec, item, image_dir, device, max_chars, decoder, tau, gate_threshold):
    prep = bb.prepare(item, image_dir)
    if prep is None:
        return []
    H, V = bb.features(prep, grad=False)
    ex = build_example(item, V, H, prep["tok_char"], prep["answer_len"], max_chars=max_chars)
    Hb, Vb, vmask, t2c, inpos, tgt = single_batch(ex, device)
    ql, pl, tl, gl, bl, _ = dec(Hb, Vb, t2c, inpos, vmask)
    n = int(tgt["valid"][0].sum().item())
    p = torch.sigmoid(pl)[0, :n].cpu().numpy()
    t = torch.sigmoid(tl)[0, :n].cpu().numpy()
    g = float(torch.sigmoid(gl)[0].cpu())
    bt = bl.argmax(-1)[0, :n].cpu().numpy()
    spans = decode_spans(decoder, p, t, g, tau, gate_threshold, bt, None,
                         resp_len=len(item.response))
    return spans, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--gate_threshold", type=float, default=None)
    ap.add_argument("--decoder", choices=["simple", "gate", "hyst", "gate_hyst", "v2", "bio"],
                    default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--max_chars", type=int, default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8")) if args.config else {}
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    ck_args = ck.get("args") or {}
    ck_metrics = ck.get("metrics") or {}

    model_id = _config_value(args, cfg, ck_args, "model_id")
    if not model_id:
        raise SystemExit("--model_id is required unless checkpoint args or --config provide it")
    layers = _parse_layers(cfg.get("layers", ck_args.get("layers", "24,32,40,47")))
    quant = args.quant or cfg.get("quant") or cfg.get("dtype") or ck_args.get("quant", "bf16")
    decoder = args.decoder or cfg.get("decoder") or ck_args.get("decoder", "bio")
    tau = args.tau
    if tau is None:
        tau = cfg.get("tau", (cfg.get("thresholds") or {}).get("tau", ck_metrics.get("tau")))
    gate_threshold = args.gate_threshold
    if gate_threshold is None:
        thresholds = cfg.get("thresholds") or {}
        gate_threshold = cfg.get("gate_threshold",
                                 thresholds.get("gate_threshold",
                                                thresholds.get("g_thr", ck_metrics.get("g_thr"))))
    if tau is None or gate_threshold is None:
        raise SystemExit("fixed --tau and --gate_threshold are required unless stored in checkpoint/config")

    max_chars = int(_config_value(args, cfg, ck_args, "max_chars", 4000))
    bb = LiveBackbone(model_id, device, layers, max_side=int(ck_args.get("max_side", 768)),
                      quant=quant)
    inject_lora(bb.model, from_layer=int(ck_args.get("lora_from", cfg.get("lora_from", 24))),
                r=int(ck_args.get("lora_r", cfg.get("lora_r", 16))),
                alpha=int(ck_args.get("lora_alpha", cfg.get("lora_alpha", 32))),
                dropout=float(ck_args.get("lora_dropout", cfg.get("lora_dropout", 0.0))))
    if "lora" not in ck:
        raise SystemExit(f"{args.checkpoint} has no LoRA adapter state")
    load_lora_state(bb.model, ck["lora"])
    set_lora_training(bb.model, False)

    dec = Connector(_model_hidden_size(bb.model), len(layers),
                    dim=int(ck_args.get("dim", cfg.get("dim", 768))),
                    blocks=int(ck_args.get("blocks", cfg.get("blocks", 3))),
                    arch=ck_args.get("arch", cfg.get("arch", "linear"))).to(device)
    dec.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    dec.eval()
    bb.model.eval()

    items = load_jsonl(args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with torch.no_grad(), open(args.output, "w", encoding="utf-8") as f:
        for item in items:
            out = _infer_one(bb, dec, item, args.image_dir, device, max_chars,
                             decoder, float(tau), float(gate_threshold))
            spans, probs = out if isinstance(out, tuple) else (out, [])
            # char_probs power downstream ensembling; harmless extra field otherwise
            f.write(json.dumps({"id": item.id, "labels": spans,
                                "char_probs": [round(float(x), 3) for x in probs]},
                               ensure_ascii=False) + "\n")

    print(f"wrote {len(items)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
