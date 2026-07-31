# SHROOM Vision Detector Final Protocol Report

Status: implementation fixes are staged locally; final training/submission is blocked by missing local artifacts.

## What Was Completed

| System | Official IoU | Cor | Cor_lbl | dirty | cleanOK | seed mean +/- std | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A2 deterministic | n/a | n/a | n/a | n/a | n/a | n/a | blocked: checkpoint not present |
| LoRA cascade | n/a | n/a | n/a | n/a | n/a | n/a | code implemented; training not run |
| Held-out winner | n/a | n/a | n/a | n/a | n/a | n/a | not evaluated |
| Final full-data | n/a | n/a | n/a | n/a | n/a | n/a | not submitted |

Implemented locally:

- deterministic LoRA train/eval mode control via `set_lora_training`;
- official scorer metrics in connector evaluation summaries;
- BIO diagnostics that account for both global gate and locator confidence nulling;
- shared `decode_spans` path for metrics and JSONL serialization;
- best-checkpoint reload before final dev prediction emission;
- `scripts/connector/predict_lora.py` fixed-threshold inference entrypoint;
- `scripts/connector/train_lora.py --cascade_train` for shared LoRA backbone with gate-on-all and dirty-only locator losses.

## Blockers

- Missing checkpoint: `results/lora_h100/a2/best_iou_lora_linear_f24_r16_s13.pt`.
- Missing local gold/test files named in the protocol, including `splits/dev.en.jsonl` and `../Shroom-Vision/distrib/shroom-vision.test.en.jsonl`.
- No CUDA GPU is available in the local environment.
- `git pull` was attempted before edits but could not write `.git/FETCH_HEAD` under the sandbox and was interrupted.

Held-out was not used.
