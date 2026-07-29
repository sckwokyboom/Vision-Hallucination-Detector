# Data

This repository does **not** ship the SHROOM-visions dataset (it has its own terms and a
closed test set). Obtain it from the shared task and place it as described below.

## Get the data

Automated path — downloads, verifies and unpacks everything into the layout below:

```bash
bash scripts/get_data.sh                      # or --data-dir /scratch/$USER/Shroom-Vision
```

Idempotent and resumable (`curl -C -`); `--no-images` fetches the 16 MB of annotations only,
`--force` re-fetches. On a cluster run it on a **login node** — compute nodes usually have no
outbound network. The archives are kept, so re-extracting later needs no network.

Manual path, if you already have the archives or the automated one is blocked:

SHROOM-visions 2026 shared task: <https://helsinki-nlp.github.io/shroom/2026>

You need two things:

1. **The JSONL splits** (`train.<lang>.labeled.jsonl`, `test.<lang>.unlabeled.jsonl` for
   `en`, `fr`, `it`, `zh`).
2. **The image archive** (`shroom-visions-images.tar.gz`).

## Expected layout

The scripts default to a sibling `Shroom-Vision/` directory (all paths are overridable via
CLI flags such as `--input`, `--image_dir`, `--train`):

```
<parent>/
├── Vision-Hallucination-Detector/   # this repo
└── Shroom-Vision/
    ├── distrib/
    │   ├── shroom-vision.train.en.labeled.jsonl
    │   ├── shroom-vision.test.en.unlabeled.jsonl
    │   └── ... (fr, it, zh)
    ├── shroom-visions-images.tar.gz
    └── images/                       # created by scripts/extract_images.sh
```

Extract the images once:

```bash
bash scripts/extract_images.sh ../Shroom-Vision/shroom-visions-images.tar.gz ../Shroom-Vision/images
```

## Data schema (one line of a labeled split)

```json
{
  "id": "train-en-413", "language": "en",
  "prompt": "Does this desk have legs? Please elaborate.",
  "image_name": "desk_....jpg",
  "response": "No, this desk does not have legs...",
  "labels": [{"start": 148, "end": 154, "prob": 0.33, "label": "mischaracterization"}]
}
```

`start`/`end` are character offsets into `response`; `prob` is the multi-annotator agreement
fraction for that span; `label` ∈ {invention, mischaracterization, OCR, miscounting, other}.
