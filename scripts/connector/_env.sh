# Shared environment setup for the H100 runners. Sourced, not executed.
#
#   source scripts/connector/_env.sh
#   setup_cluster_env
#
# Activates .venv-cluster (creating it on first use) and guarantees every import the
# Gemma-4 feature extractor performs, so a missing package fails here with a fix rather
# than deep inside transformers' lazy-import machinery.

setup_cluster_env() {
  source .venv-cluster/bin/activate 2>/dev/null || {
    python3 -m venv .venv-cluster && source .venv-cluster/bin/activate
    pip install -q --upgrade pip
    local req=requirements/cluster.txt
    [ -f "$req" ] || req=requirements-cluster.txt
    pip install -q -r "$req" bitsandbytes accelerate
  }
  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA torch'"

  # Gemma-4's image processor runs `from torchvision.transforms.v2 import functional` at
  # module-import time. Without torchvision, AutoProcessor.from_pretrained dies with an
  # opaque "Could not import module 'Gemma4UnifiedProcessor'" that names neither the real
  # missing package nor the fix. Install against the SAME wheel index torch came from, and
  # pin torch to its exact local build, so resolving torchvision can never swap the CUDA
  # wheel underneath us.
  python -c "import torchvision" 2>/dev/null && return 0
  echo "[dep ] torchvision missing — required by the Gemma-4 image processor"
  local ver idx
  ver=$(python -c "import torch; print(torch.__version__)")
  idx=$(python - <<'PY'
import re, torch
m = re.search(r"\+(cu\d+|rocm[\d.]+|cpu)$", torch.__version__)
print(f"https://download.pytorch.org/whl/{m.group(1)}" if m else "")
PY
)
  local -a cmd=(pip install "torch==$ver" torchvision)
  [ -n "$idx" ] && cmd+=(--index-url "$idx")
  echo "[dep ] ${cmd[*]}"
  "${cmd[@]}" || {
    {
      echo "ERROR: torchvision is missing and could not be installed automatically."
      echo "       Run this yourself, from a shell with network access:"
      echo "         source .venv-cluster/bin/activate && ${cmd[*]}"
    } >&2
    exit 1
  }
  python -c "import torchvision; print('[dep ] torchvision', torchvision.__version__, 'OK')"
}
