#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/nag-nanochat}"
BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/nanochat}"

cd "$REPO_DIR"
mkdir -p "$BASE_DIR" "$BASE_DIR/base_checkpoints" "$BASE_DIR/base_data_climbmix" "$BASE_DIR/tokenizer"

export NANOCHAT_BASE_DIR="$BASE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --extra gpu

cat > "$BASE_DIR/env.sh" <<EOF
export NANOCHAT_BASE_DIR="$BASE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=1
EOF

echo "Wrote $BASE_DIR/env.sh"
echo
df -h "$BASE_DIR" /workspace || true
echo
nvidia-smi || true
echo
uv run python - <<'PY'
import importlib.util
import torch

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count", torch.cuda.device_count())
    print("gpu 0", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("hf_transfer importable", importlib.util.find_spec("hf_transfer") is not None)
try:
    import wandb
    print("wandb", wandb.__version__)
except Exception as e:
    print("wandb import failed", repr(e))
PY

echo
echo "Setup complete. For this shell, run:"
echo "source $BASE_DIR/env.sh"
