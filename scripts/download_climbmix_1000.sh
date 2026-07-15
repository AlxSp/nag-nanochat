#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
BASE_DIR="${NANOCHAT_BASE_DIR:-/workspace/nanochat}"
NUM_WORKERS="${NUM_WORKERS:-16}"

cd "$REPO_DIR"
export NANOCHAT_BASE_DIR="$BASE_DIR"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

mkdir -p "$BASE_DIR/base_data_climbmix"

echo "Downloading exactly 1000 train shards plus validation shard_06542 into:"
echo "$BASE_DIR/base_data_climbmix"
uv run python -m nanochat.dataset -n 1000 -w "$NUM_WORKERS"

DATA_DIR="$BASE_DIR/base_data_climbmix"
COUNT="$(find "$DATA_DIR" -maxdepth 1 -name '*.parquet' | wc -l)"
echo "Parquet file count: $COUNT"
test "$COUNT" -eq 1001
test -f "$DATA_DIR/shard_00000.parquet"
test -f "$DATA_DIR/shard_00999.parquet"
test -f "$DATA_DIR/shard_06542.parquet"

du -sh "$DATA_DIR"
echo "Dataset verified: 1000 train shards + shard_06542 validation"
