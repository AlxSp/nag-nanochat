#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_DIR"
source runs/h100_64x640.env

test -d "$NANOCHAT_BASE_DIR/tokenizer"
test -f "$NANOCHAT_BASE_DIR/base_data_climbmix/shard_00000.parquet"
test -f "$NANOCHAT_BASE_DIR/base_data_climbmix/shard_06542.parquet"

uv run python -m scripts.check_fa3

CMD=(
  uv run torchrun --standalone "--nproc_per_node=$NPROC_PER_NODE" -m scripts.base_train --
  --arch=nag-gpt
  "--depth=$DEPTH"
  "--model-dim=$MODEL_DIM"
  "--head-dim=$HEAD_DIM"
  "--window-pattern=$WINDOW_PATTERN"
  --num-iterations=1
  --target-param-data-ratio=-1
  "--device-batch-size=$DEVICE_BATCH_SIZE"
  "--total-batch-size=$TOTAL_BATCH_SIZE"
  "--matrix-lr=$MATRIX_LR"
  "--embedding-lr=$EMBEDDING_LR"
  "--unembedding-lr=$UNEMBEDDING_LR"
  "--scalar-lr=$SCALAR_LR"
  --eval-every=-1
  --save-every=999999
  --sample-every=-1
  --core-metric-every=-1
  --model-tag=smoke_nag_gpt_d64_w640
  --run=dummy
)

printf 'Launching smoke command:\n'
printf '%q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
