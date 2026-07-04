#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/nag-nanochat}"
cd "$REPO_DIR"
source runs/h100_64x640.env

RUN_NAME="${RUN_NAME:-gpt_d64_w640_3e19}"
MODEL_TAG="${MODEL_TAG:-$RUN_NAME}"
SCREEN_NAME="${SCREEN_NAME:-train_${RUN_NAME}}"
RUN_IN_SCREEN="${RUN_IN_SCREEN:-1}"

if [ "$RUN_IN_SCREEN" = "1" ] && [ -z "${INSIDE_TRAIN_SCREEN:-}" ]; then
  command -v screen >/dev/null 2>&1 || { echo "screen is not installed. Set RUN_IN_SCREEN=0 or install screen."; exit 1; }
  echo "Starting GPT baseline training in screen session: $SCREEN_NAME"
  echo "Attach with: screen -r $SCREEN_NAME"
  INSIDE_TRAIN_SCREEN=1 screen -dmS "$SCREEN_NAME" bash "$0"
  exit 0
fi

test -d "$NANOCHAT_BASE_DIR/tokenizer"
test -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
test -f "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt"
test -f "$NANOCHAT_BASE_DIR/base_data_climbmix/shard_00000.parquet"
test -f "$NANOCHAT_BASE_DIR/base_data_climbmix/shard_00999.parquet"
test -f "$NANOCHAT_BASE_DIR/base_data_climbmix/shard_06542.parquet"

uv run python -m scripts.check_fa3

CMD=(
  uv run torchrun --standalone "--nproc_per_node=$NPROC_PER_NODE" -m scripts.base_train --
  --arch=gpt
  "--depth=$DEPTH"
  "--model-dim=$MODEL_DIM"
  "--head-dim=$HEAD_DIM"
  "--window-pattern=$WINDOW_PATTERN"
  "--target-flops=$TARGET_FLOPS"
  "--target-param-data-ratio=$TARGET_PARAM_DATA_RATIO"
  "--device-batch-size=$DEVICE_BATCH_SIZE"
  "--total-batch-size=$TOTAL_BATCH_SIZE"
  "--matrix-lr=$MATRIX_LR"
  "--embedding-lr=$EMBEDDING_LR"
  "--unembedding-lr=$UNEMBEDDING_LR"
  "--scalar-lr=$SCALAR_LR"
  "--eval-every=$EVAL_EVERY"
  "--save-every=$SAVE_EVERY"
  "--sample-every=$SAMPLE_EVERY"
  "--core-metric-every=$CORE_METRIC_EVERY"
  "--model-tag=$MODEL_TAG"
  "--run=$RUN_NAME"
)

printf 'Launching command:\n'
printf '%q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"

