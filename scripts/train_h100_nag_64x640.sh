#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/nag-nanochat}"
cd "$REPO_DIR"
source runs/h100_64x640.env

RUN_NAME="${RUN_NAME:-nag_gpt_d64_w640_3e19_gatefix}"
MODEL_TAG="${MODEL_TAG:-$RUN_NAME}"
SCREEN_NAME="${SCREEN_NAME:-train_${RUN_NAME}}"
RUN_IN_SCREEN="${RUN_IN_SCREEN:-1}"

if [ "$RUN_IN_SCREEN" = "1" ] && [ -z "${INSIDE_TRAIN_SCREEN:-}" ]; then
  if command -v screen >/dev/null 2>&1; then
    echo "Starting NAG training in screen session: $SCREEN_NAME"
    echo "Attach with: screen -r $SCREEN_NAME"
    INSIDE_TRAIN_SCREEN=1 screen -dmS "$SCREEN_NAME" bash "$0"
  elif command -v tmux >/dev/null 2>&1; then
    echo "Starting NAG training in tmux session: $SCREEN_NAME"
    echo "Attach with: tmux attach -t $SCREEN_NAME"
    tmux new-session -d -s "$SCREEN_NAME" "cd '$REPO_DIR' && INSIDE_TRAIN_SCREEN=1 bash '$0'"
  else
    echo "Neither screen nor tmux is installed. Set RUN_IN_SCREEN=0 to run in the foreground."
    exit 1
  fi
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
  --arch=nag-gpt
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
  "--nag-gate-log-every=$NAG_GATE_LOG_EVERY"
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
