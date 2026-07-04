#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 user@host port identity_file [remote_base_dir]"
  echo "   or: RUNPOD_REMOTE=user@host RUNPOD_PORT=port $0"
  echo "Example: $0 root@103.207.149.156 15265 ~/.ssh/id_ed25519 /workspace/nanochat"
  echo "Example: RUNPOD_REMOTE=root@103.207.149.156 RUNPOD_PORT=15265 $0"
}

if [ "$#" -eq 0 ]; then
  REMOTE="${RUNPOD_REMOTE:-}"
  PORT="${RUNPOD_PORT:-}"
  IDENTITY_FILE="${RUNPOD_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
  REMOTE_BASE_DIR="${RUNPOD_REMOTE_BASE_DIR:-/workspace/nanochat}"
elif [ "$#" -ge 3 ] && [ "$#" -le 4 ]; then
  REMOTE="$1"
  PORT="$2"
  IDENTITY_FILE="$3"
  REMOTE_BASE_DIR="${4:-/workspace/nanochat}"
else
  usage
  exit 2
fi

if [ -z "$REMOTE" ] || [ -z "$PORT" ] || [ -z "$IDENTITY_FILE" ]; then
  usage
  exit 2
fi

LOCAL_BASE_DIR="${LOCAL_NANOCHAT_BASE_DIR:-${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}}"
LOCAL_TOKENIZER_DIR="$LOCAL_BASE_DIR/tokenizer"

test -d "$LOCAL_TOKENIZER_DIR"
test -f "$LOCAL_TOKENIZER_DIR/tokenizer.pkl"
test -f "$LOCAL_TOKENIZER_DIR/token_bytes.pt"

SSH_OPTS=(-p "$PORT" -i "$IDENTITY_FILE" -o StrictHostKeyChecking=accept-new)

echo "Uploading tokenizer from $LOCAL_TOKENIZER_DIR to $REMOTE:$REMOTE_BASE_DIR/tokenizer"
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_BASE_DIR'"

if command -v rsync >/dev/null 2>&1 && ssh "${SSH_OPTS[@]}" "$REMOTE" "command -v rsync >/dev/null 2>&1"; then
  rsync -az -e "ssh -p $PORT -i $IDENTITY_FILE -o StrictHostKeyChecking=accept-new" \
    "$LOCAL_TOKENIZER_DIR/" "$REMOTE:$REMOTE_BASE_DIR/tokenizer/"
else
  tar -C "$LOCAL_BASE_DIR" -cf - tokenizer | ssh "${SSH_OPTS[@]}" "$REMOTE" "tar -C '$REMOTE_BASE_DIR' -xf -"
fi

ssh "${SSH_OPTS[@]}" "$REMOTE" "ls -lh '$REMOTE_BASE_DIR/tokenizer' && test -f '$REMOTE_BASE_DIR/tokenizer/tokenizer.pkl' && test -f '$REMOTE_BASE_DIR/tokenizer/token_bytes.pt'"
echo "Tokenizer upload verified"
