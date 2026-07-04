# NAG/GPT 64x640 Experiment Runbook

Goal: run a fair paired comparison between the NAG architecture and the clean GPT baseline on the same 8xH100/H200 machine.

Do not commit anything from the training instance.

## Storage layout

- Repo/code: `/nag-nanochat`
- Data/cache/checkpoints: `/workspace/nanochat`
- Always set:

```bash
export NANOCHAT_BASE_DIR=/workspace/nanochat
```

Expected cache layout:

```text
/workspace/nanochat/tokenizer
/workspace/nanochat/base_data_climbmix
/workspace/nanochat/base_checkpoints
/workspace/nanochat/report
```

## RunPod Storage Recommendation

Use persistent `/workspace` storage, not only ephemeral container storage, for this experiment.

Recommended RunPod settings:

```text
Container disk: 100-150 GB
Volume/network disk mounted at /workspace: 300-500 GB
```

Rationale:

- Container disk only needs to cover the image, repo, venv, package/build caches, and temporary compile artifacts.
- `/workspace` should hold the tokenizer, 1000-shard ClimbMix subset, checkpoints, optimizer shards, and logs.
- A single saved 8-rank checkpoint set can be several GB; the downloaded step-3000/3500 checkpoint sample is already about 5.5 GB.
- For paired NAG + GPT runs, 300 GB is the practical minimum I would choose; 500 GB is more comfortable if saving every 100 steps or keeping multiple failed/resumed attempts.
- On a 250 GB `/workspace` volume, do not save every 100 steps for this run. Use `SAVE_EVERY=500` or larger, and prune partial/old checkpoints after failures.

Before launch, verify:

```bash
df -h /workspace
du -sh /workspace/nanochat 2>/dev/null || true
```

## Tokenizer

Use the copied local tokenizer. Do not retrain it on the H100/H200 instance.

Required files:

```text
/workspace/nanochat/tokenizer/tokenizer.pkl
/workspace/nanochat/tokenizer/token_bytes.pt
```

Verify:

```bash
ls -lh /workspace/nanochat/tokenizer
```

## Dataset

Download exactly 1000 train shards plus the fixed validation shard. This matches the local setup used for the prior experiments.

Use:

```bash
cd /nag-nanochat
export NANOCHAT_BASE_DIR=/workspace/nanochat
uv run python -m nanochat.dataset -n 1000 -w 16
```

Do not download the full 6543-shard ClimbMix dataset for this run.

Expected result:

```text
shard_00000.parquet ... shard_00999.parquet
shard_06542.parquet
```

That is 1001 parquet files total: 1000 train shards and 1 validation shard.

Verify:

```bash
find /workspace/nanochat/base_data_climbmix -maxdepth 1 -name '*.parquet' | wc -l
ls /workspace/nanochat/base_data_climbmix/shard_00000.parquet
ls /workspace/nanochat/base_data_climbmix/shard_00999.parquet
ls /workspace/nanochat/base_data_climbmix/shard_06542.parquet
```

## Shared Model/Training Settings

Use these settings for both NAG and GPT:

```text
--depth=64
--model-dim=640
--head-dim=128
--window-pattern=L
--target-flops=3e19
--target-param-data-ratio=-1
--eval-every=250
--save-every=500
--sample-every=-1
--core-metric-every=999999
```

Keep `torch.compile` enabled. Use all 8 GPUs with `torchrun`.

The earlier H100 run fit with `--device-batch-size=16`. Keep `DEVICE_BATCH_SIZE=16` for both runs unless memory forces a change, and document any override. The identified NaN trigger was the NAG modulator `pow` backward at `s=0`, not a confirmed microbatch-size issue.

Shared settings live in:

```bash
runs/h100_64x640.env
```

## Smoke Test

Before the main run, smoke test the NAG architecture on all 8 GPUs with the final dimensions:

```text
--arch=nag-gpt
--depth=64
--model-dim=640
--head-dim=128
--window-pattern=L
```

If the smoke test fails, adjust only runtime/training parameters. Do not change architecture code.

Before launching the full run, also require FA3 to be active:

```bash
cd /nag-nanochat
source runs/h100_64x640.env
uv run python -m scripts.check_fa3
```

The train scripts run this check automatically and refuse to launch if FA3 is not selected.

Then run the full-shape one-step smoke:

```bash
cd /nag-nanochat
scripts/smoke_h100_nag_64x640.sh
```

## Main Runs

Instance setup:

```bash
cd /nag-nanochat
scripts/setup_runpod_h100.sh
source /workspace/nanochat/env.sh
```

Dataset download:

```bash
cd /nag-nanochat
scripts/download_climbmix_1000.sh
```

Tokenizer upload from the local machine, using the direct RunPod SSH endpoint:

```bash
scripts/upload_tokenizer_to_runpod.sh root@HOST PORT ~/.ssh/id_ed25519 /workspace/nanochat
```

Run NAG first:

```text
--arch=nag-gpt
--model-tag=nag_gpt_d64_w640_3e19
--run=nag_gpt_d64_w640_3e19
```

Or use:

```bash
cd /nag-nanochat
scripts/train_h100_nag_64x640.sh
```

Then run GPT baseline with the exact same data/tokenizer/training budget:

```text
--arch=gpt
--model-tag=gpt_d64_w640_3e19
--run=gpt_d64_w640_3e19
```

Or use:

```bash
cd /nag-nanochat
scripts/train_h100_gpt_64x640.sh
```

Both training scripts start in `screen` by default, or `tmux` if `screen` is not installed. Attach with one of:

```bash
screen -r train_nag_gpt_d64_w640_3e19
screen -r train_gpt_d64_w640_3e19
tmux attach -t train_nag_gpt_d64_w640_3e19
tmux attach -t train_gpt_d64_w640_3e19
```

## Fairness Requirements

- Same tokenizer.
- Same 1000 train shards plus `shard_06542` validation.
- Same target FLOPs.
- Same model dimensions.
- Same sequence length/default training script behavior.
- Same eval/checkpoint cadence.
- Same batch size policy unless one architecture OOMs, in which case document the difference.
- Compare validation bpb at matched FLOPs/steps from the run metadata.

## Before Launch Checklist

- `git status` inspected.
- Branch is `nag`.
- `/workspace` has enough free disk.
- `nvidia-smi` shows 8 GPUs.
- CUDA/PyTorch visible from the environment.
- Tokenizer files exist.
- Dataset count is 1001 parquet files.
- NAG smoke test passed.

## After Both Runs

- Compare validation bpb at matched FLOPs.
- Compare throughput, MFU, wall time, and actual trained tokens.
- Inspect final NAG checkpoint for gain/modulator behavior.
- Generate blog plots from both runs where possible.


# Log:

8xH100 run that resulted in NANs:
```
OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=/workspace/nanochat .venv/bin/torchrun --standalone --nproc_per_node=8 -m
  scripts.base_train -- --arch=nag-gpt --depth=64 --model-dim=640 --head-dim=128 --window-pattern=L --target-flops=3e19 --target-
  param-data-ratio=-1 --eval-every=250 --save-every=500 --sample-every=-1 --core-metric-every=999999 --model-
  tag=nag_gpt_d64_w640_3e19 --run=nag_gpt_d64_w640_3e19 --device-batch-size=16
```
