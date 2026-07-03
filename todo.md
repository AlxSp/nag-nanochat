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

Use the largest safe `--device-batch-size` that fits without OOM. Keep the same batch size policy for both runs.

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

## Main Runs

Run NAG first:

```text
--arch=nag-gpt
--model-tag=nag_gpt_d64_w640_3e19
--run=nag_gpt_d64_w640_3e19
```

Then run GPT baseline with the exact same data/tokenizer/training budget:

```text
--arch=gpt
--model-tag=gpt_d64_w640_3e19
--run=gpt_d64_w640_3e19
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
