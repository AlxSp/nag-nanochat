"""Local Figure 3-style residual norm and cumulative rotation diagnostics."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nag-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT as BaseGPT, GPTConfig as BaseConfig
from nanochat.nag_gpt import GPT as NAGGPT, GPTConfig as NAGConfig
from nanochat.tokenizer import get_tokenizer


OUTPUT = Path(__file__).resolve().parent / "output/fig3_residual_geometry"
BASE_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints/gpt_d64_w640_3e19"
NAG_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints/nag_gpt_d64_w640_3e19_gatefix"
SEQUENCE_LENGTH = 64
NUM_SEQUENCES = 24


def rms_norm(x):
    return F.rms_norm(x, (x.shape[-1],))


def load_state(path):
    state = torch.load(path, map_location="cpu")
    return {
        key.removeprefix("_orig_mod."): value.float()
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for key, value in state.items()
    }


def load_model(checkpoint, model_class, config_class):
    meta_path = sorted(checkpoint.glob("meta_*.json"))[-1]
    meta = json.loads(meta_path.read_text())
    step = meta_path.stem.removeprefix("meta_")
    with torch.device("meta"):
        model = model_class(config_class(**meta["model_config"]))
    model.to_empty(device="cpu")
    model.init_weights()
    model.load_state_dict(load_state(checkpoint / f"model_{step}.pt"), strict=True, assign=True)
    model.eval()
    return model, meta


def make_eval_batch():
    import pyarrow.parquet as pq

    tokenizer = get_tokenizer()
    data_root = Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache/nanochat"))
    shard = sorted((data_root / "base_data_climbmix").glob("shard_*.parquet"))[-1]
    parquet = pq.ParquetFile(shard)
    bos = tokenizer.get_bos_token_id()
    sequences = []
    for row_group in range(parquet.num_row_groups):
        for text in parquet.read_row_group(row_group).column("text").to_pylist():
            tokens = tokenizer.encode(text, prepend=bos)
            if len(tokens) >= SEQUENCE_LENGTH:
                sequences.append(tokens[:SEQUENCE_LENGTH])
            if len(sequences) == NUM_SEQUENCES:
                return torch.tensor(sequences, dtype=torch.long)
    raise RuntimeError(f"Found only {len(sequences)} usable sequences")


def mean_rms(x):
    return float(x.float().square().mean(dim=-1).sqrt().mean())


def mean_rotation_degrees(before, after):
    before = F.normalize(before.float(), dim=-1)
    after = F.normalize(after.float(), dim=-1)
    cosine = (before * after).sum(dim=-1).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)).mean())


@torch.no_grad()
def collect_baseline(model, token_ids):
    sequence_len = token_ids.shape[1]
    cos_sin = model.cos[:, :sequence_len].float(), model.sin[:, :sequence_len].float()
    residual = rms_norm(model.transformer.wte(token_ids).float())
    norms = [mean_rms(residual)]
    rotations = []
    labels = ["input"]
    for index, block in enumerate(model.transformer.h):
        before = residual
        residual = residual + block.attn(rms_norm(residual), cos_sin, model.window_sizes[index], None)
        rotations.append(mean_rotation_degrees(before, residual))
        norms.append(mean_rms(residual))
        labels.append(f"{index}:attn")

        before = residual
        residual = residual + block.mlp(rms_norm(residual))
        rotations.append(mean_rotation_degrees(before, residual))
        norms.append(mean_rms(residual))
        labels.append(f"{index}:mlp")
    return np.asarray(norms), np.cumsum(rotations), labels


@torch.no_grad()
def collect_nag(model, token_ids):
    sequence_len = token_ids.shape[1]
    cos_sin = model.cos[:, :sequence_len].float(), model.sin[:, :sequence_len].float()
    embedding = model.transformer.wte(token_ids).float()
    rho = embedding.norm(dim=-1, keepdim=True) / model.config.n_embd**0.5
    log_norm = rho.log() + model.g_log_encode.float()
    direction = embedding / rho
    norms = [float(log_norm.exp().mean())]
    rotations = []
    labels = ["input"]
    for index, block in enumerate(model.transformer.h):
        before = direction
        output = block.attn(direction, None, cos_sin, model.window_sizes[index], None)
        log_norm, direction = block.attn_branch(log_norm, direction, output)
        rotations.append(mean_rotation_degrees(before, direction))
        norms.append(float(log_norm.exp().mean()))
        labels.append(f"{index}:attn")

        before = direction
        log_norm, direction = block.mlp_branch(log_norm, direction, block.mlp(direction))
        rotations.append(mean_rotation_degrees(before, direction))
        norms.append(float(log_norm.exp().mean()))
        labels.append(f"{index}:mlp")
    return np.asarray(norms), np.cumsum(rotations), labels


def plot(base_norm, nag_norm, base_rotation, nag_rotation):
    colors = {"baseline": "#d84b3e", "nag": "#2864a8"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    norm_layers = np.arange(len(base_norm))
    axes[0].plot(norm_layers, base_norm, color=colors["baseline"], linewidth=2, label="Baseline GPT")
    axes[0].plot(norm_layers, nag_norm, color=colors["nag"], linewidth=2, label="NAG")
    axes[0].set(title="Residual-stream norm", xlabel="layer (attention / MLP interleaved)", ylabel="mean token RMS norm")
    axes[0].grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.65)
    axes[0].legend(frameon=False)

    rotation_layers = np.arange(1, len(base_rotation) + 1)
    axes[1].plot(rotation_layers, base_rotation, color=colors["baseline"], linewidth=2, label="Baseline GPT")
    axes[1].plot(rotation_layers, nag_rotation, color=colors["nag"], linewidth=2, label="NAG")
    axes[1].set(title="Cumulative residual rotation", xlabel="layer (attention / MLP interleaved)", ylabel="cumulative rotation (degrees)")
    axes[1].grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.65)
    axes[1].legend(frameon=False)
    fig.suptitle("Residual geometry through depth")
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"fig3_residual_norm_and_rotation.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    token_ids = make_eval_batch()
    print("Loading baseline")
    baseline, baseline_meta = load_model(BASE_CHECKPOINT, BaseGPT, BaseConfig)
    base_norm, base_rotation, labels = collect_baseline(baseline, token_ids)
    del baseline
    print("Loading NAG")
    nag, nag_meta = load_model(NAG_CHECKPOINT, NAGGPT, NAGConfig)
    nag_norm, nag_rotation, nag_labels = collect_nag(nag, token_ids)
    assert labels == nag_labels
    plot(base_norm, nag_norm, base_rotation, nag_rotation)
    np.savez_compressed(
        OUTPUT / "fig3_residual_geometry_arrays.npz",
        baseline_norm=base_norm,
        nag_norm=nag_norm,
        baseline_cumulative_rotation=base_rotation,
        nag_cumulative_rotation=nag_rotation,
    )
    metrics = {
        "sample_shape": list(token_ids.shape),
        "baseline_step": baseline_meta["step"],
        "nag_step": nag_meta["step"],
        "baseline_final_norm": float(base_norm[-1]),
        "nag_final_norm": float(nag_norm[-1]),
        "baseline_cumulative_rotation_degrees": float(base_rotation[-1]),
        "nag_cumulative_rotation_degrees": float(nag_rotation[-1]),
    }
    (OUTPUT / "fig3_residual_geometry_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
