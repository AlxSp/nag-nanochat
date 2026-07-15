"""Checkpoint diagnostics inspired by Figures 5 and 6 of the NAG paper."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nag-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nanochat.nag_gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints/nag_gpt_d64_w640_3e19_gatefix"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output/nag_mechanisms"


def load_state(path):
    state = torch.load(path, map_location="cpu")
    return {
        key.removeprefix("_orig_mod."): value.float()
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for key, value in state.items()
    }


def load_model(checkpoint):
    metas = sorted(checkpoint.glob("meta_*.json"))
    if not metas:
        raise FileNotFoundError(f"No checkpoint metadata in {checkpoint}")
    meta_path = metas[-1]
    meta = json.loads(meta_path.read_text())
    step = meta_path.stem.removeprefix("meta_")
    model_path = checkpoint / f"model_{step}.pt"
    with torch.device("meta"):
        model = GPT(GPTConfig(**meta["model_config"]))
    model.to_empty(device="cpu")
    model.init_weights()
    model.load_state_dict(load_state(model_path), strict=True, assign=True)
    model.eval()
    return model, meta


def make_eval_batch(sequence_len, num_sequences):
    import pyarrow.parquet as pq

    tokenizer = get_tokenizer()
    data_dir = Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache/nanochat"))
    shards = sorted((data_dir / "base_data_climbmix").glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No validation shards under {data_dir / 'base_data_climbmix'}")
    parquet = pq.ParquetFile(shards[-1])
    bos = tokenizer.get_bos_token_id()
    sequences = []
    for row_group in range(parquet.num_row_groups):
        texts = parquet.read_row_group(row_group).column("text").to_pylist()
        for text in texts:
            token_ids = tokenizer.encode(text, prepend=bos)
            if len(token_ids) >= sequence_len:
                sequences.append(token_ids[:sequence_len])
            if len(sequences) == num_sequences:
                return torch.tensor(sequences, dtype=torch.long)
    raise RuntimeError(f"Found only {len(sequences)} usable sequences")


def modulator(branch, residual_direction):
    selector = torch.sigmoid(branch.m_down(residual_direction))
    mixture = (F.softmax(branch.coef, dim=0) * selector).sum(dim=-1)
    power = F.softplus(branch.beta)
    return mixture.clamp_min(1e-6).pow(power)


def preferred_direction(residual_direction, weights):
    flat_residual = residual_direction.reshape(-1, residual_direction.shape[-1]).float()
    flat_weights = weights.reshape(-1).float()
    return (flat_residual * flat_weights[:, None]).sum(0) / flat_weights.sum().clamp_min(1e-12)


@torch.no_grad()
def collect(model, token_ids):
    _, sequence_len = token_ids.shape
    cos_sin = model.cos[:, :sequence_len].float(), model.sin[:, :sequence_len].float()
    embedding = model.transformer.wte(token_ids).float()
    rho = embedding.norm(dim=-1, keepdim=True) / model.config.n_embd**0.5
    log_norm = rho.log() + model.g_log_encode.float()
    direction = embedding / rho

    preferred = {"attention": [], "mlp": [], "unweighted": []}
    alpha = {"attention": [], "mlp": []}
    mean_modulator = {"attention": [], "mlp": []}
    mean_gain = {"attention": [], "mlp": []}
    mean_abs_gain = {"attention": [], "mlp": []}

    for layer_index, block in enumerate(model.transformer.h):
        preferred["unweighted"].append(direction.reshape(-1, direction.shape[-1]).mean(0))

        attn_mod = modulator(block.attn_branch, direction)
        preferred["attention"].append(preferred_direction(direction, attn_mod))
        attn_alpha = block.attn_branch.alpha.float() * block.attn_branch.alpha_warmup_scale.float()
        attn_gain = attn_alpha * attn_mod
        alpha["attention"].append(float(attn_alpha))
        mean_modulator["attention"].append(float(attn_mod.mean()))
        mean_gain["attention"].append(float(attn_gain.mean()))
        mean_abs_gain["attention"].append(float(attn_gain.abs().mean()))

        attention_output = block.attn(direction, None, cos_sin, model.window_sizes[layer_index], None)
        log_norm, direction = block.attn_branch(log_norm, direction, attention_output)

        mlp_mod = modulator(block.mlp_branch, direction)
        preferred["mlp"].append(preferred_direction(direction, mlp_mod))
        mlp_alpha = block.mlp_branch.alpha.float() * block.mlp_branch.alpha_warmup_scale.float()
        mlp_gain = mlp_alpha * mlp_mod
        alpha["mlp"].append(float(mlp_alpha))
        mean_modulator["mlp"].append(float(mlp_mod.mean()))
        mean_gain["mlp"].append(float(mlp_gain.mean()))
        mean_abs_gain["mlp"].append(float(mlp_gain.abs().mean()))

        log_norm, direction = block.mlp_branch(log_norm, direction, block.mlp(direction))

    logits = model.lm_head(direction) / model.config.n_embd
    logits = logits[..., : model.config.vocab_size].float() * log_norm.exp()
    prediction_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    targets = token_ids[:, 1:].reshape(-1)
    log_probs = prediction_logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    confidence = {
        "norm": log_norm[:, :-1].exp().reshape(-1).cpu().numpy(),
        "entropy": (-(probs * log_probs).sum(-1)).cpu().numpy(),
        "nll": (-log_probs.gather(1, targets[:, None]).squeeze(1)).cpu().numpy(),
        "correct": (prediction_logits.argmax(-1) == targets).float().cpu().numpy(),
    }
    return {
        "preferred": {key: torch.stack(value).cpu().numpy() for key, value in preferred.items()},
        "alpha": alpha,
        "mean_modulator": mean_modulator,
        "mean_gain": mean_gain,
        "mean_abs_gain": mean_abs_gain,
        "confidence": confidence,
    }


def cosine_matrix(directions):
    normalized = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    return normalized @ normalized.T


def separation_profile(matrix):
    return np.array([np.diag(matrix, offset).mean() for offset in range(matrix.shape[0])])


def interleave(attention, mlp):
    attention = np.asarray(attention)
    mlp = np.asarray(mlp)
    if attention.shape != mlp.shape:
        raise ValueError(f"Cannot interleave shapes {attention.shape} and {mlp.shape}")
    joined = np.empty((attention.shape[0] * 2, *attention.shape[1:]), dtype=attention.dtype)
    joined[0::2] = attention
    joined[1::2] = mlp
    return joined


def save_figure6(results, output):
    matrices = {key: cosine_matrix(value) for key, value in results["preferred"].items()}
    joined_directions = interleave(results["preferred"]["attention"], results["preferred"]["mlp"])
    matrices["joined"] = cosine_matrix(joined_directions)

    fig, ax = plt.subplots(figsize=(5.0, 4.35), constrained_layout=True)
    image = ax.imshow(matrices["joined"], cmap="viridis", vmin=0, vmax=1, origin="upper")
    ax.set(
        title="Preferred-direction similarity",
        xlabel="layer (attention / MLP interleaved)",
        ylabel="layer (attention / MLP interleaved)",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    for extension in ("png", "svg"):
        fig.savefig(output / f"fig6_preferred_direction_similarity_joined.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.65), constrained_layout=True)
    panels = [("attention", "Attention modulator"), ("mlp", "MLP modulator"), ("unweighted", "Unweighted control")]
    for ax, (key, title) in zip(axes, panels):
        image = ax.imshow(matrices[key], cmap="viridis", vmin=0, vmax=1, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("layer")
        ax.set_ylabel("layer")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="cosine similarity")
    for extension in ("png", "svg"):
        fig.savefig(output / f"fig6_preferred_direction_similarity.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    colors = {"attention": "#2864a8", "mlp": "#c4473a", "unweighted": "#555555"}
    for key, label in panels:
        ax.plot(separation_profile(matrices[key]), label=label, color=colors[key], linewidth=2)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set(xlabel="layer separation |i-j|", ylabel="mean cosine similarity", title="Preferred-direction locality")
    ax.legend(frameon=False)
    for extension in ("png", "svg"):
        fig.savefig(output / f"fig6_similarity_by_separation.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    return matrices


def save_figure5(results, output):
    joined_alpha = interleave(results["alpha"]["attention"], results["alpha"]["mlp"])
    joined_gain = interleave(results["mean_gain"]["attention"], results["mean_gain"]["mlp"])
    layers = np.arange(len(joined_alpha))
    fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    ax.plot(layers, joined_alpha, linestyle=":", linewidth=1.8, color="#2455ff", label="alpha")
    ax.plot(layers, joined_gain, linewidth=1.5, color="#ef2b2d", label="mean gain")
    ax.set(
        xlabel="layer (attention / MLP interleaved)",
        ylabel="value",
        title="Layer-wise NAG scale and realized gain",
    )
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#b8b8b8", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=True)
    for extension in ("png", "svg"):
        fig.savefig(output / f"fig5_layerwise_gain_joined.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.1), sharex=True, constrained_layout=True)
    colors = {"attention": "#2864a8", "mlp": "#c4473a"}
    for ax, branch in zip(axes, ("attention", "mlp")):
        layers = np.arange(len(results["alpha"][branch]))
        ax.plot(layers, results["alpha"][branch], linestyle=":", linewidth=2, color=colors[branch], label="alpha")
        ax.plot(layers, results["mean_gain"][branch], linewidth=2, color=colors[branch], label="mean signed gain")
        ax.plot(layers, results["mean_abs_gain"][branch], linewidth=1.5, color="#222222", label="mean |gain|")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_ylabel("value")
        ax.set_title(f"{branch.capitalize()} branch")
        ax.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("layer")
    fig.suptitle("Layer-wise NAG scale and realized gain")
    for extension in ("png", "svg"):
        fig.savefig(output / f"fig5_layerwise_gain.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def binned_confidence(confidence, bins=10):
    order = np.argsort(confidence["norm"])
    chunks = np.array_split(order, bins)
    return {
        key: np.array([values[chunk].mean() for chunk in chunks])
        for key, values in confidence.items()
    }, np.array([len(chunk) for chunk in chunks])


def ranking_auroc(signal, correct):
    ranks = np.empty(len(signal))
    ranks[np.argsort(signal)] = np.arange(len(signal))
    positive, negative = ranks[correct == 1], ranks[correct == 0]
    return float((positive.mean() - negative.mean()) / len(signal) + 0.5)


def save_confidence(results, output):
    binned, counts = binned_confidence(results["confidence"])
    confidence = results["confidence"]
    entropy_order = np.argsort(-confidence["entropy"])
    entropy_ranked_accuracy = np.array(
        [confidence["correct"][chunk].mean() for chunk in np.array_split(entropy_order, len(counts))]
    )
    deciles = np.arange(1, len(counts) + 1)
    charcoal, reference = "#2b2b2b", "#8a8a8a"

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5), constrained_layout=True)
    axes[0].plot(binned["norm"], binned["entropy"], marker="o", linewidth=2, color=charcoal)
    axes[0].set(xlabel="mean final residual RMS norm", ylabel="predictive entropy", title="sharpness (mechanical)")
    axes[1].plot(binned["norm"], binned["nll"], marker="o", linewidth=2, color=charcoal)
    axes[1].set(xlabel="mean final residual RMS norm", ylabel="next-token NLL", title="error (empirical)")
    axes[2].plot(deciles, binned["correct"], marker="o", linewidth=2, color=charcoal, label="ranked by residual norm")
    axes[2].plot(deciles, entropy_ranked_accuracy, marker="o", linewidth=1.6, linestyle="--", color=reference, label="ranked by output entropy")
    axes[2].set(xlabel="confidence decile (low → high)", ylabel="top-1 accuracy", title="accuracy (empirical)", xticks=deciles)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("NAG residual norm as inverse temperature and confidence")
    for extension in ("png", "svg"):
        fig.savefig(output / f"nag_norm_confidence_calibration.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    correlations = {
        key: float(np.corrcoef(results["confidence"]["norm"], results["confidence"][key])[0, 1])
        for key in ("entropy", "nll", "correct")
    }
    correlations["auroc_norm"] = ranking_auroc(confidence["norm"], confidence["correct"])
    correlations["auroc_entropy"] = ranking_auroc(-confidence["entropy"], confidence["correct"])
    return binned, counts, correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--num-sequences", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint}")
    model, meta = load_model(args.checkpoint)
    print("Loading validation tokens")
    token_ids = make_eval_batch(args.sequence_length, args.num_sequences)
    print(f"Collecting diagnostics for {tuple(token_ids.shape)} tokens")
    results = collect(model, token_ids)
    matrices = save_figure6(results, args.output)
    save_figure5(results, args.output)
    binned, counts, correlations = save_confidence(results, args.output)

    summary = {
        "checkpoint": str(args.checkpoint),
        "step": meta["step"],
        "sample_shape": list(token_ids.shape),
        "confidence_correlations": correlations,
        "confidence_bin_counts": counts.tolist(),
        "confidence_bins": {key: value.tolist() for key, value in binned.items()},
        "adjacent_layer_cosine": {
            key: float(np.diag(matrix, 1).mean()) for key, matrix in matrices.items()
        },
        "distant_layer_cosine": {
            key: float(np.diag(matrix, matrix.shape[0] // 2).mean()) for key, matrix in matrices.items()
        },
        "layerwise": {
            key: results[key] for key in ("alpha", "mean_modulator", "mean_gain", "mean_abs_gain")
        },
    }
    (args.output / "nag_mechanism_metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output / "nag_mechanism_arrays.npz",
        **{f"preferred_{key}": value for key, value in results["preferred"].items()},
        **{f"confidence_{key}": value for key, value in results["confidence"].items()},
    )
    print(json.dumps({"confidence_correlations": correlations, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
