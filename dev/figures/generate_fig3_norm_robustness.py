"""Experimental residual-norm plot separating BOS outliers from typical tokens."""

import json
import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parent
sys.path.insert(0, str(PARENT))

import matplotlib.pyplot as plt
import numpy as np
import torch

import generate_fig3_residual_geometry as fig3


OUTPUT = PARENT / "output/fig3_residual_geometry"


def summarize_norms(norms):
    flat = norms.reshape(-1)
    return {
        "mean_all": float(flat.mean()),
        "mean_without_bos": float(norms[:, 1:].mean()),
        "median": float(flat.median()),
        "bos_mean": float(norms[:, 0].mean()),
    }


def baseline_token_norms(residual):
    return residual.float().square().mean(dim=-1).sqrt()


@torch.no_grad()
def collect_baseline(model, token_ids):
    sequence_len = token_ids.shape[1]
    cos_sin = model.cos[:, :sequence_len].float(), model.sin[:, :sequence_len].float()
    residual = fig3.rms_norm(model.transformer.wte(token_ids).float())
    summaries = [summarize_norms(baseline_token_norms(residual))]
    for index, block in enumerate(model.transformer.h):
        residual = residual + block.attn(fig3.rms_norm(residual), cos_sin, model.window_sizes[index], None)
        summaries.append(summarize_norms(baseline_token_norms(residual)))
        residual = residual + block.mlp(fig3.rms_norm(residual))
        summaries.append(summarize_norms(baseline_token_norms(residual)))
    return summaries


@torch.no_grad()
def collect_nag(model, token_ids):
    sequence_len = token_ids.shape[1]
    cos_sin = model.cos[:, :sequence_len].float(), model.sin[:, :sequence_len].float()
    embedding = model.transformer.wte(token_ids).float()
    rho = embedding.norm(dim=-1, keepdim=True) / model.config.n_embd**0.5
    log_norm = rho.log() + model.g_log_encode.float()
    direction = embedding / rho
    summaries = [summarize_norms(log_norm.exp().squeeze(-1))]
    for index, block in enumerate(model.transformer.h):
        output = block.attn(direction, None, cos_sin, model.window_sizes[index], None)
        log_norm, direction = block.attn_branch(log_norm, direction, output)
        summaries.append(summarize_norms(log_norm.exp().squeeze(-1)))
        log_norm, direction = block.mlp_branch(log_norm, direction, block.mlp(direction))
        summaries.append(summarize_norms(log_norm.exp().squeeze(-1)))
    return summaries


def series(summaries, key):
    return np.asarray([item[key] for item in summaries])


def plot(baseline, nag):
    styles = {
        "mean_all": ("Mean, all tokens", "#d84b3e", "-"),
        "mean_without_bos": ("Mean, excluding BOS", "#2864a8", "-"),
        "median": ("Median", "#278052", "--"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for ax, summaries, title in zip(axes, (baseline, nag), ("Baseline GPT", "NAG")):
        layers = np.arange(len(summaries))
        for key, (label, color, linestyle) in styles.items():
            ax.plot(layers, series(summaries, key), label=label, color=color, linestyle=linestyle, linewidth=2)
        ax.set(title=title, xlabel="layer (attention / MLP interleaved)", ylabel="token RMS norm")
        ax.grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.65)
        ax.legend(frameon=False)
    fig.suptitle("Residual norm: BOS outlier versus typical tokens")
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"experimental_norm_bos_robustness.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    token_ids = fig3.make_eval_batch()
    print("Loading baseline")
    baseline_model, _ = fig3.load_model(fig3.BASE_CHECKPOINT, fig3.BaseGPT, fig3.BaseConfig)
    baseline = collect_baseline(baseline_model, token_ids)
    del baseline_model
    print("Loading NAG")
    nag_model, _ = fig3.load_model(fig3.NAG_CHECKPOINT, fig3.NAGGPT, fig3.NAGConfig)
    nag = collect_nag(nag_model, token_ids)
    plot(baseline, nag)
    metrics = {
        "sample_shape": list(token_ids.shape),
        "baseline_final": baseline[-1],
        "nag_final": nag[-1],
    }
    (OUTPUT / "experimental_norm_bos_robustness_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        OUTPUT / "experimental_norm_bos_robustness_arrays.npz",
        **{f"baseline_{key}": series(baseline, key) for key in baseline[0]},
        **{f"nag_{key}": series(nag, key) for key in nag[0]},
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
