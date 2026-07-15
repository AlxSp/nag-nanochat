"""Alternate NAG norm-confidence plot with a GPT entropy reference."""

import json
import os
import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parent
REPO = PARENT.parents[1]
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nag-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import generate_nag_mechanism_figures as mechanism
from nanochat.gpt import GPT, GPTConfig


OUTPUT = PARENT / "output/nag_mechanisms"
BASE_CHECKPOINT = PARENT.parents[1] / "checkpoints/gpt_d64_w640_3e19"
NAG_ARRAYS = OUTPUT / "nag_mechanism_arrays.npz"
NUM_BINS = 10


def load_baseline():
    meta_path = sorted(BASE_CHECKPOINT.glob("meta_*.json"))[-1]
    meta = json.loads(meta_path.read_text())
    step = meta_path.stem.removeprefix("meta_")
    with torch.device("meta"):
        model = GPT(GPTConfig(**meta["model_config"]))
    model.to_empty(device="cpu")
    model.init_weights()
    model.load_state_dict(
        mechanism.load_state(BASE_CHECKPOINT / f"model_{step}.pt"),
        strict=True,
        assign=True,
    )
    model.eval()
    return model


@torch.no_grad()
def baseline_confidence(model, token_ids):
    logits = model(token_ids)[:, :-1].reshape(-1, model.config.vocab_size)
    targets = token_ids[:, 1:].reshape(-1)
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    return {
        "entropy": (-(probs * log_probs).sum(-1)).cpu().numpy(),
        "correct": (logits.argmax(-1) == targets).float().cpu().numpy(),
    }


def equal_count_means(values, order, bins=NUM_BINS):
    return np.asarray([values[index].mean() for index in np.array_split(order, bins)])


def correctness_auroc(confidence, correct):
    confidence = np.asarray(confidence)
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(confidence)
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    positives = correct.sum()
    negatives = len(correct) - positives
    return float((ranks[correct].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def main():
    nag_arrays = np.load(NAG_ARRAYS)
    nag = {
        key: nag_arrays[f"confidence_{key}"]
        for key in ("norm", "entropy", "nll", "correct")
    }
    token_ids = mechanism.make_eval_batch(sequence_len=64, num_sequences=24)
    baseline = baseline_confidence(load_baseline(), token_ids)

    nag_order = np.argsort(nag["norm"])
    nag_entropy_order = np.argsort(-nag["entropy"])
    gpt_order = np.argsort(-baseline["entropy"])
    nag_binned = {
        key: equal_count_means(values, nag_order) for key, values in nag.items()
    }
    nag_entropy_accuracy = equal_count_means(nag["correct"], nag_entropy_order)
    gpt_accuracy = equal_count_means(baseline["correct"], gpt_order)
    auroc = {
        "nag_norm": correctness_auroc(nag["norm"], nag["correct"]),
        "nag_entropy": correctness_auroc(-nag["entropy"], nag["correct"]),
        "gpt_entropy": correctness_auroc(-baseline["entropy"], baseline["correct"]),
    }

    charcoal = "#2b2b2b"
    reference = "#718096"
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5), constrained_layout=True)
    axes[0].plot(nag_binned["norm"], nag_binned["entropy"], marker="o", color=charcoal, linewidth=2)
    axes[0].set(xlabel="mean final residual RMS norm", ylabel="predictive entropy")
    axes[1].plot(nag_binned["norm"], nag_binned["nll"], marker="o", color=charcoal, linewidth=2)
    axes[1].set(xlabel="mean final residual RMS norm", ylabel="next-token NLL")

    deciles = np.arange(1, NUM_BINS + 1)
    axes[2].plot(deciles, nag_binned["correct"], marker="o", color=charcoal, linewidth=2.2, label="NAG by residual norm")
    axes[2].plot(deciles, nag_entropy_accuracy, marker="o", markersize=4, color=reference, linewidth=1.25, label="NAG by output entropy")
    axes[2].plot(deciles, gpt_accuracy, marker="o", markersize=4, color=reference, linestyle="--", linewidth=1.25, label="GPT by output entropy")
    axes[2].set(
        xlabel="confidence decile (low to high)",
        ylabel="top-1 accuracy",
        xticks=deciles,
    )
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("NAG residual norm as explicit confidence")
    fig.text(
        0.5,
        -0.035,
        f"Correctness AUROC: NAG norm {auroc['nag_norm']:.3f}  |  NAG entropy {auroc['nag_entropy']:.3f}  |  GPT entropy {auroc['gpt_entropy']:.3f}",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"nag_norm_confidence_with_gpt_reference.{extension}", bbox_inches="tight", dpi=220)
    plt.close(fig)

    metrics = {
        "sample_shape": list(token_ids.shape),
        "bin_count": NUM_BINS,
        "nag_norm_accuracy_by_confidence_decile": nag_binned["correct"].tolist(),
        "nag_entropy_accuracy_by_confidence_decile": nag_entropy_accuracy.tolist(),
        "gpt_entropy_accuracy_by_confidence_decile": gpt_accuracy.tolist(),
        "correctness_auroc": auroc,
        "ranking_signals": {
            "nag_norm": "final residual RMS norm (ascending)",
            "nag_entropy": "predictive entropy (descending)",
            "gpt_entropy": "predictive entropy (descending)",
        },
    }
    (OUTPUT / "nag_norm_confidence_with_gpt_reference_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
