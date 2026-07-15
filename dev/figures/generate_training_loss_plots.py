"""Plot matched GPT/NAG training-loss histories from W&B logs."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARENT = Path(__file__).resolve().parent
REPO = PARENT.parents[1]
OUTPUT = PARENT / "output/training_losses"
TOKENS_PER_STEP = 1_048_576

RUNS = {
    "gpt": REPO / "runs/wandb/gpt_d64_w640_3e19/files/output.log",
    "nag": REPO / "runs/wandb/nag_gpt_d64_w640_3e19_gatefix/files/output.log",
    "collapsed_resume": REPO / "runs/wandb/nag_gpt_d64_w640_3e19/files/output.log",
}

STEP_RE = re.compile(r"^step (\d+)/(\d+).*?\| loss: ([0-9.]+)")


def parse_history(path):
    points = []
    for line in path.read_text().splitlines():
        match = STEP_RE.search(line)
        if match:
            points.append((int(match.group(1)), float(match.group(3))))
    if not points:
        raise RuntimeError(f"No training losses found in {path}")
    array = np.asarray(points, dtype=np.float64)
    return {
        "step": array[:, 0].astype(np.int64),
        "tokens_b": array[:, 0] * TOKENS_PER_STEP / 1e9,
        "loss": array[:, 1],
    }


def wandb_history(run_id):
    import wandb

    run = wandb.Api(timeout=60).run(f"alxsp/nanochat/{run_id}")
    points = []
    for row in run.scan_history(keys=["step", "train/loss"], page_size=10_000):
        step, loss = row.get("step"), row.get("train/loss")
        if step is not None and loss is not None and np.isfinite(loss):
            points.append((int(step), float(loss)))
    array = np.asarray(points, dtype=np.float64)
    order = np.argsort(array[:, 0])
    array = array[order]
    return {
        "step": array[:, 0].astype(np.int64),
        "tokens_b": array[:, 0] * TOKENS_PER_STEP / 1e9,
        "loss": array[:, 1],
    }


def merge_histories(first, second):
    cutoff = second["step"][0]
    keep = first["step"] < cutoff
    return {
        field: np.concatenate((first[field][keep], second[field]))
        for field in ("step", "tokens_b", "loss")
    }


def smooth_history(history, step_bin=100, window=1):
    bin_ids = history["step"] // step_bin
    unique_bins = np.unique(bin_ids)
    steps = np.asarray([history["step"][bin_ids == bin_id].mean() for bin_id in unique_bins])
    losses = np.asarray([history["loss"][bin_ids == bin_id].mean() for bin_id in unique_bins])
    if len(losses) < window:
        return {"tokens_b": steps * TOKENS_PER_STEP / 1e9, "loss": losses}
    kernel = np.ones(window) / window
    return {
        "tokens_b": np.convolve(steps, kernel, mode="valid") * TOKENS_PER_STEP / 1e9,
        "loss": np.convolve(losses, kernel, mode="valid"),
    }


def draw(histories, include_collapsed, filename):
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    curves = [
        ("gpt", "Baseline GPT", "#d84b3e", "-", 1.25),
        ("nag", "NAG", "#2864a8", "-", 1.25),
    ]
    if include_collapsed:
        curves.append(("collapsed", "NAG (gate collapse)", "#7f96b0", "--", 1.15))
    for key, label, color, linestyle, linewidth in curves:
        history = smooth_history(histories[key])
        ax.plot(
            history["tokens_b"],
            history["loss"],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    ax.set(
        xlim=(0.25, 10.05),
        ylim=(2.4, 4.0),
        xlabel="training tokens (billions)",
        ylabel="debiased EMA training loss",
        title="64-layer, width-640 training loss",
    )
    ax.set_axisbelow(True)
    ax.grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / f"{filename}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{filename}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    histories = {key: parse_history(path) for key, path in RUNS.items()}
    histories["collapsed"] = merge_histories(
        wandb_history("v4cqn023"), histories["collapsed_resume"]
    )
    draw(histories, include_collapsed=False, filename="training_loss_gpt_vs_nag")
    draw(histories, include_collapsed=True, filename="training_loss_with_collapsed_nag")
    np.savez_compressed(
        OUTPUT / "training_loss_histories.npz",
        **{
            f"{key}_{field}": values
            for key, history in histories.items()
            for field, values in history.items()
        },
    )
    metrics = {
        key: {
            "first_step": int(history["step"][0]),
            "last_step": int(history["step"][-1]),
            "num_points": len(history["step"]),
            "final_loss": float(history["loss"][-1]),
        }
        for key, history in histories.items()
    }
    (OUTPUT / "training_loss_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
