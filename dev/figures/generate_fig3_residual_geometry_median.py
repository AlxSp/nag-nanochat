"""Median-norm variant of the local Figure 3 residual-geometry plot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output/fig3_residual_geometry"
ROBUST_ARRAYS = OUTPUT / "experimental_norm_bos_robustness_arrays.npz"
GEOMETRY_ARRAYS = OUTPUT / "fig3_residual_geometry_arrays.npz"


def main():
    robust = np.load(ROBUST_ARRAYS)
    geometry = np.load(GEOMETRY_ARRAYS)

    baseline_norm = robust["baseline_median"]
    nag_norm = robust["nag_median"]
    baseline_rotation = geometry["baseline_cumulative_rotation"]
    nag_rotation = geometry["nag_cumulative_rotation"]

    colors = {"baseline": "#d84b3e", "nag": "#2864a8"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    norm_layers = np.arange(len(baseline_norm))
    axes[0].plot(
        norm_layers,
        baseline_norm,
        color=colors["baseline"],
        linewidth=2,
        label="Baseline GPT",
    )
    axes[0].plot(
        norm_layers,
        nag_norm,
        color=colors["nag"],
        linewidth=2,
        label="NAG",
    )
    axes[0].set(
        title="Residual-stream norm",
        xlabel="layer (attention / MLP interleaved)",
        ylabel="median token RMS norm",
    )
    axes[0].grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.65)
    axes[0].legend(frameon=False)

    rotation_layers = np.arange(1, len(baseline_rotation) + 1)
    axes[1].plot(
        rotation_layers,
        baseline_rotation,
        color=colors["baseline"],
        linewidth=2,
        label="Baseline GPT",
    )
    axes[1].plot(
        rotation_layers,
        nag_rotation,
        color=colors["nag"],
        linewidth=2,
        label="NAG",
    )
    axes[1].set(
        title="Cumulative residual rotation",
        xlabel="layer (attention / MLP interleaved)",
        ylabel="cumulative rotation (degrees)",
    )
    axes[1].grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.65)
    axes[1].legend(frameon=False)

    fig.suptitle("Residual geometry through depth")
    for extension in ("png", "svg"):
        fig.savefig(
            OUTPUT / f"fig3_residual_norm_median_and_rotation.{extension}",
            bbox_inches="tight",
            dpi=220,
        )
    plt.close(fig)

    print(f"Baseline median norm: {baseline_norm[-2]:.3f} -> {baseline_norm[-1]:.3f} at final MLP")
    print(f"NAG final median norm: {nag_norm[-1]:.3f}")


if __name__ == "__main__":
    main()
