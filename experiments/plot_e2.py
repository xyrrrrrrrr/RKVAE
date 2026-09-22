"""Notebook-style E2 trajectory and mean +/- std plots."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rkvae import wrap_angle


def plot(root, system, task, seed=2, trial=0):
    folder = root / system / f"seed_{seed}"
    paths = {m: folder / f"{m}_{task}" for m in ("lifted", "bilinear")}
    counts = [len(json.loads((p / "metrics.json").read_text())["trials"]) for p in paths.values()]
    if len(set(counts)) != 1:
        raise ValueError(f"paired E2 groups have different trial counts: {counts}")
    n = counts[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    with np.load(paths["lifted"] / f"trial_{trial:03}.npz") as linear, np.load(paths["bilinear"] / f"trial_{trial:03}.npz") as bilinear:
        np.testing.assert_allclose(linear["states"][0], bilinear["states"][0], atol=1e-12)
        np.testing.assert_allclose(linear["references"], bilinear["references"], atol=1e-12)
        if system == "franka":
            np.testing.assert_allclose(linear["targets"], bilinear["targets"], atol=1e-12)
            axes[0].plot(linear["targets"][:, 1], linear["targets"][:, 2], "k:", label="Desired")
        else:
            axes[0].axhline(0, color="k", linestyle=":", label="Desired")
        for mode, data, label, color, style in (("lifted", linear, "Linear (lifted)", "tab:blue", "-"),
                                                  ("bilinear", bilinear, "Bilinear", "tab:orange", "--")):
            if system == "franka":
                axes[0].plot(data["ee"][:, 1], data["ee"][:, 2], style, color=color, label=label)
            else:
                t = np.arange(len(data["states"])) * .02
                axes[0].plot(t, wrap_angle(data["states"][:, 0]), style, color=color, label=label + " theta")
            with np.load(paths[mode] / "error_statistics.npz") as stats:
                mean, std = stats["mean"][:, 0], stats["std"][:, 0]
                np.testing.assert_array_equal(stats["count"][:, 0], n)
                axes[1].plot(stats["time"], mean, style, color=color, label=label)
                axes[1].fill_between(stats["time"], mean - std, mean + std, color=color, alpha=.18)
    axes[0].set_title("State response" if system == "damping" else "End-effector trajectory")
    axes[1].set_title(f"Mean +/- 1 SD across {n} trials")
    axes[0].set_xlabel("Time (s)" if system == "damping" else "Y (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angle error (rad)" if system == "damping" else "EE position error (m)")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.suptitle(f"E2 {system.title()} / {task}: {root.name}")
    fig.tight_layout()
    out = root / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{system}_{task}.png", dpi=180); fig.savefig(out / f"{system}_{task}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--seed", type=int, default=2)
    p.add_argument("--systems", nargs="+", choices=("damping", "franka"), default=("damping", "franka"))
    a = p.parse_args()
    tasks = (("damping", "regulation"),) if a.systems == ["damping"] else (("franka", "eight"), ("franka", "star"))
    for system, task in tasks:
        plot(a.root, system, task, a.seed)
