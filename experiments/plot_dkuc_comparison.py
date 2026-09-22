"""Plot DKUC and RKVAE pendulum control comparisons."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results/e2_pendulum_norm_sampled_z_sigma002_theta2_eta001_200ep_gpu")
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
SIGMAS = [0.0, 0.001, 0.002, 0.005]
SERIES = {
    "DKUC (unnormalized)": "control_dkuc_unnorm_sigma{tag}",
    "RKVAE-bilinear": "control_bilinear_sigma{tag}",
    "RKVAE-lifted": "control_lifted_sigma{tag}",
}
TAGS = {0.0: "0", 0.001: "001", 0.002: "002", 0.005: "005"}


def metrics(label, pattern):
    success, costs, moments = [], [], []
    for sigma in SIGMAS:
        data = json.loads((ROOT / pattern.format(tag=TAGS[sigma]) / "metrics.json").read_text())
        success.append(100.0 * data["success_rate"])
        costs.append(data["completed_cost_mean"])
        if "tail_m_mean" in data:
            moments.append(data["tail_m_mean"])
        else:
            trials = []
            for path in sorted((ROOT / pattern.format(tag=TAGS[sigma])).glob("trial_*.npz")):
                states = np.load(path)["states"][-20:]
                angle = (states[:, 0] + np.pi) % (2 * np.pi) - np.pi
                trials.append(np.mean(angle**2 + states[:, 1]**2))
            moments.append(float(np.mean(trials)))
    return np.asarray(success), np.asarray(costs), np.asarray(moments)


values = {label: metrics(label, pattern) for label, pattern in SERIES.items()}
colors = {"DKUC (unnormalized)": "#6b7280", "RKVAE-bilinear": "#2563eb", "RKVAE-lifted": "#dc2626"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
specs = [(0, "MS-success rate (%)", "Success rate"), (1, "Mean control cost", "Control cost"), (2, r"Tail $\hat m$", "Tail empirical second moment")]
for index, ylabel, title in specs:
    ax = axes[index]
    for label, (success, costs, moments) in values.items():
        series = [success, costs, moments][index]
        ax.plot(SIGMAS, series, marker="o", linewidth=2, label=label, color=colors[label])
    ax.set_xlabel("Process-noise sigma")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xticks(SIGMAS)
    ax.ticklabel_format(axis="x", style="plain")
axes[0].axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
axes[2].axhline(0.02, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label=r"$\tau_m=0.02$")
axes[0].legend(fontsize=8)
fig.savefig(OUT / "dkuc_rkvae_noise_comparison.png", dpi=180)
plt.close(fig)

sigma = 0.005
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
for label, pattern in SERIES.items():
    path = ROOT / pattern.format(tag=TAGS[sigma]) / "trial_000.npz"
    states = np.load(path)["states"]
    angle = (states[:, 0] + np.pi) % (2 * np.pi) - np.pi
    axes[0].plot(angle, label=label, color=colors[label])
    axes[1].plot(states[:, 1], label=label, color=colors[label])
axes[0].axhline(0.1, color="black", linestyle="--", linewidth=0.8)
axes[0].axhline(-0.1, color="black", linestyle="--", linewidth=0.8)
axes[0].set_ylabel(r"$\theta$ (rad)")
axes[1].set_ylabel(r"$\dot\theta$ (rad/s)")
axes[1].set_xlabel("Step")
axes[0].set_title(r"Representative trajectories at $\sigma=0.005$")
axes[0].legend()
for ax in axes:
    ax.grid(alpha=0.3)
fig.savefig(OUT / "trajectories_sigma005.png", dpi=180)
plt.close(fig)

print(OUT / "dkuc_rkvae_noise_comparison.png")
print(OUT / "trajectories_sigma005.png")
