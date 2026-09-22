"""Plot saved Franka Eight/Star trajectories at sigma=0.005."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/e2_franka_e2_wide_norm_sampled_z_sigma001_200ep_cpu/evaluation_gain10_eta001_10trials'
OUT = BASE / 'franka_sigma005_trajectories'

methods = [
    ('DKUC', 'dkuc_{task}_sigma0005', '#e67e22'),
    ('RKVAE-bilinear', 'rkvae_bilinear_{task}_sigma0005', '#2ca02c'),
    ('RKVAE-lifted', 'rkvae_lifted_{task}_sigma0005', '#d62728'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
for row, task in enumerate(('eight', 'star')):
    for col, (label, pattern, color) in enumerate(methods):
        if label == 'DKUC':
            folder = 'dkuc_gain10_eight_sigma0005' if task == 'eight' else 'dkuc_gain10_star_sigma0005'
        elif label == 'RKVAE-bilinear' and task == 'eight':
            folder = 'rkvae_bilinear_rerun_sigma0005'
        elif label == 'RKVAE-lifted' and task == 'eight':
            folder = 'rkvae_lifted_sigma0005'
        else:
            folder = pattern.format(task=task)
        path = BASE / folder / 'trial_000.npz'
        with np.load(path) as data:
            desired = data['targets']
            actual = data['ee']
        ax = axes[row, col]
        ax.plot(desired[:, 1], desired[:, 2], color='#1f77b4', linewidth=1.4,
                label='Desired')
        ax.plot(actual[:, 1], actual[:, 2], color=color, linewidth=1.0,
                label=label)
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'Franka-{task.capitalize()} ({label})')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=9, loc='upper right')

        # Match the notebook's local-zoom view of the lower-left trajectory
        # segment while keeping the same limits for all methods in a task.
        if task == 'eight':
            x_min, x_max, y_min, y_max = -0.24, -0.17, 0.36, 0.44
        else:
            x_min, x_max, y_min, y_max = -0.19, -0.12, 0.35, 0.43
        ax_inset = inset_axes(
            ax, width='42%', height='42%', loc='lower left',
            bbox_to_anchor=(0.14, 0.43, 0.82, 0.82),
            bbox_transform=ax.transAxes, borderpad=0.8,
        )
        ax_inset.plot(desired[:, 1], desired[:, 2], color='#1f77b4', linewidth=0.9)
        ax_inset.plot(actual[:, 1], actual[:, 2], color=color, linewidth=0.75)
        ax_inset.set_xlim(x_min, x_max)
        ax_inset.set_ylim(y_min, y_max)
        ax_inset.grid(alpha=0.3)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.patch.set_facecolor('white')
        ax_inset.patch.set_alpha(0.92)
        mark_inset(ax, ax_inset, loc1=1, loc2=2, fc='none', ec='0.5', linestyle='--', linewidth=0.8)

fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'franka_sigma005_trajectories.png', dpi=220, bbox_inches='tight')
fig.savefig(OUT / 'franka_sigma005_trajectories.pdf', bbox_inches='tight')
plt.close(fig)
print(OUT / 'franka_sigma005_trajectories.png')
