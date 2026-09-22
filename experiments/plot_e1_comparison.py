"""Overlay linear (lifted) and bilinear on paired E1 trajectories and error bands."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from rkvae import wrap_angle


def plot_comparison(root, system, task, trial=0, seed=2):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), squeeze=False)
    for column, seed in enumerate((seed,)):
        folder = root / system / f'seed_{seed}'
        paths = {mode: folder / f'{mode}_{task}' for mode in ('lifted', 'bilinear')}
        counts = [len(json.loads((path/'metrics.json').read_text())['trials']) for path in paths.values()]
        if len(set(counts)) != 1:
            raise ValueError(f'Finish paired evaluations before plotting: {counts}')
        with np.load(paths['lifted'] / f'trial_{trial:03}.npz') as linear, np.load(paths['bilinear'] / f'trial_{trial:03}.npz') as bilinear:
            np.testing.assert_allclose(linear['states'][0], bilinear['states'][0], atol=1e-12)
            np.testing.assert_allclose(linear['references'], bilinear['references'], atol=1e-12)
            ax = axes[0, column]
            if system == 'franka':
                np.testing.assert_allclose(linear['targets'], bilinear['targets'], atol=1e-12)
                ax.plot(linear['targets'][:, 1], linear['targets'][:, 2], color='black', linestyle=':', label='Desired')
            else:
                ax.axhline(0, color='black', linestyle=':', label='Desired')
            for mode, data, label, color, style in (
                    ('lifted', linear, 'Linear (lifted)', 'tab:blue', '-'),
                    ('bilinear', bilinear, 'Bilinear', 'tab:orange', '--')):
                if system == 'franka':
                    ax.plot(data['ee'][:, 1], data['ee'][:, 2], style, color=color, label=label)
                    ax.set(xlabel='Y (m)', ylabel='Z (m)')
                else:
                    t = np.arange(len(data['states']))*.02
                    ax.plot(t, wrap_angle(data['states'][:, 0]), style, color=color, label=label+' theta')
                    ax.plot(t, data['states'][:, 1], style, color=color, alpha=.5, label=label+' velocity')
                    ax.set(xlabel='Time (s)', ylabel='State (rad; rad/s)')
                with np.load(paths[mode] / 'error_statistics.npz') as stats:
                    index = 0  # damping angular error, or Franka Euclidean EE error
                    mean, std = stats['mean'][:, index], stats['std'][:, index]
                    error_ax = axes[0, 1]
                    error_ax.plot(stats['time'], mean, style, color=color, label=label)
                    error_ax.fill_between(stats['time'], mean-std, mean+std, color=color, alpha=.18)
                    n = stats['count'][:, index]
                    trials = len(json.loads((paths[mode]/'metrics.json').read_text())['trials'])
                    assert np.all(n == trials)
            ax.set_title('State response' if system == 'damping' else 'End-effector trajectory')
            axes[0, 1].set(xlabel='Time (s)', ylabel='Angle error (rad)' if system == 'damping' else 'EE position error (m)',
                                title=f'Mean +/- 1 SD across {trials} trials')
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.suptitle(f'{system.title()} / {task}: Linear (lifted) vs Bilinear')
    fig.tight_layout()
    out = root / 'comparison'
    out.mkdir(exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'{system}_{task}.{extension}', dpi=180)
    (out / f'{system}_{task}.json').write_text(json.dumps(dict(
        training_seed=seed, trial=trial, selection='User-selected display run; aggregate results retain all seeds',
        band='Mean +/- population SD over all trials of this checkpoint',
        sources={key: str(value) for key, value in paths.items()}), indent=2))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/e1_validation_v2'))
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    for system, task in (('damping', 'regulation'), ('franka', 'eight'), ('franka', 'star')):
        plot_comparison(args.root, system, task, args.trial, args.seed)
        print('Saved', system, task)
