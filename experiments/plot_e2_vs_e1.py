"""Side-by-side E1 (clean) and E2 (process-noise) notebook-style plots."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rkvae import wrap_angle


def row(ax_path, ax_err, system, task, trial=0):
    with np.load(ax_path['lifted']/f'trial_{trial:03}.npz') as linear, np.load(ax_path['bilinear']/f'trial_{trial:03}.npz') as bilinear:
        np.testing.assert_allclose(linear['states'][0], bilinear['states'][0], atol=1e-12)
        if system == 'franka':
            ax_path['_ax'].plot(linear['targets'][:,1], linear['targets'][:,2], 'k:', label='Desired')
        else:
            ax_path['_ax'].axhline(0, color='k', linestyle=':', label='Desired')
        for mode, data, label, color, style in (('lifted',linear,'Linear (lifted)','tab:blue','-'),('bilinear',bilinear,'Bilinear','tab:orange','--')):
            if system == 'franka':
                ax_path['_ax'].plot(data['ee'][:,1], data['ee'][:,2], style, color=color, label=label)
            else:
                t=np.arange(len(data['states']))*.02
                ax_path['_ax'].plot(t, wrap_angle(data['states'][:,0]), style, color=color, label=label)
            with np.load(ax_path[mode]/'error_statistics.npz') as stats:
                n=int(stats['count'][0,0]); mean=stats['mean'][:,0]; std=stats['std'][:,0]
                ax_err.plot(stats['time'],mean,style,color=color,label=label)
                ax_err.fill_between(stats['time'],mean-std,mean+std,color=color,alpha=.18)
    return n


def plot(e1, e2, system, task, seed=2):
    fig, axes = plt.subplots(2,2,figsize=(13,8),squeeze=False)
    for r,(root,title) in enumerate(((e1,'E1: sigma=0'),(e2,'E2: sigma=0.05'))):
        folder=root/system/f'seed_{seed}'; paths={m:folder/f'{m}_{task}' for m in ('lifted','bilinear')}
        if not all((p/'metrics.json').exists() for p in paths.values()): raise FileNotFoundError(paths)
        counts=[len(json.loads((p/'metrics.json').read_text())['trials']) for p in paths.values()]
        if len(set(counts))!=1: raise ValueError(f'incomplete paired groups: {counts}')
        paths['_ax']=axes[r,0]; n=row(paths,axes[r,1],system,task)
        axes[r,0].set_title(f'{title}: representative trajectory')
        axes[r,1].set_title(f'{title}: mean +/- 1 SD across {n} trials')
        axes[r,0].set_xlabel('Time (s)' if system=='damping' else 'Y (m)')
        axes[r,1].set_xlabel('Time (s)')
        axes[r,1].set_ylabel('Angle error (rad)' if system=='damping' else 'EE position error (m)')
        for ax in axes[r]: ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle(f'{system.title()} / {task}: clean vs process noise')
    fig.tight_layout(); out=e2/'figures'; out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{system}_{task}_e1_vs_e2.png',dpi=180); fig.savefig(out/f'{system}_{task}_e1_vs_e2.pdf'); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--e1',type=Path,default=Path('results/e1_validation_v2')); p.add_argument('--e2',type=Path,required=True); p.add_argument('--seed',type=int,default=2); p.add_argument('--systems',nargs='+',default=['damping','franka']); a=p.parse_args()
    for system,task in (('damping','regulation'),('franka','eight')):
        if system in a.systems: plot(a.e1,a.e2,system,task,a.seed)
