"""Frozen three-seed E1: independent prediction data and paired, serial control runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import torch

from rkvae import System, collect, make_net, tracking_error

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1, 2, 3)


def checkpoint(root, system, seed):
    folder = (ROOT / 'results/e1_retrain' if seed == 1 else root) / system / f'seed_{seed}' / 'train'
    if not (folder / 'loss.png').exists():
        raise RuntimeError(f'Training has not finished: {folder}')
    return folder / 'best.pt'


def plot_prediction(root, name, results, seed=2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for mode, color in (('lifted', 'tab:blue'), ('bilinear', 'tab:orange')):
        row, = [r for r in results if r['mode'] == mode and r['seed'] == seed]
        draw = ax.semilogy if name == 'damping' else ax.plot
        draw(np.arange(1,16), row['rollout_normalized_rmse'],
             '-' if mode == 'lifted' else '--', color=color,
             label='Linear (lifted)' if mode == 'lifted' else 'Bilinear')
    title = f'{name.title()}: open-loop prediction'
    ax.set(xlabel='Prediction horizon (steps)', ylabel='Normalized RMSE', title=title)
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(root/name/'prediction.png', dpi=160); plt.close(fig)
    (root/name/'prediction_display.json').write_text(json.dumps(dict(training_seed=seed,
        selection='User-selected display run; prediction.json retains all seeds'), indent=2))


def prediction(root):
    torch.set_num_threads(1)
    for name in ('damping', 'franka'):
        system = System(name)
        try:
            path = root / name / 'prediction_test.npy'
            if not path.exists():
                np.save(path, collect(system, 20000, 15, 0., 700001))
            data = np.load(path)
            x = data[..., system.m:]
            if name == 'damping':
                x = tracking_error(x * system.scale + system.center, system.center, name) / system.scale
            results = []
            for seed in SEEDS:
                saved = torch.load(checkpoint(root, name, seed), map_location='cpu', weights_only=False)
                net = make_net(saved['config'], 'cpu'); net.load_state_dict(saved['model']); net.eval()
                for mode in ('lifted', 'bilinear'):
                    squared, rollout = [], []
                    with torch.no_grad():
                        for start in range(0, 20000, 512):
                            xt = torch.tensor(x[:, start:start+512], dtype=torch.float64)
                            u = torch.tensor(data[:-1, start:start+512, :system.m], dtype=torch.float64)
                            mu, _, _ = net.encode_only(xt)
                            lifted = torch.cat((xt, mu), dim=-1)
                            estimate = net.predict(lifted[:-1], u, mode)[..., :system.n].numpy()
                            error = tracking_error(estimate * system.scale, xt[1:].numpy() * system.scale, name)
                            squared.append(np.mean((error / system.scale)**2, axis=0))
                            z = lifted[0]
                            errors = []
                            for k in range(15):
                                z = net.predict(z, u[k], mode)
                                error = tracking_error(z[..., :system.n].numpy() * system.scale,
                                                       xt[k+1].numpy() * system.scale, name)
                                errors.append((error / system.scale)**2)
                            rollout.append(np.stack(errors, axis=1))
                    one = np.concatenate(squared); multi = np.concatenate(rollout)
                    if not np.isfinite(one).all() or not np.isfinite(multi).all():
                        raise FloatingPointError(f'Nonfinite prediction errors: {name}/{seed}/{mode}')
                    result = dict(seed=seed, mode=mode, checkpoint_epoch=saved['epoch'],
                                  one_step_normalized_rmse=float(np.sqrt(one.mean())),
                                  one_step_physical_rmse=(np.sqrt(one.mean(axis=0))*system.scale).tolist(),
                                  rollout_normalized_rmse=np.sqrt(multi.mean(axis=(0, 2))).tolist(),
                                  rollout_physical_rmse=(np.sqrt(multi.mean(axis=0))*system.scale).tolist())
                    results.append(result)
                    print('PREDICTION', name, seed, mode, result['one_step_normalized_rmse'], flush=True)
            (root / name / 'prediction.json').write_text(json.dumps(dict(
                samples=20000, transitions=15, sigma=0., seed=700001,
                data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rollout_definition='Open-loop lifted recurrence, no teacher forcing, errors in physical state projection',
                results=results), indent=2))
            plot_prediction(root, name, results)
        finally:
            system.close()


def control(root, systems=('damping', 'franka'), seeds=SEEDS):
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR='/tmp/matplotlib-rkvae')
    for system in systems:
        for seed in seeds:
            for task in (('regulation',) if system == 'damping' else ('eight', 'star')):
                for mode in ('lifted', 'bilinear', 'zero') if seed == 1 else ('lifted', 'bilinear'):
                    folder = root / system / f'seed_{seed}' / f'{mode}_{task}'
                    if (folder / 'metrics.json').exists() and len(json.loads((folder/'metrics.json').read_text())['trials']) >= 100:
                        print('EXISTS', folder, flush=True)
                        continue
                    command = [sys.executable, str(ROOT/'experiments/rkvae.py'), 'evaluate',
                               '--checkpoint', str(checkpoint(root, system, seed)),
                               '--mode', 'lifted' if mode == 'zero' else mode,
                               '--controller', 'zero' if mode == 'zero' else 'rkvae',
                               '--resume', '--sigma', '0', '--seed', '10001', '--threads', '1',
                               '--angle-coordinates', 'periodic', '--eta', '0',
                               '--trials', '100',
                               '--horizon', '200' if system == 'damping' else '3000',
                               '--task', 'eight' if task == 'regulation' else task, '--output', str(folder)]
                    print('CONTROL', system, seed, task, mode, flush=True)
                    with (folder.parent / f'{mode}_{task}.log').open('w') as log:
                        subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def report(root):
    runs = []
    for system in ('damping', 'franka'):
        for seed in SEEDS:
            for task in (('regulation',) if system == 'damping' else ('eight', 'star')):
                for mode in ('lifted', 'bilinear', 'zero') if seed == 1 else ('lifted', 'bilinear'):
                    folder = root / system / f'seed_{seed}' / f'{mode}_{task}'
                    d = json.loads((folder/'metrics.json').read_text())
                    assert d['config']['lifted_capture_angle'] is None
                    assert d['config']['seed'] == 10001
                    baseline = root / system / 'seed_1' / f'zero_{task}'
                    latencies = []
                    for trial in d['trials']:
                        path = f"trial_{trial['trial']:03}.npz"
                        paired = baseline if (baseline/path).exists() else root/system/f'seed_{seed}'/f'lifted_{task}'
                        with np.load(folder/path) as a, np.load(paired/path) as b:
                            np.testing.assert_allclose(a['states'][0], b['states'][0], rtol=0, atol=1e-12)
                            count = len(a['references'])
                            np.testing.assert_allclose(a['references'], b['references'][:count], rtol=0, atol=1e-12)
                            if system == 'franka':
                                np.testing.assert_allclose(a['targets'], b['targets'][:len(a['targets'])], rtol=0, atol=1e-12)
                            latencies.extend(a['latency'][10:]*1000)
                    costs = [r['cost'] for r in d['trials'] if r['completed']]
                    run = dict(system=system, seed=seed, task=task, mode=mode,
                               checkpoint_epoch=d['checkpoint_epoch'], trials=len(d['trials']),
                               completion=d['completion_rate'], success=d['success_rate'],
                               cost_mean=float(np.mean(costs)), cost_std=float(np.std(costs)),
                               latency_median_ms=float(np.median(latencies)),
                               latency_p95_ms=float(np.quantile(latencies,.95)),
                               gain_setup_ms=d['gain_setup_ms'], rows=d['trials'])
                    if system == 'franka':
                        rmses = [r['position_rmse_m'] for r in d['trials'] if r['completed']]
                        run['position_rmse_mean_m'] = float(np.mean(rmses))
                        run['position_rmse_std_m'] = float(np.std(rmses))
                    else:
                        run['initial_groups'] = {}
                        for group in ('narrow', 'broad'):
                            rows = [r for r in d['trials'] if r['initial_group'] == group]
                            run['initial_groups'][group] = dict(
                                cost_mean=float(np.mean([r['cost'] for r in rows if r['completed']])),
                                cost_std=float(np.std([r['cost'] for r in rows if r['completed']])),
                                success=float(np.mean([r['success'] for r in rows])))
                    runs.append(run)
    (root/'summary.json').write_text(json.dumps(runs, indent=2, allow_nan=False))
    lines = ['# E1 Validation', '',
             'Three training seeds (1, 2, 3); zero input is shared, not three independent models.',
             'Cost SD across trials differs from SD across training-seed means. Both use ddof=0.',
             'All controls have sigma=0, eta=0, no capture/PD. Damping errors are periodic.',
             'Franka reference IK is precomputed identically for all methods. Paired arrays were checked.',
             'Trial counts are reported per seed. Trials beyond the shared zero baseline are paired between learned methods only.',
             'The aggregate table uses the common trial count across all seeds/methods per task; per-seed tables use all available trials.', '',
             '| System/task | Method | Trials per seed | Trial cost mean +/- SD | Across-seed cost mean +/- SD | Success | EE RMSE (m) |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for system, task in (('damping','regulation'), ('franka','eight'), ('franka','star')):
        common_trials = min(r['trials'] for r in runs if (r['system'], r['task']) == (system, task))
        for mode in ('lifted','bilinear','zero'):
            selected = [r for r in runs if (r['system'],r['task'],r['mode']) == (system,task,mode)]
            costs = [x['cost'] for r in selected for x in r['rows'][:common_trials] if x['completed']]
            means = [np.mean([x['cost'] for x in r['rows'][:common_trials] if x['completed']]) for r in selected]
            across = f'{np.mean(means):.4f} +/- {np.std(means):.4f}' if mode != 'zero' else 'shared baseline'
            success = f"{100*np.mean([r['success'] for r in selected]):.1f}%" if system == 'damping' else 'N/A'
            ee = f"{np.mean([x['position_rmse_m'] for r in selected for x in r['rows'][:common_trials] if x['completed']]):.6f}" if system == 'franka' else 'N/A'
            lines.append(f'| {system}/{task} | {mode} | {common_trials} | {np.mean(costs):.4f} +/- {np.std(costs):.4f} | {across} | {success} | {ee} |')
    lines += ['', '## Per-Seed Control Timing', '',
              '| System/task | Seed | Method | Completion | Median ms | P95 ms | Setup ms |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for r in runs:
        lines.append(f"| {r['system']}/{r['task']} | {r['seed']} | {r['mode']} | {r['completion']:.3f} | {r['latency_median_ms']:.3f} | {r['latency_p95_ms']:.3f} | {r['gain_setup_ms']:.3f} |")
    lines += ['', '## Franka Per-Seed Tracking', '',
              '| Task | Seed | Mode | Trials | Cost mean +/- trial SD | Position RMSE mean +/- trial SD (m) |',
              '| --- | --- | --- | --- | --- | --- |']
    for r in runs:
        if r['system'] == 'franka':
            lines.append(f"| {r['task']} | {r['seed']} | {r['mode']} | {r['trials']} | {r['cost_mean']:.4f} +/- {r['cost_std']:.4f} | {r['position_rmse_mean_m']:.6f} +/- {r['position_rmse_std_m']:.6f} |")
    lines += ['', '## Damping Initial-State Groups', '',
              '| Seed | Mode | Group | Cost mean +/- trial SD | Success |',
              '| --- | --- | --- | --- | --- |']
    for r in runs:
        for group, values in r.get('initial_groups', {}).items():
            lines.append(f"| {r['seed']} | {r['mode']} | {group} | {values['cost_mean']:.4f} +/- {values['cost_std']:.4f} | {100*values['success']:.1f}% |")
    lines += ['', '## Training', '', '| System | Seed | Epochs run | Selected epoch | Validation prediction loss |',
              '| --- | --- | --- | --- | --- |']
    for system in ('damping', 'franka'):
        for seed in SEEDS:
            path = checkpoint(root, system, seed)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            history = [json.loads(line) for line in (path.parent/'history.jsonl').read_text().splitlines()]
            lines.append(f"| {system} | {seed} | {history[-1]['epoch']} | {saved['epoch']} | {saved['score']:.6f} |")
    lines += ['', '## Prediction', '', '| System | Seed | Mode | One-step NRMSE | 5-step NRMSE | 15-step NRMSE |',
              '| --- | --- | --- | --- | --- | --- |']
    for system in ('damping','franka'):
        predictions = json.loads((root/system/'prediction.json').read_text())['results']
        plot_prediction(root, system, predictions)
        for r in predictions:
            lines.append(f"| {system} | {r['seed']} | {r['mode']} | {r['one_step_normalized_rmse']:.6f} | {r['rollout_normalized_rmse'][4]:.6f} | {r['rollout_normalized_rmse'][14]:.6f} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    postprocessing = root/'source_postprocessing'
    postprocessing.mkdir(exist_ok=True)
    for name in ('validate_e1.py', 'inspect_e1_prediction.py'):
        shutil.copy2(ROOT/'experiments'/name, postprocessing/name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'results/e1_validation_v2')
    parser.add_argument('--stage', choices=['prediction','control','report','all'], default='all')
    parser.add_argument('--systems', nargs='+', choices=['damping', 'franka'], default=['damping', 'franka'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for system in ('damping', 'franka'):
        for seed in SEEDS:
            checkpoint(args.root, system, seed)
            (args.root/system/f'seed_{seed}').mkdir(parents=True, exist_ok=True)
    if not (args.root/'protocol.json').exists():
        files = ['experiments/rkvae.py', 'experiments/validate_e1.py', 'train/Learn_Kovae_with_KlinearEig.py',
                 'utility/Utility.py', 'franka/franka_env.py']
        hashes = {}
        for path in files:
            source = ROOT/path; target = args.root/'source'/path
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source,target)
            hashes[path] = hashlib.sha256(source.read_bytes()).hexdigest()
        (args.root/'protocol.json').write_text(json.dumps(dict(
            training_seeds=SEEDS, control_seed=10001, prediction_seed=700001,
            samples=50000, validation_samples=20000, prediction_samples=20000,
            sigma_train=.05, sigma_test=0., max_epochs=100, control_threads=1,
            python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
            cpu=platform.processor(), sources=hashes), indent=2))
    for stage in ('prediction', 'control', 'report') if args.stage == 'all' else (args.stage,):
        if stage == 'control':
            control(args.root, args.systems, args.seeds)
        else:
            globals()[stage](args.root)
