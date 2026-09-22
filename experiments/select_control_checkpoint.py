"""Select damping deployment checkpoints on a fixed independent control set."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil
import torch


def rank(metrics):
    """Prefer completion, then success, then lower cost; ties keep earlier epoch."""
    rows = metrics['trials']
    return (sum(r['completed'] for r in rows) / len(rows),
            sum(bool(r['success']) for r in rows) / len(rows),
            -sum(r['cost'] for r in rows) / len(rows))


def run(args):
    if args.every < 1 or args.trials < 1 or args.horizon < 1 or any(s < 0 for s in args.sigmas):
        raise ValueError('Invalid validation size or noise')
    if args.seed < 2000000:
        raise ValueError('Control validation uses reserved seeds >= 2000000')
    args.output.mkdir(parents=True, exist_ok=False)
    candidates = sorted((args.train_dir/'candidates').glob('epoch_*.pt'))
    candidates = [p for p in candidates if int(p.stem.split('_')[1]) % args.every == 0]
    candidates += [args.train_dir/name for name in ('best.pt', 'last.pt')]
    models = {}
    for path in candidates:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['config']['system'] != 'damping':
            raise ValueError('This selector currently supports damping only')
        models.setdefault(saved['epoch'], (path, saved['config']))
    report = {'protocol': vars(args).copy(), 'candidates': [], 'selected': {}}
    report['protocol'] = {k: str(v) if isinstance(v, Path) else v for k,v in report['protocol'].items()}
    report['protocol']['ranking'] = 'completion rate, success rate, negative mean cost; earlier epoch wins ties'
    for mode in ('lifted', 'bilinear'):
        winner = None
        for epoch, (checkpoint, config) in sorted(models.items()):
            if config['variant'] in ('lifted', 'bilinear') and config['variant'] != mode:
                continue
            rows = []
            rejected = False
            for sigma in args.sigmas:
                output = args.output/f'epoch_{epoch:04d}'/f'{mode}_sigma{sigma:g}'
                command = [sys.executable, str(Path(__file__).with_name('rkvae.py')), 'evaluate',
                           '--checkpoint', str(checkpoint), '--mode', mode, '--sigma', str(sigma),
                           '--seed', str(args.seed), '--trials', str(args.trials),
                           '--horizon', str(args.horizon), '--eta', '0', '--output', str(output)]
                output.parent.mkdir(parents=True, exist_ok=True)
                with (output.parent/f'{output.name}.log').open('w') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    log_text = (output.parent/f'{output.name}.log').read_text()
                    if 'LinAlgError' not in log_text:
                        raise RuntimeError(f'Validation failed; see {output.parent}/{output.name}.log')
                    report['candidates'].append(dict(mode=mode, epoch=epoch, rejected='LQR solve failed'))
                    rejected = True
                    break
                metrics = json.loads((output/'metrics.json').read_text())
                if len(metrics['trials']) != args.trials:
                    raise ValueError('Incomplete validation')
                rows.extend(metrics['trials'])
            if rejected:
                continue
            score = rank({'trials': rows})
            entry = dict(mode=mode, epoch=epoch, checkpoint=str(checkpoint), rank=score)
            report['candidates'].append(entry)
            print(entry, flush=True)
            if winner is None or score > tuple(winner['rank']):
                winner = entry
            (args.output/'selection.json').write_text(json.dumps(report, indent=2))
        if winner is not None:
            target = args.output/f'deploy_{mode}.pt'
            shutil.copy2(winner['checkpoint'], target)
            report['selected'][mode] = winner | {'deployment_checkpoint': str(target)}
    (args.output/'selection.json').write_text(json.dumps(report, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2000001)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--every', type=int, default=10)
    p.add_argument('--horizon', type=int, default=200)
    p.add_argument('--sigmas', nargs='+', type=float, default=[0, .0005, .001, .002])
    return p


if __name__ == '__main__':
    run(parser().parse_args())
