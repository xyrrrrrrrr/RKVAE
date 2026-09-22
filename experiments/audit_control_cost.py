"""Recompute damping costs and expose the trials responsible for their variance."""
import argparse
import json
from pathlib import Path

import numpy as np
from rkvae import tracking_error, wrap_angle


def audit(directory):
    metrics = json.loads((directory / 'metrics.json').read_text())
    if metrics['training_config']['system'] != 'damping':
        raise ValueError('This audit uses the damping state penalty diag(5, .01)')
    rows = []
    for trial in metrics['trials']:
        with np.load(directory / f"trial_{trial['trial']:03}.npz") as data:
            states, refs, controls = data['states'], data['references'], data['controls']
            state_cost = float(np.sum((states[:-1] - refs)**2 * [5., .01]))
            input_cost = float(.1 * np.sum(controls**2))
            terminal_cost = float(np.sum((states[-1] - refs[-1])**2 * [5., .01])) if trial['completed'] else 0.
            periodic_state = float(np.sum(tracking_error(states[:-1], refs, 'damping')**2 * [5., .01]))
            periodic_terminal = float(np.sum(tracking_error(states[-1], refs[-1], 'damping')**2 * [5., .01])) if trial['completed'] else 0.
            periodic_cost = periodic_state + input_cost + periodic_terminal
            expected = periodic_cost if metrics.get('angle_metric') == 'shortest_arc' else state_cost + input_cost + terminal_cost
            np.testing.assert_allclose(expected, trial['cost'], rtol=1e-10, atol=1e-9)
            rows.append(dict(trial=trial['trial'], cost=trial['cost'], success=trial['success'],
                             periodic_cost=periodic_cost,
                             periodic_success=bool(trial['completed'] and np.all(abs(wrap_angle(states[-20:, 0])) <= .1)),
                             initial_state=states[0].tolist(), final_state=states[-1].tolist(),
                             state_cost=state_cost, input_cost=input_cost, terminal_cost=terminal_cost,
                             max_abs_angle=float(np.max(abs(states[:, 0])))))
    costs = np.array([r['cost'] for r in rows])
    deviations = (costs - costs.mean())**2
    for row, deviation in zip(rows, deviations):
        row['variance_share'] = float(deviation / deviations.sum()) if deviations.sum() else 0.
    groups = {}
    # collect/reset alternates narrow and broad initial-state distributions.
    for name, group in [('all', rows), ('narrow', rows[::2]), ('broad', rows[1::2])]:
        values = np.array([r['cost'] for r in group])
        groups[name] = dict(n=len(values), mean=float(values.mean()), std=float(values.std()),
                            median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                            max=float(values.max()))
    return dict(directory=str(directory), groups=groups,
                periodic_mean=float(np.mean([r['periodic_cost'] for r in rows])),
                periodic_std=float(np.std([r['periodic_cost'] for r in rows])),
                periodic_success_rate=float(np.mean([r['periodic_success'] for r in rows])),
                trials=sorted(rows, key=lambda r: r['cost'], reverse=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [audit(directory) for directory in args.directories]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, allow_nan=False))
    for report in reports:
        print(report['directory'], report['groups']['all'], 'periodic:',
              report['periodic_mean'], report['periodic_std'], report['periodic_success_rate'])
