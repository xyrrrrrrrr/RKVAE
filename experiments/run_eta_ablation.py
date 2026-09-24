"""Run and summarize an eta ablation for the lifted damping controller."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def summarize_trial_directory(directory, tau):
    metrics = json.loads((directory / "metrics.json").read_text())
    costs = []
    tails = []
    for row in metrics["trials"]:
        with np.load(directory / f"trial_{row['trial']:03d}.npz") as data:
            state = np.asarray(data["states"])[-20:]
        tails.append(float(np.mean(state[:, 0] ** 2 + state[:, 1] ** 2)))
        costs.append(float(row["cost"]))
    tails = np.asarray(tails)
    costs = np.asarray(costs)
    return {
        "eta": float(metrics["config"]["eta"]),
        "trials": int(len(tails)),
        "success_rate_percent": float(100 * np.mean(tails <= tau)),
        "tail_m_mean": float(np.mean(tails)),
        "tail_m_std": float(np.std(tails)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "output": str(directory),
    }


def main(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for eta in args.etas:
        tag = f"eta{eta:g}".replace(".", "p")
        directory = output_root / tag
        command = [
            sys.executable, str(ROOT / "experiments/rkvae.py"), "evaluate",
            "--checkpoint", str(args.checkpoint), "--mode", "lifted",
            "--controller", "rkvae", "--eta", str(eta),
            "--gain-scale", str(args.gain_scale), "--control-r", str(args.control_r),
            "--initial-angle-limit", str(args.initial_angle_limit),
            "--trials", str(args.trials), "--horizon", str(args.horizon),
            "--seed", str(args.seed), "--sigma", str(args.sigma),
            "--threads", str(args.threads), "--output", str(directory),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        rows.append(summarize_trial_directory(directory, args.tau))

    # Prefer stability, then lower residual fluctuation, then lower cost.
    ranked = sorted(rows, key=lambda row: (
        -row["success_rate_percent"], row["tail_m_mean"], row["cost_mean"]))
    report = {
        "protocol": {
            "checkpoint": str(args.checkpoint), "mode": "lifted",
            "sigma": args.sigma, "tau_m": args.tau, "trials": args.trials,
            "horizon": args.horizon, "seed": args.seed,
            "gain_scale": args.gain_scale, "control_r": args.control_r,
        },
        "ranking": ranked,
        "best_eta": ranked[0]["eta"],
    }
    (output_root / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--etas", nargs="+", type=float,
                        default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--gain-scale", type=float, default=1.0)
    parser.add_argument("--control-r", type=float, default=0.1)
    parser.add_argument("--initial-angle-limit", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=1)
    main(parser.parse_args())
