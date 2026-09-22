"""E2 process-noise evaluation using existing E1 checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def checkpoint(root, system, seed):
    source = ROOT / "results/e1_retrain" if seed == 1 else root
    return source / system / f"seed_{seed}" / "train" / "best.pt"


def run(args):
    root = Path(args.output_root) / f"sigma_{args.sigma:g}"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLCONFIGDIR="/tmp/matplotlib-rkvae")
    for system in args.systems:
        tasks = ("regulation",) if system == "damping" else ("eight", "star")
        for seed in args.seeds:
            for task in tasks:
                for mode in ("lifted", "bilinear"):
                    out = root / system / f"seed_{seed}" / f"{mode}_{task}"
                    metrics = out / "metrics.json"
                    if metrics.exists() and len(json.loads(metrics.read_text())["trials"]) >= args.trials:
                        print("EXISTS", out, flush=True)
                        continue
                    out.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, str(ROOT / "experiments/rkvae.py"), "evaluate",
                               "--checkpoint", str(checkpoint(args.e1_root, system, seed)),
                               "--mode", mode, "--controller", "rkvae",
                               "--sigma", str(args.sigma), "--seed", str(args.control_seed), "--eta", str(args.eta),
                               "--gain-scale", str(args.gain_scale),
                               "--threads", "1", "--angle-coordinates", "periodic",
                               "--trials", str(args.trials), "--horizon",
                               str(200 if system == "damping" else 3000), "--output", str(out)]
                    if system == "franka":
                        command[command.index("--controller") + 2:command.index("--controller") + 2] = ["--task", task]
                    if metrics.exists():
                        command.insert(command.index("--sigma"), "--resume")
                    print("RUN", system, seed, task, mode, flush=True)
                    subprocess.run(command, check=True, env=env)
    protocol = dict(stage="E2", sigma=args.sigma, systems=args.systems, seeds=args.seeds,
                    trials=args.trials, control_seed=args.control_seed,
                    training="reused E1 checkpoints", controller="rkvae", eta=args.eta, gain_scale=args.gain_scale)
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--e1-root", type=Path, default=ROOT / "results/e1_validation_v2")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/e2_validation")
    p.add_argument("--sigma", type=float, nargs="+", default=[0, .0005, .001, .002])
    p.add_argument("--systems", nargs="+", choices=("damping", "franka"), default=("damping", "franka"))
    p.add_argument("--seeds", nargs="+", type=int, choices=(1, 2, 3), default=(2,))
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--control-seed", type=int, default=10001)
    p.add_argument("--eta", type=float, default=0.)
    p.add_argument("--gain-scale", type=float, default=1.)
    args = p.parse_args()
    for sigma in args.sigma:
        run(argparse.Namespace(**(vars(args) | {"sigma": sigma})))
