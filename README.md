# RKVAE Robust Control Experiments

This repository contains the Koopman/VAE models, controllers, simulation
notebooks, and the reproducible experiments used by `main.tex`.

## Repository layout

- `train/`: Koopman and K-VAE model implementations.
- `control/`: controller notebooks and trajectory utilities.
- `franka/`: Franka environment, model helpers, and reference notebooks.
- `gym_env/`, `utility/`: environments and shared numerical utilities.
- `experiments/`: canonical command-line training, evaluation, and plotting entry points.
- `docs/`: experiment protocols and implementation notes.
- `results/`: only the three result trees used by the paper text.

## Canonical paper results

- `results/e2_pendulum_norm_sampled_z_sigma002_theta2_eta001_200ep_gpu/`
- `results/e2_franka_e2_wide_norm_sampled_z_sigma001_200ep_cpu/`
- `results/e2_pendulum_ablation_sigma005/`

The results use the `DL` Conda environment. Run commands from the repository
root, for example:

```bash
PYTHON=/home/xyrrrrrrrr/.conda/envs/DL/bin/python
$PYTHON experiments/rkvae.py --help
$PYTHON experiments/evaluate_dkuc_franka.py --help
$PYTHON experiments/plot_franka_sigma005.py
$PYTHON experiments/run_eta_ablation.py --checkpoint results/e2_pendulum_norm_sampled_z_sigma002_theta2_eta001_200ep_gpu/train/best_lifted.pt --output-root results/e2_pendulum_ablation_sigma005/eta_ablation_lifted
```

The Franka trajectory figure is written to the Franka result tree under
`evaluation_gain10_eta001_10trials/franka_sigma005_trajectories/`.

## Reproducibility notes

Training and evaluation configurations are stored beside each checkpoint in
`config.json` and beside each rollout in `metrics.json`. The ablation uses
(σ=0.005), 100 trials, and the strict success threshold
\(\tau_m=0.01\) defined in `main.tex`.

The older notebooks remain available as historical reference material. New
experiments should use the scripts in `experiments/` so that paths, seeds,
noise injection, and metrics are explicit.
