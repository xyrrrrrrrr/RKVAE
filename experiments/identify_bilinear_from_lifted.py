"""Identify the bilinear input matrices after a lifted-only training run.

The representation and A are frozen. For every initial state, the physical
system is stepped once with zero input and once with each unit input e_i.
The latter is the input-limit-scaled version of e_i in the simulator, so its
normalized value is exactly e_i. For each input channel we solve

    z_next(e_i) - A z = B0[:, i] + Bi z.

This is the per-channel least-squares form of Eq. (lifted dynamics) in
main.tex.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "utility"), str(ROOT / "train")]
from rkvae import System, make_net  # noqa: E402


def lifted_state(net, system, x):
    with torch.no_grad():
        xt = torch.as_tensor(system.norm(x), dtype=torch.float64)
        mu, _, _ = net.encode_only(xt)
        return np.r_[xt.numpy(), mu.numpy()]


def collect_unit_pairs(net, system, samples, seed):
    rng = np.random.default_rng(seed)
    nkoopman = net.Nkoopman
    z0 = np.empty((samples, nkoopman))
    znext0 = np.empty_like(z0)
    znext_e = np.empty((system.m, samples, nkoopman))
    for j in range(samples):
        x0 = system.reset(rng, broad=True).copy()
        # Zero-input and e_i transitions start from the identical state.
        system.set(x0)
        x_zero, _ = system.step(np.zeros(system.m), 0.0, rng)
        z0[j] = lifted_state(net, system, x0)
        znext0[j] = lifted_state(net, system, x_zero)
        for i in range(system.m):
            system.set(x0)
            unit = np.zeros(system.m)
            unit[i] = system.limit[i]
            x_unit, _ = system.step(unit, 0.0, rng)
            znext_e[i, j] = lifted_state(net, system, x_unit)
    return z0, znext0, znext_e


def identify(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(saved["config"])
    net = make_net(cfg, "cpu")
    net.load_state_dict(saved["model"])
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)

    system = System(cfg["system"])
    try:
        z, znext0, znext_e = collect_unit_pairs(net, system, args.samples, args.seed)
    finally:
        system.close()

    A = net.lA.weight.detach().numpy().copy()
    # The zero-input pairs provide a direct check of the frozen A model.
    zero_residual = znext0 - z @ A.T
    b0 = np.empty((net.Nkoopman, system.m))
    btilde = np.empty((net.Nkoopman, system.m, net.Nkoopman))
    fit_mse = []
    condition = []
    ridge = float(args.ridge)
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    # Penalize the constant and state-dependent input matrices separately.
    # With --zero-b0, the paper model is tested under the hard constraint B0=0.
    penalty = np.eye(1 + net.Nkoopman)
    penalty[0, 0] *= args.ridge_b0_scale
    for i in range(system.m):
        y = znext_e[i] - z @ A.T
        state_features = z
        if args.zero_b0:
            design = state_features
            normal = design.T @ design + ridge * np.eye(net.Nkoopman)
            state_coef = np.linalg.solve(normal, design.T @ y)
            coef = np.vstack([np.zeros((1, net.Nkoopman)), state_coef])
            fit_prediction = design @ state_coef
        else:
            design = np.column_stack([np.ones(len(z)), state_features])
            normal = design.T @ design + ridge * penalty
            coef = np.linalg.solve(normal, design.T @ y)
            fit_prediction = design @ coef
        singular = np.linalg.svd(design, compute_uv=False)
        b0[:, i] = coef[0]
        # lstsq returns feature-by-output coefficients; Network expects
        # output-by-feature weights for its Linear layer.
        btilde[:, i, :] = coef[1:].T
        fit_mse.append(float(np.mean((fit_prediction - y) ** 2)))
        condition.append(float(singular[0] / max(singular[-1], 1e-15)))

    with torch.no_grad():
        net.lB0.weight.copy_(torch.as_tensor(b0, dtype=torch.float64))
        # Network flattens (u_i * z_j) in i-major order.
        net.lBbilinear.weight.copy_(torch.as_tensor(btilde.reshape(net.Nkoopman, -1), dtype=torch.float64))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identified_cfg = dict(cfg)
    identified_cfg.update({"variant": "joint", "identification": "unit_input_least_squares"})
    report = {
        "protocol": "zero plus e_i paired transitions",
        "samples": args.samples,
        "ridge": ridge,
        "ridge_b0_scale": args.ridge_b0_scale,
        "zero_b0": args.zero_b0,
        "input_vectors_normalized": "e_i^(m)",
        "zero_input_mse_against_A": float(np.mean(zero_residual ** 2)),
        "per_channel_fit_mse": fit_mse,
        "per_channel_design_condition": condition,
        "A_frobenius_norm": float(np.linalg.norm(A)),
        "B0_frobenius_norm": float(np.linalg.norm(b0)),
        "Btilde_frobenius_norm": float(np.linalg.norm(btilde)),
        "B0": b0.tolist(),
    }
    payload = {
        "model": net.state_dict(),
        "config": identified_cfg,
        "epoch": saved.get("epoch", 0),
        "score": saved.get("score", None),
        "identification": report,
    }
    torch.save(payload, out / "identified.pt")
    (out / "identification.json").write_text(json.dumps(report, indent=2))
    np.savez(out / "identification_data.npz", z=z, znext_zero=znext0,
             znext_unit=znext_e, A=A, B0=b0, Btilde=btilde)
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--ridge", type=float, default=1e-4,
                   help="Frobenius/Tikhonov coefficient penalty for B0 and Btilde")
    p.add_argument("--ridge-b0-scale", type=float, default=1.0,
                   help="Relative penalty multiplier for B0")
    p.add_argument("--zero-b0", action="store_true",
                   help="Hard constrain B0 to zero and fit only Btilde")
    return p


if __name__ == "__main__":
    identify(parser().parse_args())
