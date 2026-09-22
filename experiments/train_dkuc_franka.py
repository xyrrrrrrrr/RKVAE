"""Train a normalized DKUC Koopman model for the Franka benchmark."""
import json
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "utility"), str(ROOT / "train")]
from rkvae import System, collect
from Learn_Koopman_with_KlinearEig import Network

torch.set_default_dtype(torch.float64)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', default=os.environ.get('DKUC_OUTPUT', ROOT / 'results/e2_franka_norm_sigma002_200ep_gpu/dkuc_train'))
p.add_argument('--sigma', type=float, default=float(os.environ.get('DKUC_SIGMA', '0.002')))
p.add_argument('--data-cache', default=os.environ.get('DKUC_DATA_CACHE'))
p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
p.add_argument('--seed', type=int, default=2)
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--samples', type=int, default=50000)
p.add_argument('--validation-samples', type=int, default=20000)
p.add_argument('--sequence', type=int, default=15)
p.add_argument('--batch-size', type=int, default=1024)
p.add_argument('--franka-init-position-width', type=float, default=0.15)
p.add_argument('--franka-init-velocity-width', type=float, default=0.0)
args = p.parse_args()
if args.sigma < 0:
    raise ValueError('--sigma must be nonnegative')
device = torch.device(args.device)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
cache = Path(args.data_cache) if args.data_cache else out.parent / 'dataset'
torch.manual_seed(args.seed)
seq, samples, validation_samples, latent = args.sequence, args.samples, args.validation_samples, 20
sigma = args.sigma
system = System("franka")
try:
    cache.mkdir(parents=True, exist_ok=True)
    train_file, val_file = cache / "train.npy", cache / "validation.npy"
    if train_file.exists() and val_file.exists():
        train_np = np.load(train_file, mmap_mode="r")
        val_np = np.load(val_file, mmap_mode="r")
    else:
        train_np = collect(system, samples, seq, sigma, args.seed, normalize_states=True,
                           franka_init_position_width=args.franka_init_position_width,
                           franka_init_velocity_width=args.franka_init_velocity_width)
        val_np = collect(system, validation_samples, seq, sigma, args.seed + 100000,
                         normalize_states=True,
                         franka_init_position_width=args.franka_init_position_width,
                         franka_init_velocity_width=args.franka_init_velocity_width)
        np.save(train_file, train_np)
        np.save(val_file, val_np)
    input_scale = system.limit.copy()
finally:
    system.close()
train = torch.as_tensor(train_np, device=device)
val = torch.as_tensor(val_np, device=device)
n, m = 14, 7
net = Network([n, 128, 128, 128, latent], n + latent, m, device=str(device)).to(device).double()

def prediction_loss(data):
    z = net.encode(data[0, :, m:])
    total = data.new_zeros(()); recon = data.new_zeros(())
    beta, denom = 1.0, 0.0
    for k in range(seq):
        u = data[k, :, :m] / torch.as_tensor(input_scale, device=device)
        z_next = net(z, u)
        target = net.encode(data[k + 1, :, m:])
        total = total + beta * (z_next - target).square().mean()
        recon = recon + beta * (net.encode(z_next[:, :n]) - z_next).square().mean()
        denom += beta; beta *= 0.9; z = z_next
    return total / denom + 0.5 * recon / denom

def regularizers():
    A, B = net.lA.weight, net.lB.weight
    spectral = torch.relu(torch.linalg.eigvals(A).abs() - 0.995).square().mean()
    gain_floor = torch.relu(torch.as_tensor(0.01, device=device) - torch.linalg.matrix_norm(B, ord=2)).square()
    gram = torch.zeros((A.shape[0], A.shape[0]), device=device)
    power = torch.eye(A.shape[0], device=device)
    for _ in range(A.shape[0]):
        ab = power @ B; gram = gram + ab @ ab.T; power = A @ power
    gram = gram / torch.trace(gram).detach().clamp_min(1e-12)
    controllability = -torch.logdet(gram + 1e-5 * torch.eye(A.shape[0], device=device))
    return spectral + 0.1 * gain_floor + 1e-4 * controllability

opt = torch.optim.Adam(net.parameters(), lr=1e-3)
best, best_epoch, history = float("inf"), 0, []
for epoch in range(1, args.epochs + 1):
    order = torch.randperm(train.shape[1], device=device); total = 0.0
    for start in range(0, samples, args.batch_size):
        batch = train[:, order[start:start + args.batch_size]]
        loss = prediction_loss(batch) + regularizers()
        opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss.detach()) * batch.shape[1]
    with torch.no_grad():
        val_loss = float((prediction_loss(val) + regularizers()).detach())
        bnorm = float(torch.linalg.matrix_norm(net.lB.weight).detach())
        rho = float(torch.linalg.eigvals(net.lA.weight).abs().max().detach())
    rec = {"epoch": epoch, "train_loss": total / samples, "validation_loss": val_loss,
           "B_spectral_norm": bnorm, "A_spectral_radius": rho}; history.append(rec)
    if val_loss < best:
        best, best_epoch = val_loss, epoch
        torch.save({"model": net.state_dict(), "epoch": epoch,
                    "config": {"system": "franka", "n": n, "m": m, "latent": latent,
                               "sigma": sigma, "normalize_states": True,
                               "normalize_inputs": True, "input_scale": input_scale.tolist(),
                               "seq": seq}}, out / "best.pt")
    if epoch % 20 == 0:
        for group in opt.param_groups: group["lr"] *= 0.9
        print(rec, flush=True)
torch.save({"model": net.state_dict(), "epoch": args.epochs,
            "config": {"system": "franka", "n": n, "m": m, "latent": latent,
                       "sigma": sigma, "normalize_states": True, "normalize_inputs": True,
                       "input_scale": input_scale.tolist(), "seq": seq}}, out / "last.pt")
(out / "history.jsonl").write_text("\n".join(json.dumps(x) for x in history) + "\n")
(out / "config.json").write_text(json.dumps({"epochs": args.epochs, "samples": samples,
    "validation_samples": validation_samples, "sequence": seq, "sigma": sigma,
    "output": str(out), "data_cache": str(cache), "seed": args.seed,
    "device": str(device), "batch_size": args.batch_size,
    "franka_init_position_width": args.franka_init_position_width,
    "franka_init_velocity_width": args.franka_init_velocity_width,
    "normalize_states": True, "normalize_inputs": True, "best_epoch": best_epoch,
    "best_validation_loss": best}, indent=2))
