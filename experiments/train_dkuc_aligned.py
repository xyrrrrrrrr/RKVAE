"""Train normalized DKUC with input-channel regularization."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from Learn_Koopman_with_KlinearEig import Network
from Utility import SinglePendulum

OUT = Path("results/e2_pendulum_norm_sampled_z_sigma002_theta2_eta001_200ep_gpu/dkuc_aligned_train")
OUT.mkdir(parents=True, exist_ok=True)
torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2)
scale = np.array([np.pi, 8.0]); sigma = 0.002; seq, n, h = 15, 2, 20

def collect(count, seed):
    rng = np.random.default_rng(seed); env = SinglePendulum()
    data = np.empty((seq + 1, count, 3))
    for j in range(count):
        x = rng.uniform(-np.array([3.0, 2.0]), np.array([3.0, 2.0])); env.reset_state(x)
        for k in range(seq + 1):
            u = rng.uniform(-8.0, 8.0); data[k, j] = [u / 8.0, *(x / scale)]
            if k < seq:
                x = env.step(np.array([u]))[0] + sigma * rng.standard_normal(2) * scale; env.reset_state(x)
    return torch.as_tensor(data, device=device)

def prediction_loss(data, net):
    z = net.encode(data[0, :, 1:]); total = data.new_zeros(()); recon = data.new_zeros(())
    beta, denom = 1.0, 0.0
    for k in range(seq):
        z_next = net(z, data[k, :, :1]); target = net.encode(data[k + 1, :, 1:])
        total += beta * (z_next - target).square().mean()
        recon += beta * (net.encode(z_next[:, :n]) - z_next).square().mean()
        denom += beta; beta *= 0.9; z = z_next
    return total / denom + 0.5 * recon / denom

def regularizers(net):
    A, B = net.lA.weight, net.lB.weight
    bnorm = torch.linalg.matrix_norm(B, ord=2)
    gain_floor = torch.relu(torch.as_tensor(0.01, device=device) - bnorm).square()
    gram = torch.zeros((A.shape[0], A.shape[0]), device=device); power = torch.eye(A.shape[0], device=device)
    for _ in range(A.shape[0]):
        ab = power @ B; gram += ab @ ab.T; power = A @ power
    gram = gram / torch.trace(gram).detach().clamp_min(1e-12)
    gramian = -torch.logdet(gram + 1e-5 * torch.eye(A.shape[0], device=device))
    jacobian = torch.relu(torch.as_tensor(0.005, device=device) - bnorm).square()
    return 0.1 * gain_floor + 1e-4 * gramian + 0.1 * jacobian

train, val = collect(50000, 2), collect(20000, 100002)
net = Network([2, 128, 128, 128, h], n + h, 1, device=str(device)).to(device).double()
opt = torch.optim.Adam(net.parameters(), lr=1e-3); best = float("inf"); history = []
for epoch in range(1, 201):
    order = torch.randperm(train.shape[1], device=device); total = 0.0
    for start in range(0, train.shape[1], 1024):
        batch = train[:, order[start:start + 1024]]; loss = prediction_loss(batch, net) + regularizers(net)
        opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.detach()) * batch.shape[1]
    with torch.no_grad(): val_loss = float((prediction_loss(val, net) + regularizers(net)).detach())
    rec = {"epoch": epoch, "train_loss": total / train.shape[1], "validation_loss": val_loss,
           "b_norm": float(torch.linalg.matrix_norm(net.lB.weight).detach())}; history.append(rec)
    if val_loss < best:
        best = val_loss; torch.save({"model": net.state_dict(), "layer": [2, 128, 128, 128, h], "epoch": epoch,
          "config": {"sigma": sigma, "normalize_states": True, "normalize_inputs": True}}, OUT / "best.pt")
    if epoch % 20 == 0:
        for group in opt.param_groups: group["lr"] *= 0.9
    if epoch == 1 or epoch % 20 == 0: print(rec, flush=True)
(OUT / "config.json").write_text(json.dumps({"epochs": 200, "samples": 50000, "validation_samples": 20000,
  "sequence": 15, "sigma": sigma, "normalize_states": True, "normalize_inputs": True, "best_validation_loss": best}, indent=2))
(OUT / "history.jsonl").write_text("\n".join(json.dumps(x) for x in history) + "\n")
