import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import os
import sys
sys.path.append("../utility/")
sys.path.append("../")
from scipy.integrate import odeint
import time
import tqdm

#define network
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

class ManifoldEmbLoss(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.k = k
        self.neighbor_indices = None 

    def compute_knn(self, X):
        n = X.shape[0]
        dist_matrix = torch.cdist(X, X, p=2)
        _, indices = torch.topk(dist_matrix, k=self.k+1, largest=False, dim=1)
        self.neighbor_indices = indices[:, 1:]
        return self.neighbor_indices

    def forward(self, z, X):

        self.compute_knn(X)
        n = z.shape[0]
        x_dim = X.shape[1]
        z_dim = z.shape[1]

        self.neighbor_indices = torch.clamp(self.neighbor_indices, 0, n-1)
        
        z_neighbors = z[self.neighbor_indices]  # [n, k, manifold_dim]
        x_neighbors = X[self.neighbor_indices]  # [n, k, x_dim]
        
        x_dist = torch.cdist(X.unsqueeze(1), x_neighbors, p=2).squeeze(1) 
        z_dist = torch.cdist(z.unsqueeze(1), z_neighbors, p=2).squeeze(1) 

        x_dist_max = torch.max(x_dist, dim=1, keepdim=True)[0]
        x_dist_max = torch.clamp(x_dist_max, min=1e-8)  
        x_dist = x_dist / x_dist_max  
        z_dist_max = torch.max(z_dist, dim=1, keepdim=True)[0]
        z_dist_max = torch.clamp(z_dist_max, min=1e-8)  
        z_dist = z_dist / z_dist_max  

        dist_diff = torch.abs(z_dist - x_dist)
        huber_loss = F.huber_loss(dist_diff, torch.zeros_like(dist_diff), delta=0.1, reduction='none')

        loss = torch.mean(huber_loss)

        return loss

class Network(nn.Module):
    def __init__(self, encode_layers, decoder_layers, Nkoopman, u_dim, x_dim, device=None, use_logvar=False, activation_name="relu"):
        """
        Args:
            encode_layers: 编码器特征提取层维度（如[64, 32]，输入→中间特征）
            decoder_layers: 解码器层维度（如[32, 64, x_dim]，latent+u→重建x）
            Nkoopman: Koopman latent空间维度（VAE的latent维度）
            u_dim: 控制量维度
            x_dim: 观测x的维度（用于解码器输出匹配输入）
            device: 计算设备
        """
        super(Network, self).__init__()
        if isinstance(device, int):
            device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.Nkoopman = Nkoopman 
        self.u_dim = u_dim 
        self.x_dim = x_dim 
        self.activation_name = activation_name
        self.activation = nn.Tanh() if activation_name == "tanh" else nn.ReLU()

        self.encode_feature = self._build_mlp(encode_layers, activation=self.activation)  

        self.fc_mu = nn.Linear(encode_layers[-1], encode_layers[-1])
        self.fc_logvar = nn.Linear(encode_layers[-1], encode_layers[-1])
        self.fc_logvar.weight.data.fill_(0.0)
        self.fc_logvar.bias.data.fill_(0.0)
        self.gx = nn.Linear(encode_layers[-1],u_dim)
        

        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)
        self.lB0 = nn.Linear(u_dim, Nkoopman, bias=False)
        self.lBbilinear = nn.Linear(u_dim * Nkoopman, Nkoopman, bias=False)
        self.lB = nn.Linear(u_dim, Nkoopman, bias=False)     
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1.0)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.99
        nn.init.normal_(self.lB.weight.data, mean=0.0, std=0.1)
        # Start the bilinear branch from the well-scaled linear input model:
        # B0 initially matches B, while the state-dependent correction is zero.
        # This avoids an ill-conditioned G(z)=B0+Btilde(z) at the first updates.
        with torch.no_grad():
            self.lB0.weight.copy_(self.lB.weight)
            self.lBbilinear.weight.zero_()
        self.use_logvar = use_logvar
        if not use_logvar:
            self.prior_logvar = nn.Parameter(torch.log(torch.tensor([0.01] * (Nkoopman-x_dim))), requires_grad=False)

        self.decode_net = self._build_mlp(decoder_layers, activation=self.activation)
        self.to(self.device)

    def _build_mlp(self, layers, activation=nn.ReLU()):
        mlp = OrderedDict()
        for i in range(len(layers) - 1):
            mlp[f"linear_{i}"] = nn.Linear(layers[i], layers[i+1])
            if i != len(layers) - 2:
                mlp[f"act_{i}"] = activation
        return nn.Sequential(mlp)

    def reparameterize(self, mu, logvar=None):
        if logvar is None:
            return mu
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std) 
        return mu + eps * std

    def encode_only(self, x):
        feat = self.activation(self.encode_feature(x))
        mu_z = self.fc_mu(feat)
        logvar_z = self.fc_logvar(feat).clamp(-20, 10)
        gx = 0.1 + 0.9 * torch.sigmoid(self.gx(feat))

        return mu_z, logvar_z, gx
    
    def control_encode(self,x):
        feat = self.activation(self.encode_feature(x))
        gx = 0.1 + 0.9 * torch.sigmoid(self.gx(feat))
        return gx

    def encode(self, x):
        mu_z, logvar_z, gx = self.encode_only(x)
        z = self.reparameterize(mu_z, logvar_z)
        mu_xz = torch.cat([x, mu_z], axis=-1)

        return mu_xz, z, mu_z, logvar_z, gx
        
    def predict(self, lifted, u, mode="lifted"):
        if mode == "bilinear":
            coupling = (u.unsqueeze(-1) * lifted.unsqueeze(-2)).flatten(-2)
            return self.lA(lifted) + self.lB0(u) + self.lBbilinear(coupling)
        if mode != "lifted":
            raise ValueError("mode must be lifted or bilinear")
        return self.lA(lifted) + self.lB(u * self.control_encode(lifted[..., :self.x_dim]))

    def prior(self, mu_xz, mu_z, u, logvar_z, gx=None, eps=1e-12, mode="lifted"):
        prediction = self.predict(mu_xz, u, mode)
        transition = self.lA.weight
        if mode == "bilinear":
            blocks = self.lBbilinear.weight.reshape(self.Nkoopman, self.u_dim, self.Nkoopman)
            transition = transition + torch.einsum("...m,imj->...ij", u, blocks)
        variance = torch.cat([torch.zeros_like(mu_xz[..., :self.x_dim]), logvar_z.exp()], -1)
        propagated = (transition.square() * variance.unsqueeze(-2)).sum(-1)
        return prediction, prediction[..., self.x_dim:], propagated[..., self.x_dim:].clamp_min(eps).log()

    def forward(self, mu_xz, mu_z, u_prev, logvar_z, gx=None, mode="lifted"):
        prediction, mean, logvar = self.prior(mu_xz, mu_z, u_prev, logvar_z, gx, mode=mode)
        return prediction, self.reparameterize(mean, logvar), mean, logvar

    def decode(self, z):
        x_recon = self.decode_net(z)

        return x_recon

    def compute_KL_loss(self, mu_z, logvar_z, mu_prior, logvar_prior):
        kl_loss = 0.5 * torch.sum(
            torch.exp(logvar_z - logvar_prior) + 
            (mu_z - mu_prior) ** 2 / torch.exp(logvar_prior) - 
            1 - (logvar_z - logvar_prior),
            dim=-1 
        ).mean()

        return kl_loss

def joint_losses(data, net, noise_std=0.0):
    """Sec. VI: time-summed reconstruction/Koopman MSE, time-averaged KL."""
    parameter = next(net.parameters())
    data = torch.as_tensor(data, dtype=parameter.dtype, device=parameter.device)
    if data.ndim != 3 or data.shape[0] < 2:
        raise ValueError("Expected [time>=2, batch, u_dim+x_dim]")
    states = data[..., net.u_dim:]
    states = states + noise_std * torch.randn_like(states)
    controls = data[..., :net.u_dim] / 8.0
    mu_xz, sample, mean, logvar, _ = net.encode(states)
    lifted = mu_xz
    rec = (net.decode(sample) - states).square().mean(dim=(-1, -2)).sum()
    kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    losses = {"rec": rec, "kl": kl}
    z_hat_bilinear_next = net.predict(lifted[:-1], data[:-1, :, :net.u_dim], "bilinear")
    z_hat_lifted_next = net.predict(lifted[:-1], data[:-1, :, :net.u_dim], "lifted")
    losses["bilinear"] = (z_hat_bilinear_next - lifted[1:]).square().mean(dim=(-1, -2)).sum()
    losses["lifted"] = (z_hat_lifted_next - lifted[1:]).square().mean(dim=(-1, -2)).sum()
    return losses


def Klinear_loss(data, net, mse_loss=None, emb_loss=None, u_dim=1, gamma=1,
                 Nstate=4, all_loss=1, lambda_geom=0):
    losses = joint_losses(data, net)
    return losses["rec"], losses["kl"], losses["bilinear"] + losses["lifted"], losses["rec"].new_zeros(())

def Stable_loss(net,Nstate):
    x_ref = np.zeros(Nstate) 
    device = next(net.parameters()).device
    mu_z, _, _ = net.encode_only(torch.as_tensor(x_ref, dtype=torch.float64, device=device))
    mu_xz = torch.cat([torch.zeros_like(torch.as_tensor(x_ref, dtype=torch.float64, device=device)), mu_z])
    # Penalize the deterministic lifted equilibrium, consistent with the paper.
    loss = torch.norm(mu_xz)
    return loss

def Eig_loss(net):
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs()-1.0#预留抑制噪声的量，不希望噪声保持传播
    mask = c>0
    loss = c[mask].sum()
    return loss


def Controlability_loss(net, eval_=False):
    A = net.lA.weight
    B = net.lB.weight
    n = A.size(0)  
    
    controllability_matrices = []
    current = B  
    controllability_matrices.append(current)
    
    for k in range(1, n):
        current = torch.matmul(A, current)  
        controllability_matrices.append(current)

    C = torch.cat(controllability_matrices, dim=1)

    _, S, _ = torch.linalg.svd(C, full_matrices=False)  
    min_singular = S[-1]  
    
    varepsilon = 1e-6 if eval_ else 1e-6
    loss = -min_singular + varepsilon  
    
    return loss.clamp(min=0.0) 

def train(env_name,epochs = 100,suffix="",all_loss=0,\
            encode_dim = 12,layer_depth=3,e_loss=1,gamma=0.5,Ktrain_samples=50000,\
        lambda_geom=0.0,\
        lambda_recon=1.0,\
        lambda_control=0.0,\
        lambda_KL=1.0,\
        device=0,
        use_logvar=True, noise_std=0.002, Ktest_samples=20000, eval_every_epochs=10,
        early_stopping_patience=20):
    # print(use_logvar)
    # Ktrain_samples = 1000
    # Ktest_samples = 1000
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    Ktrain_samples = Ktrain_samples
    Ktest_samples = Ktest_samples
    Ktrainsteps = 15
    Kteststeps = 30
    Kbatch_size = 512
    res = 1
    normal = 1
    #data prepare
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utility"))
    from Utility import data_collecter
    data_collect = data_collecter(env_name)
    u_dim = data_collect.udim
    # Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Kteststeps,mode="eval")
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Kteststeps,mode="train") if env_name != "CartPole-v1" and env_name !="MountainCarContinuous-v0" else data_collect.collect_koopman_data(Ktest_samples,Kteststeps,mode="eval")
    Ktest_samples = Ktest_data.shape[1]
    print("test data ok!,shape:",Ktest_data.shape)
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples,Ktrainsteps,mode="train")
    print("train data ok!,shape:",Ktrain_data.shape)
    Ktrain_samples = Ktrain_data.shape[1]
    in_dim = Ktest_data.shape[-1]-u_dim
    Nstate = in_dim
    # layer_depth = 4
    layer_width = 64
    encode_layers = [in_dim]+[layer_width]*(layer_depth-1)+[encode_dim]
    Nkoopman = encode_dim + in_dim
    decode_layers = [encode_dim] + [layer_width]*(layer_depth-1) + [in_dim]
    print("encode layers:",encode_layers)
    print("decode layers:",decode_layers)
    net = Network(encode_layers,decode_layers,Nkoopman,u_dim,in_dim,device,use_logvar)
    # print(net.named_modules())
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda() 
    net.double()
    mse_loss = nn.MSELoss()
    emb_loss = ManifoldEmbLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                    lr=learning_rate)
    # optimizer = torch.optim.SGD(net.parameters(),
    #                                 lr=learning_rate,momentum=0.9)
    for name, param in net.named_parameters():
        print("model:",name,param.requires_grad)
    #train
    best_loss = 1000.0
    best_control_loss = 1000.0
    convergence = 0
    best_iteration = 0
    best_state_dict = {}
    logdir = "../Data/"+suffix+"/KoVAE_"+env_name+"layer{}_edim{}_eloss{}_gamma{}_aloss{}_samples{}_recon{}_control{}_KL{}_geom{}_logvar{}".format(layer_depth,encode_dim,e_loss,gamma,all_loss,Ktrain_samples,lambda_recon,lambda_control,lambda_KL,lambda_geom,use_logvar)
    currentdir = "../Data/"+suffix+"/KoVAE_"+env_name + "_current"
    if not os.path.exists( "../Data/"+suffix):
        os.makedirs( "../Data/"+suffix)
    start_time = time.process_time()
    pbar = tqdm.trange(epochs, desc="epoch")
    batches_per_epoch = max(1, (Ktrain_samples + Kbatch_size - 1) // Kbatch_size)
    for epoch in pbar:
        # One epoch is a complete pass through every training trajectory.
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        epoch_loss = 0.0
        for batch_start in range(0, Ktrain_samples, Kbatch_size):
            X = Ktrain_data[:,Kindex[batch_start:batch_start + Kbatch_size],:]
            parts = joint_losses(X, net, noise_std=noise_std)
            Reconloss, KLloss = parts["rec"], parts["kl"]
            Predloss = parts["bilinear"] + parts["lifted"]
            Geomloss = Predloss.new_zeros(())
            control_loss = Eig_loss(net) + Controlability_loss(net)
            loss = Predloss + lambda_recon * Reconloss + lambda_control * (control_loss + Stable_loss(net,in_dim)) + lambda_KL * KLloss + lambda_geom * Geomloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{epoch_loss / batches_per_epoch:.6f}", "batches": batches_per_epoch})

        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (epoch + 1) % eval_every_epochs == 0 or epoch == epochs - 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95
            convergence += 1
            with torch.no_grad():
                Reconloss, KLloss, Predloss, Geomloss = Klinear_loss(Ktest_data,net,mse_loss,emb_loss,u_dim,gamma,Nstate,all_loss=0)
                Eigloss = Eig_loss(net)
                control_loss = Controlability_loss(net, eval_=True) + Stable_loss(net,in_dim)
                Predloss = Predloss.detach().cpu().numpy()
                Reconloss = Reconloss.detach().cpu().numpy()
                KLloss = KLloss.detach().cpu().numpy()
                control_loss = control_loss.detach().cpu().numpy()
                Saved_dict = {'model':net.state_dict(),'encode_layer':encode_layers,'decode_layer':decode_layers, 'schema_version': 2, 'u_dim': u_dim, 'noise_std': noise_std}
                torch.save(Saved_dict,currentdir+".pth")
                if Predloss<best_loss:
                    print("Best model updated at epoch ", epoch + 1)
                    convergence = 0
                    best_loss = copy(Predloss)
                    best_control_loss = copy(control_loss)
                    best_iteration = epoch + 1
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {'model':best_state_dict,'encode_layer':encode_layers,'decode_layer':decode_layers, 'schema_version': 2, 'u_dim': u_dim, 'noise_std': noise_std}
                    torch.save(Saved_dict,logdir+".pth")
                print("Method:KoVAE_with_KlinearEig Epoch:{} Predloss{} Reconloss:{} KLloss{} Controlloss:{} Eigloss:{} ".format(epoch + 1,Predloss,Reconloss,KLloss,control_loss,Eigloss))
            if convergence >= early_stopping_patience:
                print("Early stopping at epoch ", epoch + 1)
                break

    print("END-best_loss{}-best_iteration{}".format(best_loss, best_iteration))
    

def main():
    train(args.env,suffix=args.suffix,all_loss=args.all_loss,\
        epochs=args.epochs,\
        eval_every_epochs=args.eval_every_epochs,\
        early_stopping_patience=args.early_stopping_patience,\
        encode_dim=args.encode_dim,layer_depth=args.layer_depth,\
        e_loss=args.e_loss,gamma=args.gamma,\
        Ktrain_samples=args.K_train_samples,\
        Ktest_samples=args.K_test_samples,\
        lambda_geom=args.lambda_geom,\
        lambda_recon=args.lambda_recon,\
        lambda_control=args.lambda_control,\
        lambda_KL=args.lambda_KL,
        device=args.device,
        use_logvar=args.use_logvar, noise_std=args.noise_std)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="DampingPendulum")
    parser.add_argument("--suffix",type=str,default="5_2")
    parser.add_argument("--all_loss",type=int,default=1)
    parser.add_argument("--K_train_samples",type=int,default=50000)
    parser.add_argument("--K_test_samples",type=int,default=20000)
    parser.add_argument("--e_loss",type=int,default=1)
    parser.add_argument("--gamma",type=float,default=0.9)
    parser.add_argument("--encode_dim",type=int,default=20)
    parser.add_argument("--layer_depth",type=int,default=3)
    parser.add_argument("--lambda_geom", type=float, default=0.0, help="流形几何约束权重")
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="重建约束权重")
    parser.add_argument("--lambda_control", type=float, default=0.0, help="控制约束权重")
    parser.add_argument("--lambda_KL", type=float, default=1.0, help="散度约束权重")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_every_epochs", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--noise_std", type=float, default=0.002)
    parser.add_argument("--use_logvar", action='store_true')
    args = parser.parse_args()
    main()
