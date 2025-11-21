from ntpath import join
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
from torch.utils.tensorboard import SummaryWriter
from scipy.integrate import odeint
# physics engine
import pybullet as pb
import pybullet_data
from scipy.io import loadmat, savemat
# Franka simulator
from franka_env import FrankaEnv

#data collect
def Obs(o):
    return np.concatenate((o[:3],o[7:]),axis=0)

class data_collecter():
    def __init__(self,env_name) -> None:
        self.env_name = env_name
        self.env =  FrankaEnv(render = False)
        self.Nstates = 17
        self.uval = 0.12
        self.udim = 7
        self.reset_joint_state = np.array(self.env.reset_joint_state)

    def collect_koopman_data(self,traj_num,steps):
        train_data = np.empty((steps+1,traj_num,self.Nstates+self.udim))
        for traj_i in range(traj_num):
            noise = (np.random.rand(7)-0.5)*2*0.2
            joint_init = self.reset_joint_state+noise
            joint_init = np.clip(joint_init,self.env.joint_low,self.env.joint_high)
            s0 = self.env.reset_state(joint_init)
            s0 = Obs(s0)
            u10 = (np.random.rand(7)-0.5)*2*self.uval
            data_concat = np.concatenate([u10.reshape(-1), s0.reshape(-1)], axis=0).reshape(-1)
            data_concat[self.udim:] += np.random.normal(0, 0.1, self.Nstates)
            train_data[0,traj_i,:] = data_concat
            for i in range(1,steps+1):
                s0 = self.env.step(u10)
                s0 = Obs(s0)
                u10 = (np.random.rand(7)-0.5)*2*self.uval
                data_concat = np.concatenate([u10.reshape(-1), s0.reshape(-1)], axis=0).reshape(-1)
                data_concat[self.udim:] += np.random.normal(0, 0.1, self.Nstates)
                train_data[i,traj_i,:] = data_concat
        return train_data
        
#define network
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega
    
class Network(nn.Module):
    def __init__(self, encode_layers, decoder_layers, Nkoopman, u_dim, x_dim, device=None):
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
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Nkoopman = Nkoopman  # VAE latent维度
        self.u_dim = u_dim        # 控制量维度
        self.x_dim = x_dim        # 观测维度

        # -------------------------- 1. 变分编码器（Inference Network）--------------------------
        # 功能：输入观测x+控制量u，输出latent变量z的近似后验 q(z|x,u) 的均值/对数方差
        # （原网络编码逻辑扩展：先融合x+u特征，再输出均值/方差）
        self.encode_feature = self._build_mlp(encode_layers, activation=nn.ReLU())  # 特征提取
        # 输出latent均值（mu_z）和对数方差（logvar_z，避免方差为负）
        self.fc_mu = nn.Linear(encode_layers[-1], Nkoopman)
        self.fc_logvar = nn.Linear(encode_layers[-1], Nkoopman)

        # -------------------------- 2. Koopman latent先验（Prior Network）--------------------------
        # 功能：带控制量的线性先验 p(z_t | z_{t-1}, u_{t-1})，沿用原网络Koopman动力学
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)  # 状态转移矩阵 A
        self.lB = nn.Linear(u_dim, Nkoopman, bias=False)     # 控制输入矩阵 B
        # 初始化lA（沿用原网络正交化+幅值约束，确保初始稳定性）
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1.0)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9  # 幅值0.9，避免初始发散
        # 初始化lB（高斯初始化）
        nn.init.normal_(self.lB.weight.data, mean=0.0, std=0.1)
        # 先验方差（固定小值，或可学习；此处固定为0.01，平衡稳定性与随机性）
        # self.prior_logvar = nn.Parameter(torch.tensor([math.log(0.01)] * Nkoopman), requires_grad=False)
        # -------------------------- 3. 解码器（Generative Network）--------------------------
        # 功能：输入latent z+控制量u，重建观测x，即 p(x | z, u)
        self.decode_net = self._build_mlp(decoder_layers, activation=nn.ReLU())

        # 设备迁移
        self.to(self.device)

    def _build_mlp(self, layers, activation=nn.ReLU()):
        """辅助函数：构建MLP（用于编码器特征提取、解码器）"""
        mlp = OrderedDict()
        for i in range(len(layers) - 1):
            mlp[f"linear_{i}"] = nn.Linear(layers[i], layers[i+1])
            # 最后一层不添加激活（编码器输出特征无激活，解码器输出观测无激活）
            if i != len(layers) - 2:
                mlp[f"act_{i}"] = activation
        return nn.Sequential(mlp)

    def reparameterize(self, mu, logvar=None):
        """VAE核心：重参数化技巧，使latent采样可微分"""
        std = torch.exp(0.5 * logvar)  # 标准差 = sqrt(exp(logvar))
        eps = torch.randn_like(std, device=self.device) # 标准高斯噪声 N(0,1)
        return mu + eps * std  # z = mu + eps*std ~ q(z|x,u)

    def encode_only(self, x):
        """编码器：输入x（观测），输出latent z的近似后验参数"""
        # 步骤1：提取高维特征
        feat = self.encode_feature(x)
        # 步骤2：输出近似后验的均值和对数方差
        mu_z = self.fc_mu(feat)
        logvar_z = self.fc_logvar(feat)
        return mu_z, logvar_z
    
    def encode(self, x):
        # 获取z
        mu_z, logvar_z = self.encode_only(x)
        z = self.reparameterize(mu_z, logvar_z)
        # # 拼接
        # return torch.cat([x, z], axis=-1)
        return z

    def prior(self, mu_z, u_prev, logvar_z, eps=1e-6):
        """先验：基于前一时刻latent z_prev和控制量u_prev，计算当前z的先验 p(z|z_prev, u_prev)"""
        # Koopman线性动力学：z_t = A z_{t-1} + B u_{t-1}（先验均值）
        mu_prior = self.lA(mu_z) + self.lB(u_prev)
        # 先验方差
        Ad = self.lA.weight
        logvar_prior = torch.log(torch.sum(Ad ** 2 * torch.exp(logvar_z).unsqueeze(1), dim=2))
        return mu_prior, logvar_prior
    
    def forward(self, mu_z, u_prev, logvar_z):
        mu_prior, logvar_prior = self.prior(mu_z, u_prev, logvar_z)
        z_next = self.reparameterize(mu_prior, logvar_prior)
        return z_next, mu_prior, logvar_prior


    def decode(self, z):
        """解码器：输入latent z+控制量u，重建观测x"""
        x_recon = self.decode_net(z)    # 重建观测，维度：(batch, x_dim)
        return x_recon

    def compute_KL_loss(self, mu_z, logvar_z, mu_prior, logvar_prior):
        """计算KL散度（带控制的变分下界）"""
        # KL散度：近似后验 q(z|x,u) 与先验 p(z|z_prev,u) 的差距（高斯KL解析公式）
        # KL(q||p) = 0.5 * sum( exp(logvar_z - logvar_prior) + (mu_z - mu_prior)^2/exp(logvar_prior) - 1 - (logvar_z - logvar_prior) )
        kl_loss = 0.5 * torch.sum(
            torch.exp(logvar_z - logvar_prior) + 
            (mu_z - mu_prior) ** 2 / torch.exp(logvar_prior) - 
            1 - (logvar_z - logvar_prior),
            dim=-1  # 对latent维度求和
        ).mean()  # 对batch维度求平均

        return kl_loss

def K_loss(data,net,u_dim=1,Nstate=4):
    steps,train_traj_num,Nstates = data.shape
    device = net.device
    data = torch.DoubleTensor(data).to(device)
    mu_z, logvar_z = net.encode_only(data[0,:,u_dim:])
    z_current = net.reparameterize(mu_z, logvar_z)
    max_loss_list = []
    mean_loss_list = []
    for i in range(steps-1):
        z_next, mu_prior, logvar_prior = net.forward(mu_z,data[i,:,:u_dim],logvar_z)
        x_recon = net.decode(z_next)
        y = data[i+1,:,u_dim:]
        Err = x_recon-y
        mu_z = mu_prior
        logvar_z = logvar_prior
        max_loss_list.append(torch.mean(torch.max(torch.abs(Err),axis=0).values).detach().cpu().numpy())
        mean_loss_list.append(torch.mean(torch.mean(torch.abs(Err),axis=0)).detach().cpu().numpy())
    return np.array(max_loss_list),np.array(mean_loss_list)


#loss function
def Klinear_loss(data,net,mse_loss,u_dim=1,gamma=0.99,Nstate=4,all_loss=0):
    steps,train_traj_num,NKoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    mu_z, logvar_z = net.encode_only(data[0,:,u_dim:])
    z_current = net.reparameterize(mu_z, logvar_z)
    beta = 1.0
    beta_sum = 0.0
    Reconloss = torch.tensor(0.0, dtype=torch.float64, device=device)
    KLloss = torch.tensor(0.0, dtype=torch.float64, device=device)
    Predloss = torch.tensor(0.0, dtype=torch.float64, device=device)
    for i in range(steps-1):
        z_next, mu_prior, logvar_prior = net.forward(mu_z,data[i,:,:u_dim],logvar_z)
        z_next_real = net.encode(data[i+1,:,u_dim:])
        x_recon = net.decode(z_current)
        beta_sum += beta
        # Reconstruction loss
        Reconloss += beta*mse_loss(x_recon,data[i,:,u_dim:])
        # KL divergence loss
        KLloss += beta*net.compute_KL_loss(mu_z, logvar_z, mu_prior, logvar_prior)
        # Prediction loss
        if not all_loss:
            x_next_recon = net.decode(z_next)
            Predloss += beta*mse_loss(x_next_recon,data[i+1,:,u_dim:])
        else:
            Predloss += beta*mse_loss(z_next,z_next_real)
        mu_z = mu_prior
        logvar_z = mu_prior
        z_current = z_next_real
        beta *= gamma
    Reconloss = Reconloss/beta_sum
    KLloss = KLloss/beta_sum
    Predloss = Predloss/beta_sum
    return Reconloss, KLloss, Predloss

def Stable_loss(net,Nstate):
    x_ref = np.zeros(Nstate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_ref_lift = net.encode(torch.DoubleTensor(x_ref).to(device))
    loss = torch.norm(x_ref_lift)
    return loss

def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs()-torch.ones(1,dtype=torch.float64).to(device)
    mask = c>0
    loss = c[mask].sum()
    return loss

# 能控性损失
def Controlability_loss(net):
    A = net.lA.weight
    B = net.lB.weight
    n = A.size(0)  # 获取状态维度n
    
    # 构建能控性矩阵 C = [B, AB, A²B, ..., A^{n-1}B]
    controllability_matrices = []
    current = B  # 初始项：A^0B = B
    controllability_matrices.append(current)
    
    # 迭代计算 A^k B (k从1到n-1)
    for k in range(1, n):
        current = torch.matmul(A, current)  # A^k B = A·(A^{k-1}B)
        controllability_matrices.append(current)

    # 按列拼接得到能控性矩阵 C ∈ R^{n×(n·m)}
    C = torch.cat(controllability_matrices, dim=1)

    # 计算能控性矩阵的奇异值，最小奇异值反映秩稳健性
    _, S, _ = torch.linalg.svd(C, full_matrices=False)  # S为奇异值向量
    min_singular = S[-1]  # 最小奇异值
    
    varepsilon = 1e-6  # 避免数值不稳定的小常数
    loss = -min_singular + varepsilon  # 当最小奇异值 ≥ epsilon时，损失趋近于0
    
    return loss.clamp(min=0.0)  # 确保损失非负（奇异值过小时才产生惩罚）

def train(env_name,train_steps = 300000,suffix="",all_loss=0,\
            encode_dim = 20,layer_depth=3,e_loss=1,gamma=0.5, lambda_recon=0.4,\
        lambda_control=0.1,\
        lambda_KL=0.5):
    np.random.seed(98)
    # Ktrain_samples = 100
    # Ktest_samples = 100
    Ktrain_samples = 50000
    Ktest_samples = 20000
    Ktrainsteps = 10
    Kteststeps = 20
    Kbatch_size = 100
    res = 1
    normal = 1
    gamma = 0.8
    #data prepare
    data_collect = data_collecter(env_name)
    u_dim = data_collect.udim
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Kteststeps)
    Ktest_samples = Ktest_data.shape[1]
    print("test data ok!,shape:",Ktest_data.shape)
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples,Ktrainsteps)
    print("train data ok!,shape:",Ktrain_data.shape)
    Ktrain_samples = Ktrain_data.shape[1]
    in_dim = Ktest_data.shape[-1]-u_dim
    Nstate = in_dim
    # layer_depth = 4
    layer_width = 128
    encode_layers = [in_dim]+[layer_width]*layer_depth+[encode_dim]
    Nkoopman = encode_dim
    decode_layers = [encode_dim] + [layer_width]*layer_depth + [in_dim]
    print("encode layers:",encode_layers)
    print("decode layers:",decode_layers)
    net = Network(encode_layers,decode_layers,Nkoopman,u_dim,in_dim)
    # print(net.named_modules())
    eval_step = 1000
    learning_rate = 1e-3
    if torch.cuda.is_available():
        net.cuda() 
    net.double()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                    lr=learning_rate)
    for name, param in net.named_parameters():
        print("model:",name,param.requires_grad)
    #train
    eval_step = 1000
    best_loss = 1000.0
    best_state_dict = {}
    subsuffix = suffix+"KK_KoVAE"+env_name+"layer{}_edim{}_eloss{}_gamma{}_aloss{}".format(layer_depth,encode_dim,e_loss,gamma,all_loss)
    logdir = "Data/"+suffix+"/"+subsuffix
    if not os.path.exists( "Data/"+suffix):
        os.makedirs( "Data/"+suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    import tqdm
    pbar = tqdm.trange(train_steps)
    for i in pbar:
        #K loss
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:,Kindex[:Kbatch_size],:]
        Reconloss, KLloss, Predloss = Klinear_loss(X,net,mse_loss,u_dim,gamma,Nstate,all_loss)
        control_loss = Eig_loss(net) + Controlability_loss(net)
        loss = Predloss + lambda_recon * Reconloss + lambda_control * control_loss + lambda_KL * KLloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        pbar.set_postfix({"Franka: Total Loss": f"{loss.item():.6f}", "Pred Loss": f"{Predloss.item():.6f}", "Reconstruct Loss": f"{Reconloss:.6f}", "Control loss": f"{control_loss.item():.6f}", "KL Loss": f"{KLloss.item():.6f}"})
        optimizer.zero_grad()
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i+1) % eval_step ==0:
            #K loss
            with torch.no_grad():
                Reconloss, KLloss, Predloss = Klinear_loss(X,net,mse_loss,u_dim,gamma,Nstate,all_loss)
                control_loss = Eig_loss(net) + Controlability_loss(net)
                Predloss = Predloss.detach().cpu().numpy()
                Reconloss = Reconloss.detach().cpu().numpy()
                KLloss = KLloss.detach().cpu().numpy()
                control_loss = control_loss.detach().cpu().numpy()
                if Predloss<best_loss:
                    best_loss = copy(Predloss)
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {'model':best_state_dict,'encode_layer':encode_layers,'decode_layer':decode_layers}
                    torch.save(Saved_dict,logdir+".pth")
                print("Method:KoVAE_with_KlinearEig Step:{} Predloss{} Reconloss:{} KLloss{} Controlloss:{} ".format(i,Predloss,Reconloss,KLloss,control_loss))
    print("END-best_loss{}".format(best_loss))
    

def main():
    train(args.env,suffix=args.suffix,all_loss=args.all_loss,\
        encode_dim=args.encode_dim,layer_depth=args.layer_depth,\
            e_loss=args.eloss,gamma=args.gamma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="Franka")
    parser.add_argument("--suffix",type=str,default="")
    parser.add_argument("--all_loss",type=int,default=1)
    parser.add_argument("--eloss",type=int,default=0)
    parser.add_argument("--gamma",type=float,default=0.8)
    parser.add_argument("--encode_dim",type=int,default=20)
    parser.add_argument("--layer_depth",type=int,default=3)
    args = parser.parse_args()
    main()

