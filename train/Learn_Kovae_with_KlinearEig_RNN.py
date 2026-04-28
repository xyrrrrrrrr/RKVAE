import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
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
from Utility import data_collecter
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
        self.k = k  # K近邻数量
        self.neighbor_indices = None  # 不再预存全局索引，改为batch内临时存储

    def compute_knn(self, X):
        """针对单个batch的X，计算每个样本的K近邻索引（仅在当前batch内）"""
        # 计算X的 pairwise 距离（欧氏距离）
        n = X.shape[0]
        dist_matrix = torch.cdist(X, X, p=2)  # shape=[n, n]
        # 取每个样本的前k+1个近邻（排除自身，所以k+1），再去掉第0个（自身）
        _, indices = torch.topk(dist_matrix, k=self.k+1, largest=False, dim=1)
        self.neighbor_indices = indices[:, 1:]  # shape=[n, k]，每个样本的k个邻居索引
        return self.neighbor_indices

    def forward(self, z, X):
        """
        z: 当前batch的嵌入张量，shape=[batch*T, manifold_dim]
        X: 当前batch的原状态张量，shape=[batch*T, x_dim]
        """
        # 第一步：针对当前batch的X，动态计算K近邻索引
        self.compute_knn(X)
        # 第二步：根据邻居索引，提取z和X的邻居样本
        n = z.shape[0]
        x_dim = X.shape[1]
        z_dim = z.shape[1]
        # 确保索引在合法范围内（双重保险）
        self.neighbor_indices = torch.clamp(self.neighbor_indices, 0, n-1)
        
        # 提取每个样本的邻居（shape=[n, k, dim]）
        z_neighbors = z[self.neighbor_indices]  # [n, k, manifold_dim]
        x_neighbors = X[self.neighbor_indices]  # [n, k, x_dim]
        
        # 计算原状态与邻居的距离、嵌入后与邻居的距离
        x_dist = torch.cdist(X.unsqueeze(1), x_neighbors, p=2).squeeze(1) 
        z_dist = torch.cdist(z.unsqueeze(1), z_neighbors, p=2).squeeze(1) 

        x_dist_max = torch.max(x_dist, dim=1, keepdim=True)[0]
        x_dist_max = torch.clamp(x_dist_max, min=1e-8)  # 防止过小导致梯度爆炸
        x_dist = x_dist / x_dist_max  # 归一化，避免尺度
        
        z_dist_max = torch.max(z_dist, dim=1, keepdim=True)[0]
        z_dist_max = torch.clamp(z_dist_max, min=1e-8)  # 防止过小导致梯度爆炸
        z_dist = z_dist / z_dist_max  # 归一化，避免尺度
        # z_dist = z_dist * x_dim / (x_dist_max * z_dim)
        dist_diff = torch.abs(z_dist - x_dist)
        huber_loss = F.huber_loss(dist_diff, torch.zeros_like(dist_diff), delta=0.1, reduction='none')
        # # 计算几何一致性损失
        # loss = torch.mean(torch.abs(z_dist - x_dist))
        loss = torch.mean(huber_loss)

        return loss

class Network(nn.Module):
    def __init__(self, encode_layers, decoder_layers, Nkoopman, u_dim, x_dim, device=None, lstm_hidden_dim=None, lstm_num_layers=2):
        """
        Args:
            encode_layers: 编码器特征提取层维度（如[64, 32]，输入→中间特征→最终特征）
            decoder_layers: 解码器层维度（如[32, 64, x_dim]，latent+u→重建x）
            Nkoopman: Koopman latent空间维度（VAE的latent维度）
            u_dim: 控制量维度
            x_dim: 观测x的维度（用于解码器输出匹配输入）
            device: 计算设备
            lstm_hidden_dim: LSTM 隐藏层维度（默认=None，自动适配 encode_layers[-1]，保证兼容原代码）
            lstm_num_layers: LSTM 层数（默认=1，简单场景无需多层）
        """
        super(Network, self).__init__()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Nkoopman = Nkoopman  # VAE latent维度
        self.x_dim = x_dim        # 观测维度
        self.u_dim = u_dim        # 控制量维度
        self.traj_dim = x_dim + u_dim
        self.activation = nn.ReLU()
        
        # -------------------------- 新增：LSTM 相关参数配置 --------------------------
        self.lstm_hidden_dim = lstm_hidden_dim if lstm_hidden_dim is not None else encode_layers[-1]
        self.lstm_num_layers = lstm_num_layers
        
        # -------------------------- 1. 变分编码器（Inference Network）--------------------------
        # 改造：encode_feature 替换为「linear+lstm+linear」结构
        # 输入线性层：将原始输入（encode_layers[0]）映射到 LSTM 输入维度
        self.enc_linear_in = nn.Linear(encode_layers[0], self.lstm_hidden_dim)
        # LSTM 层：处理序列特征（batch_first=True 方便批量数据处理，输入格式 (batch, seq_len, feature)）
        self.enc_lstm = nn.LSTM(
            input_size=self.lstm_hidden_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            bidirectional=False  # 单向LSTM，保证特征维度简洁
        )
        # 输出线性层：将 LSTM 输出映射到原 encode_layers[-1] 维度，保证后续层兼容
        self.enc_linear_out = nn.Linear(self.lstm_hidden_dim + self.x_dim, encode_layers[-1])
        
        # 以下部分完全保留，维度不变，保证功能兼容
        self.fc_mu = nn.Linear(encode_layers[-1], encode_layers[-1])
        self.fc_logvar = nn.Linear(encode_layers[-1], encode_layers[-1])
        # self.gx = nn.Linear(encode_layers[-1], u_dim)
        
        # -------------------------- 2. Koopman latent先验（Prior Network）--------------------------
        # 完全保留，无任何修改
        self.lA = nn.Linear(Nkoopman, Nkoopman, bias=False)  # 状态转移矩阵 A
        self.lB = nn.Linear(u_dim, Nkoopman, bias=False)     # 控制输入矩阵 B
        self.lC = nn.ModuleList([nn.Linear(Nkoopman, Nkoopman, bias=False) for _ in range(u_dim)]) # 双线性矩阵 C
        self.process_logvar = nn.Parameter(torch.ones(1, Nkoopman-self.x_dim) * 1e-3)
        # 初始化lA（沿用原网络正交化+幅值约束，确保初始稳定性）
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1.0)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.99  # 幅值0.99，避免初始发散
        # 初始化lB（高斯初始化）
        nn.init.normal_(self.lB.weight.data, mean=0.0, std=0.1)
        # -------------------------- 3. 解码器（Generative Network）--------------------------
        # 完全保留，无任何修改
        self.decode_net = self._build_mlp(decoder_layers, activation=self.activation)
        
        # 设备迁移
        self.to(self.device)

    def _build_mlp(self, layers, activation=nn.ReLU()):
        """辅助函数：构建MLP（仅用于解码器，编码器已改造为LSTM结构）"""
        mlp = OrderedDict()
        for i in range(len(layers) - 1):
            mlp[f"linear_{i}"] = nn.Linear(layers[i], layers[i+1])
            # 最后一层不添加激活（编码器输出特征无激活，解码器输出观测无激活）
            if i != len(layers) - 2:
                mlp[f"act_{i}"] = activation
        return nn.Sequential(mlp)

    def reparameterize(self, mu, logvar=None):
        """VAE核心：重参数化技巧，使latent采样可微分（完全保留）"""
        std = torch.exp(0.5 * logvar)  # 标准差 = sqrt(exp(logvar))
        eps = torch.randn_like(std, device=self.device) # 标准高斯噪声 N(0,1)
        return mu + eps * std  # z = mu + eps*std ~ q(z|x,u)

    def _encode_feature(self, x_k, traj):
        # 步骤1：调整维度适配 LSTM 输入格式（batch_first=True）
        # transpose后形状：[1, batch, feat]（对应 LSTM 默认输入格式：(seq_len, batch, input_size)）
        traj = traj.transpose(1, 0)
        # x_k = x_k.transpose(1, 0)
        # 步骤2：输入线性层 + 激活（nn.Linear 自动作用于最后一维，不改变前两维时序和批量）
        # 形状变化：[T, batch, feat] → [T, batch, lstm_hidden_dim]
        traj_embed = self.activation(self.enc_linear_in(traj))
        
        # 步骤3：LSTM 前向传播（获取所有时间步输出和最终隐藏状态）
        # output：所有时间步的输出，形状 [T, batch, lstm_hidden_dim]（保留完整时序信息）
        # hn：各层最后一个时间步的隐藏状态，形状 [num_layers, batch, lstm_hidden_dim]
        _, (hn, _) = self.enc_lstm(traj_embed)
        output = hn[-1, :, :]
        output = torch.concat([output, x_k], dim = -1)
        feat = self.activation(self.enc_linear_out(output))
        return feat

    def encode(self, x, traj=None):
        """编码器：输入x（观测），输出latent z的近似后验参数（仅改造特征提取部分）"""
        if traj == None:
            traj = torch.zeros([x.shape[1], x.shape[0],self.traj_dim], dtype=torch.float64).to(self.device)
        # 步骤1：使用 LSTM 提取高维特征（替换原 MLP 特征提取）
        feat = self._encode_feature(x, traj)
        # 步骤2：后续逻辑完全保留，维度兼容
        mu_z = self.fc_mu(feat)
        logvar_z = self.fc_logvar(feat)
        # gx = self.gx(feat)
        z = self.reparameterize(mu_z, logvar_z)
        # print(x.shape , z.shape)
        mu_xz = torch.cat([x, mu_z], axis=-1)
        # return mu_xz, z, mu_z, logvar_z, gx
        return mu_xz, z, mu_z, logvar_z
    

    # def control_encode(self, x, traj=None):
    #     """控制量编码（仅改造特征提取部分，后续逻辑完全保留）"""
    #     if traj == None:
    #         traj = torch.zeros([1, x.shape[0] ,self.traj_dim], dtype=torch.float64).to(self.device)
    #     feat = self._encode_feature(x, traj)
    #     gx = self.gx(feat)

    #     return gx
    

    def prior(self, mu_xz, mu_z, u, logvar_z, eps=1e-6):
        """
        Args:
            mu_xz: 上一时刻的全状态均值 [batch, Nkoopman]
            mu_z: 上一时刻 latent z 的均值
            u: 控制量
            logvar_z: 上一时刻 latent z 的对数方差
        """
        # 1. 均值预测 (保持不变，或者按之前建议加入双线性项)
        mu_xz_next = self.lA(mu_xz) + self.lB(u)
        mu_prior = mu_xz_next[:, self.x_dim:] # 提取 z 的部分

        # 2. 构建上一时刻的完整方差向量 (假设观测 x 是确定的，方差为0)
        # logvar_xz 对应 [x, z]，其中 x 部分设为极小值(模拟0方差)
        var_xz = torch.zeros_like(mu_xz)
        var_xz[:, self.x_dim:] = torch.exp(logvar_z) # 只填充 z 的方差

        # 3. 计算传播方差 (Propagated Variance)
        # 原理: Var(Ax) = A * Var(x) * A^T
        # 对于对角方差矩阵，这等价于: sum(A^2 * var_x, dim=1)
        # 使用 F.linear 进行高效计算：input=var, weight=A^2
        Ad_sq = self.lA.weight ** 2
        var_propagated = F.linear(var_xz, Ad_sq) # 结果维度 [batch, Nkoopman]
        
        # 只取 z 的部分 (因为 prior 只约束 z)
        var_propagated_z = var_propagated[:, self.x_dim:]

        # ================= [修改核心 START] =================
        # 4. 加入过程噪声 (Add Process Noise)
        # 模拟系统方程中的 v_k： Var_total = Var_prop + Var_noise
        var_process = torch.exp(self.process_logvar) # [1, z_dim]
        
        # 广播相加
        var_prior_total = var_propagated_z + var_process 
        
        # 5. 转回 log 域
        logvar_prior = torch.log(var_prior_total + eps) # 加 eps 防止 log(0)
        # ================= [修改核心 END] =================

        return mu_xz_next, mu_prior, logvar_prior
    
    def forward(self, mu_xz, mu_z, u_prev, logvar_z):
        """前向传播"""
        mu_xz_next, mu_prior, logvar_prior = self.prior(mu_xz, mu_z, u_prev, logvar_z)
        z_next = self.reparameterize(mu_prior, logvar_prior)
        return mu_xz_next, z_next, mu_prior, logvar_prior

    def decode(self, z):
        """解码器：输入latent z+控制量u，重建观测x（完全保留）"""
        x_recon = self.decode_net(z)    # 重建观测，维度：(batch, x_dim)
        return x_recon

    def compute_KL_loss(self, mu_z, logvar_z, mu_prior, logvar_prior):
        """计算KL散度（完全保留）"""
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
    mu_xz, z_current, mu_z, logvar_z = net.encode(data[0,:,u_dim:])
    max_loss_list = []
    mean_loss_list = []
    x_dim = data.shape[2] - u_dim
    for i in range(steps-1):
        mu_xz_next, z_next, mu_prior, logvar_prior = net.forward(mu_xz,mu_z,data[i,:,:u_dim],logvar_z)
        x_pred = mu_xz_next[:,:x_dim]
        y = data[i+1,:,u_dim:]
        Err = x_pred-y
        mu_xz = mu_xz_next
        mu_z = mu_prior
        logvar_z = logvar_prior
        max_loss_list.append(torch.mean(torch.max(torch.abs(Err),axis=0).values).detach().cpu().numpy())
        mean_loss_list.append(torch.mean(torch.mean(torch.abs(Err),axis=0)).detach().cpu().numpy())
    return np.array(max_loss_list),np.array(mean_loss_list)


#loss function
def Klinear_loss(data,net,mse_loss,emb_loss,u_dim=1,gamma=0.99,Nstate=4,all_loss=0,lambda_geom=0):
    steps,train_traj_num,NKoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)
    batch_size = data.shape[1]
    x_dim = data.shape[2] - u_dim
    x = data[:,:,u_dim:]
    u = data[:, :, :u_dim]
    mu_xz, z_current, mu_z, logvar_z = net.encode(x[0,:,:])
    # 计算流形几何约束损失
    Geomloss = torch.tensor(0.0, dtype=torch.float64, device=device)
    if lambda_geom > 0:
        # 随机选择一些点对计算距离约束
        id_traj = torch.randint(0, train_traj_num, (min(100, batch_size//2),), device=device)
        # idx_time = torch.randint(0, steps - 1,  (1,), device=device)
        x_samples = x[0, id_traj, :]
        # compute z
        mu_xz_samples = mu_xz[id_traj, :]
        # get loss
        Geomloss = emb_loss(mu_xz_samples, x_samples)
    beta = 0.9
    beta_sum = 0.0
    Augloss = torch.tensor(0.0, dtype=torch.float64, device=device)
    Reconloss = torch.tensor(0.0, dtype=torch.float64, device=device)
    KLloss = torch.tensor(0.0, dtype=torch.float64, device=device)
    Predloss = torch.tensor(0.0, dtype=torch.float64, device=device)
    for i in range(steps-1):
        mu_xz_next, z_next, mu_prior, logvar_prior = net.forward(mu_xz, mu_z, u[i,:,:], logvar_z)
        traj = data[:i+1,:,:] if i != 0 else data[0,:,:].unsqueeze(0)
        mu_xz_next_real, z_next_real, mu_z_next_real, logvar_z_next_real = net.encode(x[i+1,:,:], traj)
        x_recon = net.decode(z_current)
        beta_sum += beta
        # Reconstruction loss
        Reconloss += beta*(mse_loss(x_recon, x[i,:,:]))
        # KL divergence loss
        KLloss += beta*net.compute_KL_loss(mu_z_next_real, logvar_z_next_real, mu_prior, logvar_prior)
        # Prediction loss
        if not all_loss:
            Predloss += beta*mse_loss(mu_xz_next[:,:x_dim], mu_xz_next_real[:,:x_dim])
        else:
            # print(mu_xz_next.shape, mu_xz_next_real.shape)
            Predloss += beta*mse_loss(mu_xz_next,mu_xz_next_real)
        Augloss += beta*mse_loss(mu_xz_next_real[:,:x_dim],mu_xz_next[:,:x_dim])
        # mu_xz, z_current, mu_z, logvar_z = mu_xz_next_real, z_next_real, mu_z_next_real, logvar_z_next_real
        mu_xz, z_current, mu_z, logvar_z = mu_xz_next, z_next_real, mu_prior, logvar_prior
    Augloss = Augloss/beta_sum
    Reconloss = Reconloss/beta_sum
    KLloss = KLloss/beta_sum
    Predloss = Predloss/beta_sum
    Predloss += 0.5*Augloss
    return Reconloss, KLloss, Predloss, Geomloss

def Stable_loss(net,Nstate):
    x_ref = np.zeros((1,Nstate)) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu_xz, z, mu_z, logvar_z = net.encode(torch.DoubleTensor(x_ref).to(device))
    loss = torch.norm(mu_xz)
    return loss

def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs()-1.0*torch.ones(1,dtype=torch.float64).to(device)#预留抑制噪声的量，不希望噪声保持传播
    mask = c>0
    loss = c[mask].sum()
    return loss

# 能控性损失
def Controlability_loss(net, eval_=False):
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
    
    varepsilon = 1e-6 if eval_ else 1e-2
    loss = -min_singular + varepsilon  # 当最小奇异值 ≥ epsilon时，损失趋近于0
    
    return loss.clamp(min=0.0)  # 确保损失非负（奇异值过小时才产生惩罚）

def train(env_name,train_steps = 200000,suffix="",all_loss=0,\
            encode_dim = 12,layer_depth=3,e_loss=1,gamma=0.5,Ktrain_samples=50000,\
        lambda_geom=0.1,\
        lambda_recon=0.1,\
        lambda_control=0.1,\
        lambda_KL=0.1,\
        device=0):
    # Ktrain_samples = 1000
    # Ktest_samples = 1000
    torch.cuda.set_device(device)
    Ktrain_samples = Ktrain_samples
    # Ktrain_samples = 200
    Ktest_samples = 20000
    Ktrainsteps = 15
    Kteststeps = 30
    Kbatch_size = 512
    #data prepare
    data_collect = data_collecter(env_name)
    u_dim = data_collect.udim
    # Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Kteststeps,mode="eval")
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Kteststeps,mode="train") if env_name != "CartPole-v1" and env_name !="MountainCarContinuous-v0" else data_collect.collect_koopman_data(Ktest_samples,Kteststeps,mode="eval")
    Ktest_samples = Ktest_data.shape[1]
    print("test data ok!,shape:",Ktest_data.shape)
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples,Ktrainsteps,mode="train")
    print("train data ok!,shape:",Ktrain_data.shape)
    Ktrain_samples = Ktrain_data.shape[1]
    # in_dim = Ktest_data.shape[-1]-u_dim
    in_dim = Ktest_data.shape[-1]
    Nstate = in_dim - u_dim
    # layer_depth = 4
    layer_width = 128
    encode_layers = [in_dim]+[layer_width]*layer_depth+[encode_dim]
    Nkoopman = encode_dim + Nstate
    decode_layers = [encode_dim] + [layer_width]*layer_depth + [Nstate]
    print("encode layers:",encode_layers)
    print("decode layers:",decode_layers)
    net = Network(encode_layers,decode_layers,Nkoopman,u_dim,Nstate)
    # print(net.named_modules())
    eval_step = 1000
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
    eval_step = 1000
    best_loss = 1000.0
    best_control_loss = 1000.0
    convergence = 0
    best_iteration = 0
    best_state_dict = {}
    logdir = "../Data/"+suffix+"/RKVAE_"+env_name+"layer{}_edim{}_eloss{}_gamma{}_aloss{}_samples{}_recon{}_control{}_KL{}_geom{}".format(layer_depth,encode_dim,e_loss,gamma,all_loss,Ktrain_samples,lambda_recon,lambda_control,lambda_KL,lambda_geom)
    currentdir = "../Data/"+suffix+"/RKVAE_"+env_name + "_current"
    if not os.path.exists( "../Data/"+suffix):
        os.makedirs( "../Data/"+suffix)
    start_time = time.process_time()
    pbar = tqdm.trange(train_steps)
    for i in pbar:
        #K loss
        Kindex = list(range(Ktrain_samples))
        random.shuffle(Kindex)
        X = Ktrain_data[:,Kindex[:Kbatch_size],:]
        Reconloss, KLloss, Predloss, Geomloss = Klinear_loss(X,net,mse_loss,emb_loss,u_dim,gamma,Nstate,all_loss,lambda_geom)
        # control_loss = Eig_loss(net) + Controlability_loss(net) + Stable_loss(net,in_dim)
        control_loss = Eig_loss(net) + Controlability_loss(net) + Stable_loss(net,Nstate)
        loss = Predloss + lambda_recon * Reconloss + lambda_control * control_loss + lambda_KL * KLloss + lambda_geom * Geomloss
        # loss = Kloss
        # pbar.set_postfix({"Total Loss": f"{loss.item():.6f}", "Pred Loss": f"{Predloss.item():.6f}", "Reconstruct Loss": f"{Reconloss:.6f}", "Control loss": f"{control_loss.item():.6f}", "KL Loss": f"{KLloss.item():.6f}", "Geom Loss": f"{Geomloss.item():.6f}"})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i+1) % eval_step ==0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.98
            convergence += 1
            with torch.no_grad():
                Reconloss, KLloss, Predloss, Geomloss = Klinear_loss(Ktest_data,net,mse_loss,emb_loss,u_dim,gamma,Nstate,all_loss=0)
                Eigloss = Eig_loss(net)
                control_loss = Controlability_loss(net, eval_=True) + Stable_loss(net,Nstate)
                Predloss = Predloss.detach().cpu().numpy()
                Reconloss = Reconloss.detach().cpu().numpy()
                KLloss = KLloss.detach().cpu().numpy()
                control_loss = control_loss.detach().cpu().numpy()
                Saved_dict = {'model':net.state_dict(),'encode_layer':encode_layers,'decode_layer':decode_layers}
                torch.save(Saved_dict,currentdir+".pth")
                if Predloss<best_loss and control_loss<best_control_loss * 2.0 and Eigloss == 0:
                    print("Best model updated at iteration ", i)
                    convergence = 0
                    best_loss = copy(Predloss)
                    best_control_loss = copy(control_loss)
                    best_iteration = i
                    best_state_dict = copy(net.state_dict())
                    Saved_dict = {'model':best_state_dict,'encode_layer':encode_layers,'decode_layer':decode_layers}
                    torch.save(Saved_dict,logdir+".pth")
                print("Method:RKVAE_with_KlinearEig Step:{} Predloss{} Reconloss:{} KLloss{} Controlloss:{} Eigloss:{} ".format(i,Predloss,Reconloss,KLloss,control_loss,Eigloss))
            if convergence >= 20:
                print("Early stopping at iteration ", i)
                break
            # print("-------------END-------------")
        # if (time.process_time()-start_time)>=210*3600:
        #     print("time out!:{}".format(time.clock()-start_time))
        #     break
    print("END-best_loss{}-best_iteration{}".format(best_loss, best_iteration))
    

def main():
    train(args.env,suffix=args.suffix,all_loss=args.all_loss,\
        encode_dim=args.encode_dim,layer_depth=args.layer_depth,\
        e_loss=args.e_loss,gamma=args.gamma,\
        Ktrain_samples=args.K_train_samples,\
        lambda_geom=args.lambda_geom,\
        lambda_recon=args.lambda_recon,\
        lambda_control=args.lambda_control,\
        lambda_KL=args.lambda_KL,
        device=args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="DampingPendulum")
    parser.add_argument("--suffix",type=str,default="5_2")
    parser.add_argument("--all_loss",type=int,default=0)
    parser.add_argument("--K_train_samples",type=int,default=50000)
    parser.add_argument("--e_loss",type=int,default=1)
    parser.add_argument("--gamma",type=float,default=0.9)
    parser.add_argument("--encode_dim",type=int,default=20)
    parser.add_argument("--layer_depth",type=int,default=2)
    parser.add_argument("--lambda_geom", type=float, default=0.0, help="流形几何约束权重")
    parser.add_argument("--lambda_recon", type=float, default=0.4, help="重建约束权重")
    parser.add_argument("--lambda_control", type=float, default=0.2, help="控制约束权重")
    parser.add_argument("--lambda_KL", type=float, default=0.5, help="散度约束权重")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()
    main()

