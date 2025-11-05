import torch
import torch.nn as nn
import numpy as np
import math

from torchdiffeq import odeint as odeint

# --- 1. 几何和 ODE 辅助模块 ---

class RGRUCell(nn.Module):
    """
    An implementation of RGRUCell. (Copied from ODERGRU.py)
    """
    def __init__(self, input_size, hidden_size, diag=True):
        super(RGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.diag = diag
        if diag:
            layer = PosLinear
            self.nonlinear = nn.Softplus()
        else:
            layer = nn.Linear
            self.nonlinear = nn.Tanh()
        self.x2h = layer(input_size, 3 * hidden_size, bias=False)
        self.h2h = layer(hidden_size, 3 * hidden_size, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_size * 3))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        b_r, b_i, b_n = self.bias.chunk(3, 0)

        if self.diag:
            resetgate = (b_r.abs() * (i_r.log() + h_r.log()).exp()).sigmoid()
            inputgate = (b_i.abs() * (i_i.log() + h_i.log()).exp()).sigmoid()
            newgate = self.nonlinear((b_n.abs() * (i_n.log() + (resetgate * h_n).log()).exp()))
            hy = (newgate.log() * (1 - inputgate) + inputgate * hidden.log()).exp()
        else:
            resetgate = (i_r + h_r + b_r).sigmoid()
            inputgate = (i_i + h_i + b_i).sigmoid()
            newgate = self.nonlinear(i_n + (resetgate * h_n) + b_n)
            hy = newgate + inputgate * (hidden - newgate)

        return hy

class PosLinear(nn.Module):
    """PosLinear layer for diagonal elements. (Copied from ODERGRU.py)"""
    def __init__(self, in_dim, out_dim, bias=False):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_dim, out_dim)))

    def forward(self, x):
        return torch.matmul(x, torch.abs(self.weight))

class odefunc(nn.Module):
    """Neural ODE function network. (Copied from ODERGRU.py)"""
    def __init__(self, n_inputs, n_layers, n_units):
        super(odefunc, self).__init__()
        self.Layers = nn.ModuleList()
        self.Layers.append(nn.Linear(n_inputs, n_units))
        for i in range(n_layers):
            self.Layers.append(
                nn.Sequential(
                    nn.Tanh(),
                    nn.Linear(n_units, n_units)
                )
            )
        self.Layers.append(nn.Tanh())
        self.Layers.append(nn.Linear(n_units, n_inputs))

    def forward(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x

class ODEFunc(nn.Module):
    """ODEFunc wrapper. (Copied from ODERGRU.py)"""
    def __init__(self, n_inputs, n_layers, n_units):
        super(ODEFunc, self).__init__()
        self.gradient_net = odefunc(n_inputs, n_layers, n_units)

    def forward(self, t_local, y, backwards=False):
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad

    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)

    def sample_next_point_from_prior(self, t_local, y):
        return self.get_ode_gradient_nn(t_local, y)

# --- 2. 协方差估计辅助函数和编码器 ---

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov (Copied from ODERGRU.py)"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def oas_cov(X):
    """OAS Shrinkage Covariance Estimation (Copied from ODERGRU.py)"""
    n_samples, n_features = X.shape
    emp_cov = cov(X)
    mu = emp_cov.diag().sum() / n_features

    alpha = (emp_cov ** 2).mean()
    num = alpha + mu ** 2
    den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)

    shrinkage = 1. if den == 0 else torch.minimum((num / den), mu.new_ones(1))
    shrunk_cov = (1. - shrinkage) * emp_cov
    shrunk_cov.flatten()[::n_features + 1] += shrinkage * mu

    return shrunk_cov

class ImputationEncoder(nn.Module):
    """
    Imputation 任务的编码器 (已修复问题 2 和 3)
    1. 卷积层输出 latents*latents 通道
    2. forward 方法使用向量化操作
    """
    def __init__(self, C, latents):
        super(ImputationEncoder, self).__init__()
        # C 是输入通道/特征数。 latents 是 SPD 矩阵维度 (d=32)。
        
        self.layers = nn.ModuleList()
        self.latents = latents
        
        # 卷积层 1
        self.layers.append(nn.Conv1d(C, latents, kernel_size=5, padding=2)) 
        self.layers.append(nn.BatchNorm1d(latents))
        self.layers.append(nn.LeakyReLU())
        
        # --- 这是您必须修改的地方 ---
        # 卷积层 2 (必须输出 latents * latents 通道)
        self.layers.append(nn.Conv1d(latents, latents * latents, kernel_size=5, padding=2))
        self.layers.append(nn.BatchNorm1d(latents * latents))
        # --- -------------------- ---
        
        self.layers.append(nn.LeakyReLU())
        

    def forward(self, x):
        # x 形状: [B, S, C]
        
        x = x.transpose(1, 2) # [B, C, S]
        b, c, s = x.shape
        
        for layer in self.layers:
            x = layer(x)
            
        # 经过修改的 __init__ 后, x 形状: [B, latents*latents, S]
        
        # 1. 调整形状 (向量化操作)
        # (B, D*D, S) -> (B, S, D*D)
        x = x.transpose(1, 2)
        
        # (B, S, D*D) -> (B, S, D, D)
        # 这就是我们构造的满秩矩阵 A
        # (这里的 b, s 来自于 x.transpose(1, 2) 之后)
        A_matrix = x.reshape(x.shape[0], x.shape[1], self.latents, self.latents)
            
        # 2. 生成满秩的 SPD 矩阵 (A @ A.T)
        cov_seq = A_matrix @ A_matrix.transpose(-1, -2)
            
        # 3. 添加扰动以确保数值稳定性和正定性
        eye = torch.eye(self.latents, device=x.device).unsqueeze(0).unsqueeze(0) * 1e-2
        cov_seq = cov_seq + eye
             
        return cov_seq

# --- 3. Imputation 任务适配后的主模型 ---

class ImputationODERGRU(nn.Module):
    """
    ODE-RGRU model adapted for Imputation (Sequence-to-Sequence Regression).
    - latents=32, units=16.
    - Classification layer replaced by Regression layer.
    """
    # 使用论文指定的 latents=32 和 units=16 作为默认值
    def __init__(self, output_dim, n_layers, n_units, latents=32, units=16, channel=6, bi=False, device='cpu'):
        super(ImputationODERGRU, self).__init__()

        self.latents = latents
        self.units = units
        self.bi = bi
        self.output_dim = output_dim
        
        # 1. 编码器: 使用修改后的 ImputationEncoder
        self.encoder = ImputationEncoder(C=channel, latents=latents) 
        
        # 隐藏状态的维度 (Cholesky空间元素数)： units * (units+1)/2
        rgru_input_dim = units * (units + 1) // 2 
        
        # 2. ODE-RGRU核心
        self.rgru_d = RGRUCell(latents, units, True)
        self.rgru_l = RGRUCell(latents * (latents - 1) // 2, units * (units - 1) // 2, False)
        # n_inputs: RGRU隐藏状态的维度
        self.odefunc = ODEFunc(n_inputs=rgru_input_dim, n_layers=n_layers, n_units=n_units)

        if bi:
            self.odefunc_re = ODEFunc(n_inputs=rgru_input_dim, n_layers=n_layers, n_units=n_units)
            self.rgru_d_re = RGRUCell(latents, units, True)
            self.rgru_l_re = RGRUCell(latents * (latents - 1) // 2, units * (units - 1) // 2, False)

        self.softplus = nn.Softplus()
        
        # 3. 回归输出层：从 RGRU 隐藏状态维度映射回原始特征维度 (output_dim)
        self.reg = nn.Linear(rgru_input_dim, output_dim) 

        self.device = device

    def chol_de(self, x):
        """Cholesky Decomposition (Vectorized)"""
        b, s, n, n = x.shape
        x = x.reshape(-1, n, n) # Shape: [B*S, N, N]
        
        # 1. 计算 Cholesky 分解
        L = torch.linalg.cholesky(x) # Shape: [B*S, N, N]

        # 2. 提取对角线元素 (Vectorized)
        # torch.diagonal 可以在最后两个维度上批量提取对角线
        d = torch.diagonal(L, dim1=-2, dim2=-1) # Shape: [B*S, N]

        # 3. 提取非对角线元素 (Vectorized)
        # 您的原始代码 l[i] = torch.cat([L[i][j: j + 1, :j] for j in range(1, n)], dim=1)[0]
        # 实际上就是在提取 L 矩阵的严格下三角部分 (不含对角线)
        
        # 生成下三角矩阵的索引 (offset=-1 表示不包括对角线)
        tril_indices = torch.tril_indices(n, n, offset=-1, device=x.device)
        
        # L[:, tril_indices[0], tril_indices[1]]
        # 这行代码会批量地从 L 中的每个矩阵里，
        # 一次性取出所有 (j, :j) 对应的元素。
        l = L[:, tril_indices[0], tril_indices[1]] # Shape: [B*S, N*(N-1)//2]

        # 4. Reshape 并返回
        return d.reshape(b, s, -1), l.reshape(b, s, -1)

    def chol_de_1(self, x):
        """Cholesky Decomposition (Copied from ODERGRU.py)"""
        b, s, n, n = x.shape
        
        # 1. 保存原始设备 (例如 'cuda:0')
        original_device = x.device
        
        # 2. 将 x 移动到 CPU 并执行 cholesky
        x_cpu = x.reshape(-1, n, n).cpu()
        
        # x.cholesky() 只能在 CPU 上运行，除非您已安装兼容 CUDA 的 torch 版本
        L = torch.linalg.cholesky(x_cpu)
        
        # 3. 在 CPU 上创建 d 和 l
        d_cpu = torch.zeros(b * s, n) # 使用 torch.zeros, 因为 x_cpu.new_zeros 仍可能指向 cuda
        l_cpu = torch.zeros(b * s, n * (n - 1) // 2)
        
        for i in range(b * s):
            d_cpu[i] = L[i].diag()
            l_cpu[i] = torch.cat([L[i][j: j + 1, :j] for j in range(1, n)], dim=1)[0]
            
        # 4. 将结果移回原始设备 (GPU)
        return d_cpu.reshape(b, s, -1).to(original_device), l_cpu.reshape(b, s, -1).to(original_device)
        
    def forward(self, x,mask):
        # x 形状: [B, S, C] (Imputation 任务输入序列)
        
        x_cov = self.encoder(x) # [b, s, latents, latents]
        b, s, c, c = x_cov.shape
        x_d, x_l = self.chol_de(x_cov) # [b, s, D_d], [b, s, D_l]

        # RGRU/ODE 状态初始化
        h_d = torch.ones(b, self.units, device=self.device)*0.1
        h_l = torch.zeros(b, self.units * (self.units - 1) // 2, device=self.device)
        times = torch.from_numpy(np.arange(s + 1)).float().to(self.device)
        h_seq = [] 
        hp = torch.cat((h_d.log(), h_l), dim=1)
        
        for i in range(s):
            # 1. ODE 传播 (始终发生)
            # hp 是 i-1 时刻的状态，hp_ode 是 ODE 预测的 i 时刻的状态
            hp_ode = odeint(self.odefunc, hp, times[i:i + 2], rtol=1e-4, atol=1e-5, method='euler')[1]
            h_d_ode = hp_ode[:, :self.units].tanh().exp()
            h_l_ode = hp_ode[:, self.units:]

            # 2. RGRU 更新 (仅在观测到数据时发生)
            # h_d_gru 是 i 时刻 RGRU 的更新状态
            h_d_gru = self.rgru_d(x_d[:, i, :], h_d_ode)
            h_l_gru = self.rgru_l(x_l[:, i, :], h_l_ode)
            
            # 3. 获取当前时间步的掩码 (mask_i)
            # 假设: 只要在时间 i 有 任何一个 (C) 特征被观测到，我们就进行 RGRU 更新
            # mask 形状: [B, S, C]
            mask_i = (mask[:, i, :].sum(dim=-1, keepdim=True) > 0).float()
            # mask_i 形状变为: [B, 1] (1.0 代表此样本在 i 时刻有观测值, 0.0 代表无)
            
            # 4. 条件更新：
            # 如果 mask_i = 1 (有观测), 则使用 h_d_gru
            # 如果 mask_i = 0 (无观测), 则保持 h_d_ode (只相信 ODE 的预测)
            h_d = (h_d_gru * mask_i) + (h_d_ode * (1 - mask_i))
            h_l = (h_l_gru * mask_i) + (h_l_ode * (1 - mask_i))
            
            # 5. 存储新的隐藏状态 (用于下一个 i+1 循环)
            hp = torch.cat((h_d.log(), h_l), dim=1)
            h_seq.append(hp)

        h = torch.stack(h_seq, dim=1) # [b, s, rgru_input_dim]
        
        # 4. 回归预测 (序列到序列)
        y_pred = self.reg(h) # [b, s, output_dim]
        
        return y_pred