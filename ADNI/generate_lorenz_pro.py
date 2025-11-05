# dataset_lorenz_pro.py
import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

def NWnetwork(N, m, p):
    """小世界网络生成器 (Newman-Watts 模型) - 此函数保持不变"""
    matrix = np.zeros((N, N), dtype=bool)
    
    # 生成环形规则网络
    for i in range(N):
        neighbors = [(i + k) % N for k in range(-m, m+1) if k != 0]
        matrix[i, neighbors] = True
    
    # 随机添加长程连接
    rand_mask = np.random.rand(N, N) < p
    np.fill_diagonal(rand_mask, False)
    matrix = matrix | rand_mask
    
    # 确保对称性
    matrix = matrix | matrix.T
    np.fill_diagonal(matrix, False)
    
    return matrix.astype(float), N

# <<< 新增函数: generate_coupled_lorenz_pro >>>
def generate_coupled_lorenz_pro(N=5, L=1000, stepsize=1, C=0.1, m=1, p=0.1, delta=0.01, noise_strength=0.1):
    """
    生成带噪声且可调整时间步距离的耦合洛伦兹系统
    - stepsize: 控制时间步距离，值越大，相邻采样点离得越远
    - noise_strength: 控制添加到最终输出数据上的噪声强度
    """
    # 核心模拟过程与原函数相同
    adjmat, M = NWnetwork(N, m, p)
    sigma = 10.0
    total_l = L * stepsize

    x = np.zeros((M, total_l))
    y = np.zeros((M, total_l))
    z = np.zeros((M, total_l))
    x[:, 0] = np.random.rand(M)
    y[:, 0] = np.random.rand(M)
    z[:, 0] = np.random.rand(M)

    for i in range(total_l-1):
        for j in range(M):
            coupling = C * np.dot(adjmat[j], x[:, i])
            dx = delta * (sigma * (y[j,i] - x[j,i]) + coupling)
            dy = delta * (28 * x[j,i] - y[j,i] - x[j,i] * z[j,i])
            dz = delta * (-8/3 * z[j,i] + x[j,i] * y[j,i])
            x[j,i+1] = x[j,i] + dx
            y[j,i+1] = y[j,i] + dy
            z[j,i+1] = z[j,i] + dz

    X = np.zeros((3*M, total_l))
    for j in range(M):
        X[3*j] = x[j]
        X[3*j+1] = y[j]
        X[3*j+2] = z[j]

    X = X.T
    ret = X[::stepsize]

    # <<< 核心修改：在返回前添加高斯噪声 >>>
    if noise_strength > 0:
        ret = ret + noise_strength * np.random.randn(*ret.shape)

    return ret, X

# --- 原有的 generate_coupled_lorenz 函数可以保留，也可以删除 ---
def generate_coupled_lorenz(N=5, L=1000, stepsize=1, C=0.1, m=1, p=0.1, delta=0.01):
    """完全对齐MATLAB行为的耦合洛伦兹系统生成 (原函数)"""
    adjmat, M = NWnetwork(N, m, p)
    sigma = 10.0
    total_l = L * stepsize
    x, y, z = np.zeros((M, total_l)), np.zeros((M, total_l)), np.zeros((M, total_l))
    x[:, 0], y[:, 0], z[:, 0] = np.random.rand(M), np.random.rand(M), np.random.rand(M)
    for i in range(total_l-1):
        for j in range(M):
            coupling = C * np.dot(adjmat[j], x[:, i])
            dx = delta * (sigma * (y[j,i] - x[j,i]) + coupling)
            dy = delta * (28 * x[j,i] - y[j,i] - x[j,i] * z[j,i])
            dz = delta * (-8/3 * z[j,i] + x[j,i] * y[j,i])
            x[j,i+1], y[j,i+1], z[j,i+1] = x[j,i] + dx, y[j,i] + dy, z[j,i] + dz
    X = np.zeros((3*M, total_l))
    for j in range(M):
        X[3*j], X[3*j+1], X[3*j+2] = x[j], y[j], z[j]
    X = X.T
    ret = X[::stepsize]
    return ret, X


class Lorenz_Dataset(Dataset):
    def __init__(self, data_array, eval_length=100):
        self.eval_length = eval_length
        self.raw_data = data_array
        mean = np.mean(self.raw_data, axis=(0, 1))
        std = np.std(self.raw_data, axis=(0, 1)) + 1e-8
        self.data_mean, self.data_std = mean, std
        self.observed_mask = np.ones_like(self.raw_data, dtype=np.float32)
        self.gt_mask = np.zeros_like(self.raw_data, dtype=np.float32)
        self.gt_mask[:, -1::-2, :] = 1.0
    def __len__(self):
        return self.raw_data.shape[0]
    def __getitem__(self, idx):
        return {
            "observed_data": self.raw_data[idx],
            "observed_mask": self.observed_mask[idx],
            "gt_mask": self.gt_mask[idx],
            "timepoints": np.arange(self.eval_length).astype(np.float32),
        }

# <<< 新增函数: get_dataloader_pro, 用于方便地调用新数据生成函数 >>>
def get_dataloader_pro(batch_size=16, seq_len=100, seq_count=50, stepsize=1, N=5, noise_strength=0.1):
    # 调用新的_pro版本函数
    all_data = np.array([generate_coupled_lorenz_pro(N=N, L=seq_len, stepsize=stepsize, noise_strength=noise_strength)[0] for _ in range(seq_count)]).astype(np.float32)

    total_size, train_size, valid_size = len(all_data), int(len(all_data) * 0.7), int(len(all_data) * 0.15)
    indices = np.random.permutation(total_size)
    train_indices, valid_indices, test_indices = indices[:train_size], indices[train_size:train_size + valid_size], indices[train_size + valid_size:]
    
    train_data, valid_data, test_data = all_data[train_indices], all_data[valid_indices], all_data[test_indices]
    
    train_dataset, valid_dataset, test_dataset = Lorenz_Dataset(train_data, seq_len), Lorenz_Dataset(valid_data, seq_len), Lorenz_Dataset(test_data, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

# --- 原有的 get_dataloader 函数可以保留，也可以删除 ---
def get_dataloader(batch_size=16, seq_len=100, seq_count=50, stepsize=1, N=5):
    # ... 原函数代码不变 ...
    all_data = np.array([generate_coupled_lorenz(N=N, L=seq_len, stepsize=stepsize)[0] for _ in range(seq_count)]).astype(np.float32)
    total_size, train_size, valid_size = len(all_data), int(len(all_data) * 0.7), int(len(all_data) * 0.15)
    indices = np.random.permutation(total_size)
    train_indices, valid_indices, test_indices = indices[:train_size], indices[train_size:train_size + valid_size], indices[train_size + valid_size:]
    train_data, valid_data, test_data = all_data[train_indices], all_data[valid_indices], all_data[test_indices]
    train_dataset, valid_dataset, test_dataset = Lorenz_Dataset(train_data, seq_len), Lorenz_Dataset(valid_data, seq_len), Lorenz_Dataset(test_data, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def generate_lorenz_data(num_samples=10, seq_len=200, num_features=6, missing_rate=0.2, noise_strength=0.1, stepsize=1):
    """
    生成耦合洛伦兹系统数据，并返回与 imputation_main.py 中
    generate_simulated_data 格式完全一致的训练/测试集。

    参数:
    num_samples (int): 要生成的总序列数。
    seq_len (int): 每个序列的长度 (L)。
    num_features (int): 特征维度。必须是 3 的倍数 (N * 3)。
    missing_rate (float): 0 到 1 之间的缺失率。
    noise_strength (float): 传递给 generate_coupled_lorenz_pro 的噪声强度。
    stepsize (int): 传递给 generate_coupled_lorenz_pro 的步长。
    
    返回:
    (X_train, M_train, Y_train, X_test, M_test, Y_test, num_features, seq_len)
    """
    
    print("Generating coupled Lorenz system data...")
    
    # 1. 验证 num_features 并计算 N
    # 洛伦兹系统有 x, y, z 三个维度，所以 N = num_features / 3
    if num_features % 3 != 0:
        raise ValueError(f"num_features ({num_features}) 必须是 3 的倍数 (N * 3) 才能用于洛伦兹系统。")
    N = int(num_features / 3)
    
    # 2. 生成 Y_true (完整的真值数据)
    # 调用 _pro 版本，它已经包含了噪声
    all_data = []
    for _ in range(num_samples):
        # generate_coupled_lorenz_pro 返回 (ret, X)，我们只需要 ret
        lorenz_seq = generate_coupled_lorenz_pro(
            N=N,
            L=seq_len,
            stepsize=stepsize,
            noise_strength=noise_strength
        )[0]
        all_data.append(lorenz_seq)
    
    # 将列表堆叠成一个大的 numpy 数组 (num_samples, seq_len, num_features)
    Y_true_all = np.array(all_data).astype(np.float32)

    # 3. 创建缺失值掩码 (Mask) 和输入数据 (X_raw)
    # 这部分逻辑与 imputation_main.py 中的完全相同
    X_raw_all = Y_true_all.copy()
    Mask_all = np.ones_like(Y_true_all)
    
    # 随机设置缺失点
    num_missing = int(num_samples * seq_len * num_features * missing_rate)
    missing_indices = np.random.choice(Y_true_all.size, num_missing, replace=False)
    
    X_raw_all.flat[missing_indices] = 0.0 # 模拟缺失值
    Mask_all.flat[missing_indices] = 0.0 # 0 表示缺失
    
    # 4. 数据集分割 (70% 训练, 30% 测试)
    # 这部分逻辑也与 imputation_main.py 中的相同
    train_size = int(0.7 * num_samples)
    
    X_train, X_test = X_raw_all[:train_size], X_raw_all[train_size:]
    M_train, M_test = Mask_all[:train_size], Mask_all[train_size:]
    Y_train, Y_test = Y_true_all[:train_size], Y_true_all[train_size:]

    print(f"Total samples: {num_samples}. Features: {num_features} (N={N}). Seq_len: {seq_len}.")
    print(f"Missing rate: {missing_rate}. Train size: {train_size}. Test size: {num_samples - train_size}.")
    
    # 5. 返回 8 个值，格式完全匹配
    return X_train, M_train, Y_train, X_test, M_test, Y_test, num_features, seq_len