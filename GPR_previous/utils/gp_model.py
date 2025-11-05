import torch
import gpytorch
import numpy as np

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # 核心修改：移除WhiteNoiseKernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def mypowerseries_gpu(X, p=2):
    """GPU加速的多项式特征生成"""
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, device='cuda', dtype=torch.float32)
    
    n_samples, n_features = X.shape
    powers = []
    
    # 生成幂次组合
    for total_power in range(p+1):
        for i in range(n_features):
            if total_power == 0:
                powers.append([0]*n_features)
            else:
                for j in range(total_power+1):
                    vec = [0]*n_features
                    vec[i] = j
                    if sum(vec) <= p:
                        powers.append(vec)
    
    # 去重
    unique_powers = []
    [unique_powers.append(x) for x in powers if x not in unique_powers]
    
    # 构建特征矩阵
    H = torch.ones(n_samples, len(unique_powers), device='cuda')
    for i, power in enumerate(unique_powers):
        for j in range(n_features):
            H[:, i] *= X[:, j]**power[j]
    
    return H

def myprediction_gp(traininputs, trainoutputs, inputs):
    """GPU加速的高斯过程预测"""
    # 自动检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 转换数据格式
    if isinstance(traininputs, torch.Tensor):
        X_train = traininputs.to(device)
    else:
        X_train = mypowerseries_gpu(traininputs)
    
    if isinstance(inputs, torch.Tensor):
        X_pred = inputs.to(device)
    else:
        X_pred = mypowerseries_gpu(inputs)
    
    y_train = trainoutputs.to(device) if isinstance(trainoutputs, torch.Tensor) \
        else torch.tensor(trainoutputs, device=device, dtype=torch.float32)
    
    # 初始化模型
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPModel(X_train, y_train, likelihood).to(device)
    
    # 训练配置
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.1)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # 混合精度训练
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    for epoch in range(50):
        optimizer.zero_grad()
        with autocast():
            output = model(X_train)
            loss = -mll(output, y_train)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    # 预测模式
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        with autocast():
            pred_dist = likelihood(model(X_pred))
    
    # 清理显存
    torch.cuda.empty_cache()
    
    return pred_dist.mean.cpu()