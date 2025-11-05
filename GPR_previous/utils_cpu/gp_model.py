import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
#from itertools import product
"""
def mypower(n, degree):
    #生成所有阶数<=degree的多项式幂次组合
    powers = []
    for indices in product(range(degree + 1), repeat=n):
        if sum(indices) <= degree:
            powers.append(indices)
    return powers

def mypowerseries(X, powers):
    #生成多项式基函数矩阵 (兼容 GPU)
    X_gpu = np.asarray(X)  # 若用 GPU 可替换为 cupy
    n_samples, n_features = X_gpu.shape
    n_basis = len(powers)
    H = np.ones((n_samples, n_basis), dtype=X_gpu.dtype)
    
    for i, power in enumerate(powers):
        for j in range(n_features):
            H[:, i] *= X_gpu[:, j]** power[j]
    return H

def myprediction_gp(traininputs, trainoutputs, inputs):
    #高斯过程回归预测 (支持 GPU)
    n_features = traininputs.shape[1]
    powers = mypower(n_features, 2)
    
    # 转换为多项式基函数
    X_train = mypowerseries(traininputs, powers)
    X_input = mypowerseries(inputs.reshape(1, -1), powers)
    
    # 定义核函数 (优化后可调整)
    kernel = ConstantKernel() * RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
    
    gpr.fit(X_train, trainoutputs)
    return gpr.predict(X_input)[0]

"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def mypower_loop(n, m):
    """ 生成多项式幂次组合 """
    combinations = []
    
    # 0阶项
    if m > 0:
        combinations.append([0]*n)
    
    # 1阶项
    if m >= 1:
        for i in range(n):
            vec = [0]*n
            vec[i] = 1
            combinations.append(vec)
    
    # 2阶项
    if m >= 2:
        for i in range(n):
            for j in range(i, n):
                vec = [0]*n
                vec[i] += 1
                vec[j] += 1
                combinations.append(vec)
    
    return np.array(combinations)

def mypowerseries(X, p=2):
    """ 生成多项式特征矩阵 """
    n_samples, n_features = X.shape
    powers = mypower_loop(n_features, p)
    n_terms = powers.shape[0]
    
    H = np.ones((n_samples, n_terms))
    for i in range(n_terms):
        for j in range(n_features):
            H[:, i] *= np.power(X[:, j], powers[i, j])
    
    return H

def myprediction_gp(traininputs, trainoutputs, inputs):
    """ 高斯过程预测主函数 """
    # 生成多项式特征
    X_train = mypowerseries(traininputs, p=2)
    X_pred = mypowerseries(inputs, p=2)
    
    # 配置高斯过程核 (RBF + 白噪声)
    kernel = ConstantKernel(1.0) * RBF() + WhiteKernel()
    
    # 初始化并训练模型
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0,         # 对应MATLAB的exact方法
        normalize_y=False # 保持原始尺度
    )
    gp.fit(X_train, trainoutputs)
    
    # 进行预测
    pred_mean, _ = gp.predict(X_pred, return_std=True)
    
    return pred_mean