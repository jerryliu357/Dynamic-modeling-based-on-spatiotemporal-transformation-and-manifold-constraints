import numpy as np
import multiprocessing as mp
from functools import partial
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import itertools
import matplotlib.pyplot as plt
from gpr_module import GaussianProcessRegressor
from utills import outlieromit, save_results
from tqdm import tqdm

import numpy as np
import time



def NWnetwork(N, m, p):
    """小世界网络生成器 (Newman-Watts 模型)"""
    matrix = np.zeros((N, N), dtype=bool)
    
    # 生成环形规则网络
    for i in range(N):
        # 计算邻居索引（含环形处理）
        neighbors = [(i + k) % N for k in range(-m, m+1) if k != 0]
        matrix[i, neighbors] = True
    
    # 随机添加长程连接 (排除自环)
    rand_mask = np.random.rand(N, N) < p
    np.fill_diagonal(rand_mask, False)  # 移除自环
    matrix = matrix | rand_mask
    
    # 确保对称性（无向图）
    matrix = matrix | matrix.T
    np.fill_diagonal(matrix, False)  # 最终移除自环
    
    return matrix.astype(float), N

def generate_coupled_lorenz(N=5, L=4000, change1=2500, change2=2700, 
                           stepsize=0.01, C=0.1, m=1, p=0.1):
    """完全对齐MATLAB行为的耦合洛伦兹系统生成"""
    # 初始化网络和状态变量
    adjmat1, M = NWnetwork(N, m, 0)    # 阶段1：纯规则网络
    adjmat2, M = NWnetwork(N, m, p)    # 阶段2：添加随机连接
    adjmat3, M = NWnetwork(N, m, p)    # 阶段3：同阶段2
    
    # 初始化状态变量 (MATLAB风格列优先)
    x = np.zeros((M, L))
    y = np.zeros((M, L))
    z = np.zeros((M, L))
    x[:, 0] = np.random.rand(M)
    y[:, 0] = np.random.rand(M)
    z[:, 0] = np.random.rand(M)
    
    # 分阶段模拟
    for i in range(L-1):
        # 动态切换网络和参数
        if i < change1-1:
            adjmat = adjmat1
            sigma = 10.0
        elif i < change2-1:
            adjmat = adjmat2
            sigma = 10.2  # MATLAB第二阶段修改的参数
        else:
            adjmat = adjmat3
            sigma = 10.0
        
        for j in range(M):
            # 计算耦合项 (保持MATLAB的矩阵乘法顺序)
            coupling = C * np.dot(adjmat[j], x[:, i])
            
            # 更新方程 (严格对齐MATLAB实现)
            dx = stepsize * (sigma * (y[j,i] - x[j,i]) + coupling)
            dy = stepsize * (28 * x[j,i] - y[j,i] - x[j,i] * z[j,i])
            dz = stepsize * (-8/3 * z[j,i] + x[j,i] * y[j,i])
            
            x[j,i+1] = x[j,i] + dx
            y[j,i+1] = y[j,i] + dy
            z[j,i+1] = z[j,i] + dz
    
    # 构建与MATLAB完全相同的输出结构
    X = np.zeros((3*M, L))
    for j in range(M):
        X[3*j] = x[j]
        X[3*j+1] = y[j]
        X[3*j+2] = z[j]
    
    return X.T  # 保持(L, 3*M)的输出格式 (4000, 15)

# 保留原单节点洛伦兹函数（仅微调参数名称）没有用到
def generate_lorenz_data(n_steps=4000, sigma=10, rho=28, beta=8/3, dt=0.01):
    """生成单节点洛伦兹数据（保持原MATLAB参数顺序）"""
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    z = np.zeros(n_steps)
    x[0], y[0], z[0] = 0.1, 0.0, 0.0  # 硬编码初始条件
    
    for i in range(1, n_steps):
        dx = sigma * (y[i-1] - x[i-1]) * dt
        dy = (x[i-1] * (rho - z[i-1]) - y[i-1]) * dt
        dz = (x[i-1] * y[i-1] - beta * z[i-1]) * dt
        
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy
        z[i] = z[i-1] + dz
    
    return np.column_stack([x, y, z])

def _parallel_predict(comb, traindata, trainlength, target_idx):
    """并行预测函数"""
    try:
        # 输入数据准备
        trainX = traindata[list(comb), :trainlength-1].T  # (n_samples, n_features)
        trainy = traindata[target_idx, 1:trainlength]
        testX = traindata[list(comb), trainlength-1].reshape(1, -1)
        
        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        trainX_scaled = scaler_X.fit_transform(trainX)
        trainy_scaled = scaler_y.fit_transform(trainy.reshape(-1, 1)).flatten()
        testX_scaled = scaler_X.transform(testX)
        
        # 训练GPR模型
        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(trainX_scaled, trainy_scaled, init_params=(1.0, 0.1, 0.1), optimize=True)
        pred_scaled, std_scaled = gp.predict(testX_scaled, return_std=True)
        
        # 逆标准化
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return pred, std_scaled[0]
    except Exception as e:
        print(f"预测失败：{str(e)}")
        return np.nan, np.nan

def temporal_cp_detection(record, maxstep=400, L=4, s=600, j=0, n_jobs=4):
    """
    时序变化点检测主函数
    :param record: 输入时序数据 (n_samples, n_features)
    :param maxstep: 最大检测步数
    :param L: 嵌入维度
    :param s: 随机嵌入基数量
    :param j: 目标变量索引
    :param n_jobs: 并行进程数
    """
    # 数据预处理
    noise_strength = 1e-4
    X = record + noise_strength * np.random.randn(*record.shape)
    trainlength = 30  # 与MATLAB的trainlength=30对齐
    timelag = 2400 - trainlength - 1  # 对齐MATLAB的timelag计算
    xx = X[timelag+1:].T  # 从timelag+1开始切片，与MATLAB的xx=X(timelag+1:end,:)'一致

    # 结果存储矩阵 [预测值, 标准差, 残差]
    result = np.zeros((3, maxstep))
    
    # 创建4个并行进程的进程池
    pool = mp.Pool(processes=n_jobs)
    
    # 初始化进度条
    with tqdm(total=maxstep, desc="Processing Steps") as pbar:
        for step in range(maxstep):
            traindata = xx[:, step:step+trainlength]
            real_value = xx[j, trainlength + step] #第一个曲线的x坐标，j=0
            
            # 生成随机嵌入基组合
            D = traindata.shape[0] #15
            combs = list(itertools.combinations(range(D), L))
            np.random.shuffle(combs)
            selected_combs = combs[:s]
            
            # 并行预测
            predictions = pool.map(
                partial(_parallel_predict, 
                        traindata=traindata,
                        trainlength=trainlength,
                        target_idx=j),
                selected_combs
            )
            
            # 后处理
            pred_values = np.array([p[0] for p in predictions])
            pred_stds = np.array([p[1] for p in predictions]) 
            valid_mask = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
            valid_preds = pred_values[valid_mask]
            valid_stds = pred_stds[valid_mask]

            if len(valid_preds) == 0:
                final_pred = np.nan
                final_std = np.nan
            elif len(valid_preds) == 1:
                final_pred = valid_preds[0]
                final_std = 0.0
            else:
                try:
                    kde = gaussian_kde(valid_preds)
                    xi = np.linspace(valid_preds.min(), valid_preds.max(), 1000)
                    density = kde(xi)
                    final_pred = np.sum(xi * density) / np.sum(density)
                    final_std = np.std(valid_preds)
                except:  # 添加异常处理
                    final_pred = np.mean(valid_preds)
                    final_std = np.std(valid_preds)
            
            result[0, step] = final_pred
            result[1, step] = final_std
            result[2, step] = real_value - final_pred
            
            # 每10步或最后一步打印详细信息
            if (step % 10 == 0) or (step == maxstep - 1):
                pbar.write(f"Step {step+1}/{maxstep} | Residual: {result[2, step]:.4f}")
            
            # 更新进度条
            pbar.update(1)

    pool.close()
    return result

if __name__ == "__main__":
    # 生成多节点耦合洛伦兹数据（M=5个节点）
    M = 5  # 节点数
    st = time.time()
    record2 = generate_coupled_lorenz(N=M, L=4000, change1=2500, change2=2700)
    print(time.time()-st)
    save_results(record2, 'record2.npy')

    # 运行变化点检测（调整参数与MATLAB一致）
    result_1000_x = temporal_cp_detection(
        record2,
        maxstep=1000,    # 与MATLAB的maxstep=400对齐
        L=4,             # 嵌入维度调整为4
        s=600,           # 随机组合数增加至600
        j=0,             # 目标变量索引（例如第一个节点）
        n_jobs=100
    )
    
    

    # 保存结果与可视化（其余代码不变）
    save_results(result_1000_x, 'detection_result_1000_x.npy')
    

   