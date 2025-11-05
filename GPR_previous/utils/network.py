import cupy as cp
import numpy as np

def NWnetwork(N, m, p):
    """生成小世界网络（返回CuPy数组）"""
    matrix = cp.zeros((N, N), dtype=cp.int_)
    
    # Middle part
    for i in range(m, N - m):
        matrix[i, (i - m):(i + m + 1)] = 1
    
    # Top rows
    for i in range(m):
        matrix[i, 0:(i + m + 1)] = 1
    
    # Bottom rows
    for i in range(N - m, N):
        start_col = i - m
        matrix[i, start_col:N] = 1
    
    # Periodic boundaries
    for i in range(m):
        start = N - m + i
        matrix[i, start:N] = 1
        matrix[start:N, i] = 1  # 对称连接
    
    # 随机添加边
    random_edges = (cp.random.rand(N, N) < p).astype(cp.int_)
    matrix = (matrix + random_edges).astype(bool)
    cp.fill_diagonal(matrix, False)
    
    return matrix, N  # 直接返回CuPy数组

def NWnetdata(N, K, p):
    stepsize = 0.01
    L = 4000
    change1 = 2500
    change2 = change1 + 200
    M = N
    
    # 生成CuPy数组的网络
    adjmat1, _ = NWnetwork(N, K, 0)
    adjmat2, _ = NWnetwork(N, K, p)
    adjmat3, _ = NWnetwork(N, K, p)
    
    # 初始化状态矩阵（CuPy数组）
    x = cp.zeros((M, L))
    y = cp.zeros((M, L))
    z = cp.zeros((M, L))
    x[:, 0] = cp.random.rand(M)
    y[:, 0] = cp.random.rand(M)
    z[:, 0] = cp.random.rand(M)
    
    C = 0.1
    X = cp.zeros((3 * M, L))
    
    # 第一阶段计算（全部使用CuPy操作）
    for i in range(change1 - 1):
        coupling = C * cp.dot(adjmat1, x[:, i])
        x[:, i+1] = x[:, i] + stepsize * (10 * (y[:, i] - x[:, i]) + coupling)
        y[:, i+1] = y[:, i] + stepsize * (28 * x[:, i] - y[:, i] - x[:, i] * z[:, i])
        z[:, i+1] = z[:, i] + stepsize * (-8/3 * z[:, i] + x[:, i] * y[:, i])
    
    # 第二阶段
    for i in range(change1-1, change2-1):
        coupling = C * cp.dot(adjmat2, x[:, i])
        x[:, i+1] = x[:, i] + stepsize * (10.2 * (y[:, i] - x[:, i]) + coupling)
        y[:, i+1] = y[:, i] + stepsize * (28 * x[:, i] - y[:, i] - x[:, i] * z[:, i])
        z[:, i+1] = z[:, i] + stepsize * (-8/3 * z[:, i] + x[:, i] * y[:, i])
    
    # 第三阶段
    for i in range(change2-1, L-1):
        coupling = C * cp.dot(adjmat3, x[:, i])
        x[:, i+1] = x[:, i] + stepsize * (10 * (y[:, i] - x[:, i]) + coupling)
        y[:, i+1] = y[:, i] + stepsize * (28 * x[:, i] - y[:, i] - x[:, i] * z[:, i])
        z[:, i+1] = z[:, i] + stepsize * (-8/3 * z[:, i] + x[:, i] * y[:, i])
    
    # 转换为NumPy数组返回
    return (
        cp.asnumpy(adjmat1),
        cp.asnumpy(adjmat2),
        cp.asnumpy(adjmat3),
        change1,
        change2,
        cp.asnumpy(X.T)
    )