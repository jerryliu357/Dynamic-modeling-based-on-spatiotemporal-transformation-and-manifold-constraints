import numpy as np
import networkx as nx
"""
def NWnetwork(N, m, p):
    #生成 Newman-Watts 小世界网络
    # 创建环形网络
    G = nx.watts_strogatz_graph(N, 2*m, 0)  # 初始无随机性
    matrix = nx.to_numpy_array(G)
    
    # 随机添加边 (Newman-Watts 模型)
    for i in range(N):
        possible_edges = [j for j in range(N) if j != i and matrix[i,j] == 0]
        k = np.random.binomial(len(possible_edges), p)
        chosen = np.random.choice(possible_edges, k, replace=False)
        matrix[i, chosen] = 1
    
    matrix = np.logical_or(matrix, matrix.T)  # 确保无向图对称
    np.fill_diagonal(matrix, False)           # 移除自环
    return matrix.astype(float), N
"""

def NWnetwork(N, m, p):
    # Initialize the matrix with integer type to allow addition
    matrix = np.zeros((N, N), dtype=int)
    
    # Middle part of the matrix (away from the boundaries)
    for i in range(m, N - m):
        matrix[i, (i - m):(i + m + 1)] = 1
    
    # Top m rows
    for i in range(m):
        matrix[i, 0:(i + 1 + m)] = 1  # i+1 is original i in 1-based
    
    # Bottom m rows
    for i in range(N - m, N):
        start_col = i - m
        matrix[i, start_col:N] = 1
    
    # Periodic boundary connections (wrap-around edges)
    for i in range(m):
        start = N - m + i
        matrix[i, start:N] = 1
        matrix[start:N, i] = 1  # Symmetric connection
    
    # Randomly add edges with probability p
    random_edges = (np.random.rand(N, N) < p).astype(int)
    matrix = (matrix + random_edges).astype(bool)
    
    # Remove self-loops by setting diagonal to False
    np.fill_diagonal(matrix, False)
    
    return matrix, N
"""
def NWnetdata(N, K, p):
    #生成含变点的网络时间序列数据
    stepsize = 0.01
    L = 4000
    change1 = 2500
    change2 = change1 + 200
    M = N  # 假设节点数等于变量数
    
    # 生成三个阶段的网络
    adjmat1, _ = NWnetwork(N, K, 0)
    adjmat2, _ = NWnetwork(N, K, p)
    adjmat3, _ = NWnetwork(N, K, p)
    
    # 初始化 Lorenz 系统
    x = np.random.rand(M, L)
    y = np.random.rand(M, L)
    z = np.random.rand(M, L)
    
    # 阶段1: 使用 adjmat1
    for i in range(change1 - 1):
        x[:, i+1] = x[:, i] + stepsize * (10*(y[:, i] - x[:, i]) + 0.1 * adjmat1 @ x[:, i])
        y[:, i+1] = y[:, i] + stepsize * (28*x[:, i] - y[:, i] - x[:, i]*z[:, i])
        z[:, i+1] = z[:, i] + stepsize * (-8/3 * z[:, i] + x[:, i]*y[:, i])
    
    # 阶段2: 使用 adjmat2 (参数变化)
    for i in range(change1 - 1, change2 - 1):
        x[:, i+1] = x[:, i] + stepsize * (10.2*(y[:, i] - x[:, i]) + 0.1 * adjmat2 @ x[:, i])
        y[:, i+1] = y[:, i] + stepsize * (28*x[:, i] - y[:, i] - x[:, i]*z[:, i])
        z[:, i+1] = z[:, i] + stepsize * (-8/3 * z[:, i] + x[:, i]*y[:, i])
    
    # 阶段3: 使用 adjmat3
    for i in range(change2 - 1, L - 1):
        x[:, i+1] = x[:, i] + stepsize * (10*(y[:, i] - x[:, i]) + 0.1 * adjmat3 @ x[:, i])
        y[:, i+1] = y[:, i] + stepsize * (28*x[:, i] - y[:, i] - x[:, i]*z[:, i])
        z[:, i+1] = z[:, i] + stepsize * (-8/3 * z[:, i] + x[:, i]*y[:, i])
    
    # 合并数据
    record = np.vstack([x, y, z]).reshape(3*M, L).T
    return adjmat1, adjmat2, adjmat3, change1, change2, record
"""
def NWnetdata(N, K, p):
    stepsize = 0.01
    L = 4000
    change1 = 2500
    change2 = change1 + 200
    M = N  # 从 NWnetwork 返回的 node 值
    
    # 生成三个网络拓扑
    adjmat1, _ = NWnetwork(N, K, 0)
    adjmat2, _ = NWnetwork(N, K, p)
    adjmat3, _ = NWnetwork(N, K, p)
    
    # 初始化状态矩阵 (MATLAB 的 zeros(M,L) 转换为 (M, L) 形状)
    x = np.zeros((M, L))
    y = np.zeros((M, L))
    z = np.zeros((M, L))
    x[:, 0] = np.random.rand(M)
    y[:, 0] = np.random.rand(M)
    z[:, 0] = np.random.rand(M)
    
    C = 0.1
    X = np.zeros((3 * M, L))
    
    # 第一阶段：使用 adjmat1 (i=0~change1-2)
    for i in range(change1 - 1):
        for j in range(M):
            # 计算耦合项 (MATLAB 的矩阵乘法转换为向量点积)
            coupling = C * np.dot(adjmat1[j, :].astype(float), x[:, i])
            x[j, i+1] = x[j, i] + stepsize * (10 * (y[j, i] - x[j, i]) + coupling)
            y[j, i+1] = y[j, i] + stepsize * (28 * x[j, i] - y[j, i] - x[j, i] * z[j, i])
            z[j, i+1] = z[j, i] + stepsize * (-8/3 * z[j, i] + x[j, i] * y[j, i])
    
    # 第二阶段：使用 adjmat2 (i=change1-1~change2-2)
    for i in range(change1-1, change2-1):
        for j in range(M):
            coupling = C * np.dot(adjmat2[j, :].astype(float), x[:, i])
            x[j, i+1] = x[j, i] + stepsize * (10.2 * (y[j, i] - x[j, i]) + coupling)
            y[j, i+1] = y[j, i] + stepsize * (28 * x[j, i] - y[j, i] - x[j, i] * z[j, i])
            z[j, i+1] = z[j, i] + stepsize * (-8/3 * z[j, i] + x[j, i] * y[j, i])
    
    # 第三阶段：使用 adjmat3 (i=change2-1~L-2)
    for i in range(change2-1, L-1):
        for j in range(M):
            coupling = C * np.dot(adjmat3[j, :].astype(float), x[:, i])
            x[j, i+1] = x[j, i] + stepsize * (10 * (y[j, i] - x[j, i]) + coupling)
            y[j, i+1] = y[j, i] + stepsize * (28 * x[j, i] - y[j, i] - x[j, i] * z[j, i])
            z[j, i+1] = z[j, i] + stepsize * (-8/3 * z[j, i] + x[j, i] * y[j, i])
    
    # 将数据合并到 X 中
    for j in range(M):
        X[3*j : 3*j+3, :] = np.array([x[j, :], y[j, :], z[j, :]])
    
    record = X.T  # 转置以匹配 MATLAB 的输出格式
    
    return adjmat1, adjmat2, adjmat3, change1, change2, record