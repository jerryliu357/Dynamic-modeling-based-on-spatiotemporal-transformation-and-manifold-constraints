import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os, argparse
from tqdm import tqdm
from generate_lorenz_pro import generate_lorenz_data

from imputation_model import ImputationODERGRU 
from imputation_utils import ImputationDataset, calculate_mape_r2, save_csv

# --- 模拟数据生成函数 (保持不变) ---
def generate_simulated_data(num_samples=1000, seq_len=50, num_features=6, missing_rate=0.2):
    # ... (与之前代码块中的定义相同)
    print("Generating simulated time series data...")
    # 1. 完整数据 (Y_true)
    time = np.linspace(0, 10, seq_len)
    base_signal = np.sin(time)
    
    Y_true = np.zeros((num_samples, seq_len, num_features))
    for i in range(num_samples):
        # 创建带有随机噪声和趋势的序列
        noise = np.random.normal(0, 0.1, (seq_len, num_features))
        trend = np.linspace(0, i / num_samples * 2, seq_len)[:, None]
        Y_true[i] = (base_signal[:, None] * (1 + np.random.rand(num_features) * 0.5) + trend + noise)

    # 2. 创建缺失值掩码 (Mask) 和输入数据 (X_raw)
    X_raw = Y_true.copy()
    Mask = np.ones((num_samples, seq_len, num_features))
    
    # 随机设置缺失点
    num_missing = int(num_samples * seq_len * num_features * missing_rate)
    missing_indices = np.random.choice(Y_true.size, num_missing, replace=False)
    
    X_raw.flat[missing_indices] = 0.0 # 模拟缺失值
    Mask.flat[missing_indices] = 0.0 
    
    # 数据集分割 (简单分割)
    train_size = int(0.7 * num_samples)
    
    X_train, X_test = X_raw[:train_size], X_raw[train_size:]
    M_train, M_test = Mask[:train_size], Mask[train_size:]
    Y_train, Y_test = Y_true[:train_size], Y_true[train_size:]

    print(f"Total samples: {num_samples}. Features: {num_features}. Seq_len: {seq_len}.")
    print(f"Missing rate: {missing_rate}. Train size: {train_size}. Test size: {num_samples - train_size}.")
    
    return X_train, M_train, Y_train, X_test, M_test, Y_test, num_features, seq_len
# ------------------------------------

# --- 训练和测试操作 (保持不变) ---
def train_op(model, device, ds, optimizer, criterion):
    model.train()
    train_loss, total_samples = 0, 0
    bar = tqdm(ds, desc="Training")
    
    for i, (x_raw, mask, y_true) in enumerate(bar):
        x_raw, mask, y_true = x_raw.to(device), mask.to(device), y_true.to(device)
        
        y_pred = model(x_raw, mask)
        
        loss_tensor = criterion(y_pred * (1 - mask), y_true * (1 - mask))
        
        num_missing_points = (1 - mask).sum()
        loss = loss_tensor.sum() / (num_missing_points + 1e-8)
        
        optimizer.zero_grad()
        loss.backward()
        
        # ========== 添加梯度裁剪 ==========
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # ==================================
        
        optimizer.step()

        train_loss += loss.item() * len(x_raw)
        total_samples += len(x_raw)
        
        bar.set_description('Train|Loss:{:.4f}'.format(train_loss / total_samples))

    return train_loss / total_samples

def test_op(model, device, ds, criterion, type='Test'):
    # ... (与之前代码块中的定义相同)
    with torch.no_grad():
        model.eval()
        test_loss, total_samples = 0, 0
        y_true_all, y_pred_all, mask_all = [], [], []
        
        bar = tqdm(ds, desc=type)
        for x_raw, mask, y_true in bar:
            x_raw, mask, y_true = x_raw.to(device), mask.to(device), y_true.to(device)
            
            y_pred = model(x_raw,mask)
            
            loss_tensor = criterion(y_pred * (1 - mask), y_true * (1 - mask))
            num_missing_points = (1 - mask).sum()
            loss = loss_tensor.sum() / (num_missing_points + 1e-8)
            
            test_loss += loss.item() * len(x_raw)
            total_samples += len(x_raw)
            
            y_true_all.append(y_true.cpu().numpy())
            y_pred_all.append(y_pred.cpu().numpy())
            mask_all.append(mask.cpu().numpy())

            bar.set_description('{}|Loss:{:.4f}'.format(type, test_loss / total_samples))

        y_true_agg = np.concatenate(y_true_all, axis=0)
        y_pred_agg = np.concatenate(y_pred_all, axis=0)
        mask_agg = np.concatenate(mask_all, axis=0)

        mape, r2 = calculate_mape_r2(y_true_agg, y_pred_agg, mask_agg)
        
    return test_loss / total_samples, mape, r2
# ------------------------------------

# --- Main 函数 (修改参数) ---
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. 数据加载 (从 .npz 文件读取)
    print("--- 正在从 .npz 文件加载数据 ---")
    save_dir = "./ADNI/lorenz_data"
    train_path = os.path.join(save_dir, f"lorenz_train_s{args.seq_len}.npz")
    test_path = os.path.join(save_dir, f"lorenz_test_s{args.seq_len}.npz")
    
    try:
        train_data = np.load(train_path)
        test_data = np.load(test_path)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件。请先运行 generate_and_save.py")
        print(f"尝试加载: {train_path} 和 {test_path}")
        exit()

    X_train, M_train, Y_train = train_data['X'], train_data['M'], train_data['Y']
    X_test, M_test, Y_test = test_data['X'], test_data['M'], test_data['Y']

    # ========== 添加这部分 ==========
    print("正在归一化数据...")
    # 只在观测值上计算统计量
    observed_values = Y_train[M_train == 1]
    mean = observed_values.mean()
    std = observed_values.std() + 1e-8

    # 归一化所有数据
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    Y_train = (Y_train - mean) / std
    Y_test = (Y_test - mean) / std

    print(f"归一化完成: mean={mean:.4f}, std={std:.4f}")
    print(f"归一化后范围: [{Y_train.min():.4f}, {Y_train.max():.4f}]")
    
    # 确保 num_features 和 seq_len 与加载的数据一致
    num_features = int(test_data['num_features'][0])
    seq_len = int(test_data['seq_len'][0])
    
    print(f"数据加载完毕。Features: {num_features}, Seq_len: {seq_len}")
    print(f"Train shapes: X={X_train.shape}, M={M_train.shape}, Y={Y_train.shape}")
    
    train_ds = ImputationDataset(X_train, M_train, Y_train)
    test_ds = ImputationDataset(X_test, M_test, Y_test)
    
    # 使用论文中 Imputation/Forecasting 的批次大小 32
    trDL = DataLoader(train_ds, batch_size=32, shuffle=True) 
    teDL = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 2. 模型设置 (使用论文指定参数)
    model = ImputationODERGRU(
        output_dim=num_features,
        n_layers=1, # 论文未明确指定 ODE layers，沿用默认或简单设置
        n_units=32, # 论文未明确指定 ODE units，沿用默认或简单设置
        latents=32, # SPD 矩阵维度设置为 32
        units=16,   # RGRU 隐藏单元维度设置为 16
        channel=num_features,
        device=device
    ).to(device)

    # 3. 损失函数和优化器 (使用论文指定参数)
    # 损失函数: 均方误差 (MSE)
    criterion = nn.MSELoss(reduction='none').to(device) 
    
    # 优化器: Adam, lr=10^-4, weight_decay=10^-3
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3) 

    cache = {'Model': 'ODE-RGRU-Imputation', 'Test_Loss': 1e9, 'MAPE': 1e9, 'R2': -1e9, 'Epoch': 0}
    
    print("\n--- Starting Imputation Training ---")

    for epoch in range(args.epochs):
        train_loss = train_op(model, device, trDL, optimizer, criterion)
        test_loss, mape, r2 = test_op(model, device, teDL, criterion, type='Test')

        if mape < cache['MAPE']:
            cache['Test_Loss'] = round(test_loss, 4)
            cache['MAPE'] = round(mape, 4)
            cache['R2'] = round(r2, 4)
            cache['Epoch'] = epoch
            torch.save(model.state_dict(), "best_lorenz_model.pth")
            
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Test Loss={test_loss:.4f} | MAPE={mape:.4f}, R2={r2:.4f}")
        print(f"BEST -> Epoch:{cache['Epoch']}, MAPE:{cache['MAPE']:.4f}, R2:{cache['R2']:.4f}")

    save_csv('Simulated_Imputation', cache)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Imputation Task')
    parser.add_argument('-g', '--gpu', default='0', help='GPU number')
    # 使用论文中 ADNI 迭代次数 400 作为默认 epochs
    parser.add_argument('--epochs', default=400, type=int, help='Number of epochs') 
    
    # SPD 矩阵维度设置为 32
    parser.add_argument('--latents', default=32, type=int) 
    # RGRU 隐藏单元维度设置为 16
    parser.add_argument('--units', default=16, type=int)   
    
    parser.add_argument('--seq_len', default=500, type=int)
    parser.add_argument('--num_features', default=6, type=int) # 模拟特征数
    parser.add_argument('--missing_rate', default=0.3, type=float) 
    parser.add_argument('--lr', default=1e-5, type=float) 
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--batch', default=32, type=int) # 论文指定批次大小 32
    args = parser.parse_args()

    main(args)