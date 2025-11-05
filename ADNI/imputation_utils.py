import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import r2_score
import csv

class ImputationDataset(Dataset):
    """
    用于 Imputation 任务的模拟数据集类。
    输入：原始序列 (raw)，带有缺失值 (missing_mask)，目标序列 (target)
    """
    def __init__(self, raw_data, mask_data, target_data):
        self.raw = raw_data  # 包含缺失值的输入 (例如，NaN或0)
        self.mask = mask_data  # 缺失值的掩码 (0: 缺失, 1: 观测)
        self.target = target_data # 完整的真值序列 (用于损失计算)

    def __getitem__(self, index):
        # 注意：这里我们返回序列本身，Imputation通常是序列到序列的任务
        x = self.raw[index].astype('float32')
        mask = self.mask[index].astype('float32')
        y_true = self.target[index].astype('float32')
        return x, mask, y_true

    def __len__(self):
        return len(self.raw)

def calculate_mape_r2(y_true, y_pred, mask):
    """
    计算 Imputation 任务的 Mean Absolute Percentage Error (MAPE) 和 R^2。
    仅在缺失的位置（mask=0）评估性能。
    """
    # 展平数据以匹配 sklearn 格式
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mask_flat = mask.flatten()

    # 只保留需要插补的点（即 mask 上的缺失点，通常 mask=0 或 NaN）
    # 假设 mask=0 表示缺失值，即需要插补
    imputation_points_indices = np.where(mask_flat == 0)[0]

    y_true_imp = y_true_flat[imputation_points_indices]
    y_pred_imp = y_pred_flat[imputation_points_indices]

    if len(y_true_imp) == 0:
        return 0.0, 0.0

    # 1. MAPE (Mean Absolute Percentage Error)
    # 避免除以零，并增加一个小的 epsilon
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true_imp - y_pred_imp) / (y_true_imp + epsilon))) * 100

    # 2. R^2 (Coefficient of Determination)
    # R^2 值越高越好，MAPE 值越低越好。
    r2 = r2_score(y_true_imp, y_pred_imp)

    return mape, r2

def save_csv(filename, cache):
    """保存结果到 CSV 文件，与现有代码逻辑相同。"""
    with open("./{}_results.csv".format(filename), 'a') as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(cache.keys())
        w.writerow(cache.values())