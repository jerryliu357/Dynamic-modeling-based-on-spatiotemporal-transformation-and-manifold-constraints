import numpy as np

def normalize_data(X):
    """数据标准化"""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma, mu, sigma

def save_results(result, filename='result.npy'):
    """保存结果矩阵"""
    np.save(filename, result)

def outlieromit(predictions, iqr_factor=3.0):
    """基于IQR的异常值剔除"""
    q1 = np.percentile(predictions, 25)
    q3 = np.percentile(predictions, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    filtered = predictions[(predictions >= lower_bound) & (predictions <= upper_bound)]
    return filtered