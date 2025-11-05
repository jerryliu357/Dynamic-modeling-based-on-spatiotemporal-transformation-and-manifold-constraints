import numpy as np

def outlieromit(predictions):
    """基于 IQR 去除异常值"""
    q25, q75 = np.percentile(predictions, [25, 75])
    iqr = q75 - q25
    lbound = q25 - 1.5 * iqr
    ubound = q75 + 1.5 * iqr
    return predictions[(predictions >= lbound) & (predictions <= ubound)]