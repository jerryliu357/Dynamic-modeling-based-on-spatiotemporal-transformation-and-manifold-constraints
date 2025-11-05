import numpy as np

def outlieromit(predictions):
    """处理可能的GPU张量输入"""
    if isinstance(predictions, np.ndarray):
        data = predictions
    else:
        data = predictions.cpu().numpy() if hasattr(predictions, 'cpu') else predictions
    
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bounds = (q25 - 1.5*iqr, q75 + 1.5*iqr)
    return data[(data >= bounds[0]) & (data <= bounds[1])]