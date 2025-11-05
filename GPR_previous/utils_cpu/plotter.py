import matplotlib.pyplot as plt
import numpy as np

def plot_results(xx, result, j, trainlength, maxstep):
    """绘制结果图"""
    plt.figure(figsize=(10, 4))
    plt.plot(xx[j, :trainlength + maxstep], '-*', label='Real Data')
    c1 = np.arange(trainlength + 1, trainlength + maxstep + 1)
    c2 = result[0, :maxstep]
    plt.plot(c1, c2, 'ro', label='Predictions')
    plt.legend()
    plt.title("Real vs Predicted")
    plt.savefig('prediction_plot.png')
    
    plt.figure(figsize=(10, 4))
    plt.plot(result[1, :maxstep])
    plt.title("Prediction Standard Error")
    plt.savefig('stderr_plot.png')
    
    plt.figure(figsize=(10, 4))
    plt.plot(result[2, :maxstep])
    plt.title("Prediction Error")
    plt.savefig('error_plot.png')