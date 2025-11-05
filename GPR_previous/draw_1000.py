import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_predicted_trajectory(result):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    time = np.linspace(0, 1, 980)  # 标准化到[0,1]
    sc = ax.scatter(result[20:1000],result[10:990],result[0:980], c=time, cmap='plasma', s=0.5, alpha=0.8)
    # 设置颜色条
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label('Normalized Time', rotation=270, labelpad=15)
    
    # 设置坐标轴
    ax.set_xlabel('x(t)', labelpad=10)
    ax.set_ylabel('x(t-10)', labelpad=10)
    ax.set_zlabel('x(t-20)', labelpad=10)
    
    # 添加图例和调整视角
    ax.view_init(elev=25, azim=45)
    plt.title('Predicted Trajectory Comparison delay embedding')
    plt.tight_layout()
    plt.savefig('delay embedding 1000 10.png', dpi=1500)
    plt.show()

def plot_actual_vs_prediction(predicted,result,record):
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(actual.shape[0])
    plt.plot(time_steps, actual, 'r-', label='Actual Values', linewidth=1.5)
    plt.plot(time_steps, predicted.T, 'bo', label='Predicted Values', alpha=0.7)
    plt.fill_between(time_steps, 
                        result[0, :] - 100*result[1, :], 
                        result[0, :] + 100*result[1, :],
                        color='blue', alpha=0.3, label='Uncertainty (±100 std)')
    plt.axvline(x=2500-2400-1, color='g', linestyle=':', label='Change Point 1')
    plt.axvline(x=2700-2400-1, color='purple', linestyle=':', label='Change Point 2')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.title('Actual vs Predicted Values Over Time', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('time_series_prediction_1000.png', dpi=1500, bbox_inches='tight')
    plt.show()

result = np.load('detection_result_1000_x.npy')
resultT = result.T
predicted = resultT[:,0]
record = np.load('record2.npy')
plot_predicted_trajectory(predicted)
actual= record[2400:3400,0]
plot_actual_vs_prediction(actual,predicted,result)