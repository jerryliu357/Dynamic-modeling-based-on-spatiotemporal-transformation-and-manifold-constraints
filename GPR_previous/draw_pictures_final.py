import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# read file
result_x = np.load('detection_result_x.npy')
result_y = np.load('detection_result_y.npy')
result_z = np.load('detection_result_z.npy')
record = np.load('record.npy')
# Residuals
plt.figure(figsize=(12, 6))
plt.plot(result_x[2], 'r-', label='Residuals_x')
plt.plot(result_y[2], 'g-', label='Residuals_y')
plt.plot(result_z[2], 'b-', label='Residuals_z')
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Time Step')
plt.ylabel('Prediction Error')
plt.legend()
plt.savefig('Residuals.png', dpi=300)
plt.show()
# Predictions xyz
actual_values_x = record[2400:2400 + 400, 0]  
predicted_values_x = result_x[0, :]
actual_values_y = record[2400:2400 + 400, 1] 
predicted_values_y = result_y[0, :]
actual_values_z = record[2400:2400 + 400, 2]  
predicted_values_z = result_z[0, :]
# nan (optional)
"""
valid_mask_x = ~np.isnan(predicted_values_x) & ~np.isnan(actual_values_x)  
actual_valid_x = actual_values_x[valid_mask_x]
predicted_valid_x = predicted_values_x[valid_mask_x]
print("Number of NaN in predictions:", np.isnan(result_x[0, :]).sum())"
"""
# Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(actual_values_x, predicted_values_x.T, c='b', alpha=0.5, label='Predictions vs Actual of X')
plt.plot([actual_values_x.min(), actual_values_x.max()], 
             [actual_values_x.min(), actual_values_x.max()], 
             'r--', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.savefig('Prediction_scatter_X.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(actual_values_y, predicted_values_y, c='b', alpha=0.5, label='Predictions vs Actual of Y')
plt.plot([actual_values_y.min(), actual_values_y.max()], 
             [actual_values_y.min(), actual_values_y.max()], 
             'r--', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.savefig('Prediction_scatter_Y.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(actual_values_z, predicted_values_z, c='b', alpha=0.5, label='Predictions vs Actual of Z')
plt.plot([actual_values_z.min(), actual_values_z.max()], 
             [actual_values_z.min(), actual_values_z.max()], 
             'r--', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.savefig('Prediction_scatter_Z.png', dpi=300)
plt.show()

# Predictions vs Actual with timestep (2D)      
plt.figure(figsize=(12, 6))
time_steps = np.arange(actual_values_x.shape[0]) 
plt.plot(time_steps, actual_values_x, 'r-', label='Actual Values X', linewidth=1.5)
plt.plot(time_steps, predicted_values_x.T, 'bo', label='Predicted Values X', alpha=0.7)
plt.fill_between(time_steps, 
                    result_x[0, :] - 100*result_x[1, :], 
                    result_x[0, :] + 100*result_x[1, :],
                    color='blue', alpha=0.3, label='Uncertainty (±100 std)')
plt.axvline(x=2500-2400-1, color='g', linestyle=':', label='Change Point 1')
plt.axvline(x=2700-2400-1, color='purple', linestyle=':', label='Change Point 2')
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Values X', fontsize=12)
plt.title('Actual vs Predicted Values Over Time', fontsize=14)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('time_series_prediction_x.png', dpi=1500, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 6))
time_steps = np.arange(actual_values_y.shape[0]) 
plt.plot(time_steps, actual_values_y, 'r-', label='Actual Values Y', linewidth=1.5)
plt.plot(time_steps, predicted_values_y.T, 'bo', label='Predicted Values Y', alpha=0.7)
plt.fill_between(time_steps, 
                    result_y[0, :] - 100*result_y[1, :], 
                    result_y[0, :] + 100*result_y[1, :],
                    color='blue', alpha=0.3, label='Uncertainty (±100 std)')
plt.axvline(x=2500-2400-1, color='g', linestyle=':', label='Change Point 1')
plt.axvline(x=2700-2400-1, color='purple', linestyle=':', label='Change Point 2')
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Values Y', fontsize=12)
plt.title('Actual vs Predicted Values Over Time', fontsize=14)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('time_series_prediction_y.png', dpi=1500, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 6))
time_steps = np.arange(actual_values_z.shape[0]) 
plt.plot(time_steps, actual_values_z, 'r-', label='Actual Values Z', linewidth=1.5)
plt.plot(time_steps, predicted_values_z.T, 'bo', label='Predicted Values Z', alpha=0.7)
plt.fill_between(time_steps, 
                    result_z[0, :] - 100*result_z[1, :], 
                    result_z[0, :] + 100*result_z[1, :],
                    color='blue', alpha=0.5, label='Uncertainty (±100 std)')
plt.axvline(x=2500-2400-1, color='g', linestyle=':', label='Change Point 1')
plt.axvline(x=2700-2400-1, color='purple', linestyle=':', label='Change Point 2')
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Values Z', fontsize=12)
plt.title('Actual vs Predicted Values Over Time', fontsize=14)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('time_series_prediction_z.png', dpi=1500, bbox_inches='tight')
plt.show()

# 3D visualization of trace
def plot_true_trajectory(record):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取第一个点的坐标 (前2800时间步)
    t_end = 2800
    x = record[:t_end, 0]  # 第1列为x
    y = record[:t_end, 1]  # 第2列为y
    z = record[:t_end, 2]  # 第3列为z
    
    # 创建时间颜色映射
    time = np.linspace(0, 1, t_end)  # 标准化到[0,1]
    sc = ax.scatter(x, y, z, c=time, cmap='plasma', s=0.5, alpha=0.8)
    
    # 设置颜色条
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label('Normalized Time', rotation=270, labelpad=15)
    
    # 设置坐标轴
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)
    
    # 调整视角
    ax.view_init(elev=25, azim=45)
    plt.title('True Lorenz System Trajectory (0-2800 steps)')
    plt.tight_layout()
    plt.savefig('True_Lorenz_System_Trajectory.png', dpi=1500)
    plt.show()

# 生成第二个图的函数
def plot_predicted_trajectory(record, pred_x, pred_y, pred_z):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取真实轨迹 (前2400时间步)
    x_real = record[:2400, 0]
    y_real = record[:2400, 1]
    z_real = record[:2400, 2]
    
    # 合并真实和预测轨迹
    x_total = np.concatenate([x_real, pred_x.flatten()])
    y_total = np.concatenate([y_real, pred_y.flatten()])
    z_total = np.concatenate([z_real, pred_z.flatten()])
    
    # 绘制真实部分
    ax.plot(x_real, y_real, z_real, 
            color='lightblue', linewidth=0.5, alpha=0.8,
            label='True Trajectory (0-2400 steps)')
    
    # 绘制预测部分
    ax.plot(x_total[2400:], y_total[2400:], z_total[2400:],
           color='red', linewidth=1.5, alpha=0.9,
           label='Predicted Trajectory (2400-2800 steps)')
    
    # 设置坐标轴
    ax.set_xlabel('X', labelpad=10)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('Z', labelpad=10)
    
    # 添加图例和调整视角
    ax.legend(loc='upper right', fontsize=8)
    ax.view_init(elev=25, azim=45)
    plt.title('True vs Predicted Trajectory Comparison')
    plt.tight_layout()
    plt.savefig('True_vs_Predicted_Trajectory.png', dpi=1500)
    plt.show()

plot_true_trajectory(record)
plot_predicted_trajectory(record, 
                         predicted_values_x,
                         predicted_values_y,
                         predicted_values_z)

