# generate_and_save.py
import numpy as np
import argparse
from generate_lorenz_pro import generate_lorenz_data
import os

def main(args):
    print(f"--- 正在生成数据 (Samples={args.num_samples}, SeqLen={args.seq_len}) ---")
    
    # 1. 调用您的数据生成函数
    X_train, M_train, Y_train, X_test, M_test, Y_test, num_features, seq_len = generate_lorenz_data(
        num_samples=args.num_samples, 
        seq_len=args.seq_len, 
        num_features=args.num_features,
        missing_rate=args.missing_rate,
        noise_strength=args.noise_strength
    )
    
    # 2. 定义保存路径
    save_dir = "./ADNI/lorenz_data"
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, f"lorenz_train_s{args.seq_len}.npz")
    test_path = os.path.join(save_dir, f"lorenz_test_s{args.seq_len}.npz")

    # 3. 保存训练数据
    print(f"正在保存训练数据到: {train_path}")
    np.savez_compressed(
        train_path,
        X=X_train,
        M=M_train,
        Y=Y_train
    )
    
    # 4. 保存测试数据
    print(f"正在保存测试数据到: {test_path}")
    np.savez_compressed(
        test_path,
        X=X_test,
        M=M_test,
        Y=Y_test,
        num_features=np.array([num_features]), # 顺便保存一下
        seq_len=np.array([seq_len])      # 顺便保存一下
    )
    
    print("--- 数据生成并保存完毕 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Generate Lorenz Data')
    
    # 您可以根据需要调整这里的默认值
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--seq_len', default=500, type=int)
    parser.add_argument('--num_features', default=6, type=int)
    parser.add_argument('--missing_rate', default=0.3, type=float)
    parser.add_argument('--noise_strength', default=0.1, type=float)
    
    args = parser.parse_args()
    main(args)