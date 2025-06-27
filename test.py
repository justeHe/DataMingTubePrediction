# wavelet_trend_extraction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

# ---------------------- 滑动平均提取趋势 ----------------------
def extract_moving_average_trend(signal, window=101):
    """
    使用反射填充的滑动平均提取趋势项
    Args:
        signal: 输入信号序列
        window: 滑动窗口大小 (必须为奇数)
    """
    pad_len = window // 2
    padded = np.pad(signal, (pad_len, pad_len), mode='reflect')
    return np.convolve(padded, np.ones(window)/window, mode='valid')


# ---------------------- 主流程 ----------------------
def extract_and_save_trend(csv_file, window_size=4001, output_suffix="_with_trend"):
    """
    提取趋势并保存到新的CSV文件
    Args:
        csv_file: 输入CSV文件路径
        window_size: 滑动窗口大小
        output_suffix: 输出文件名的后缀
    Returns:
        threshold_value: 最后一个拐点对应的阈值
        output_file: 输出文件路径
    """
    # 读取并处理数据
    df = pd.read_csv(csv_file).sort_values(by='f1')
    scaler = MinMaxScaler()
    f9_scaled = scaler.fit_transform(df[['f9']].values).flatten()
    
    # 提取趋势（滑动平均）
    trend = extract_moving_average_trend(f9_scaled, window=window_size)
    trend_real = scaler.inverse_transform(trend.reshape(-1, 1)).flatten()
    
    # 将趋势作为新列添加到DataFrame
    df['trend_f9'] = trend_real
    
    # 计算趋势的一阶与二阶导数
    first_derivative = np.gradient(trend_real, df['f1'])
    second_derivative = np.gradient(first_derivative, df['f1'])
    
    # 找到多个二阶导数变化较大的点（拐点）
    peak_indices, _ = find_peaks(np.abs(second_derivative), distance=5000, prominence=1e-10)
    peak_times = df['f1'].iloc[peak_indices].values
    peak_f9s = trend_real[peak_indices]
    
    # 设定阈值为最后一个拐点对应的趋势值
    threshold_value = None
    if len(peak_indices) > 0:
        threshold_value = peak_f9s[-1]
        print(f"Threshold value (last peak f9): {threshold_value:.4f} at time {peak_times[-1]:.2e}")
    
    # 生成输出文件名
    if '.' in csv_file:
        base_name, ext = csv_file.rsplit('.', 1)
        output_file = f"{base_name}{output_suffix}.{ext}"
    else:
        output_file = f"{csv_file}{output_suffix}.csv"
    
    # 保存到新的CSV文件
    df.to_csv(output_file, index=False)
    print(f"Saved trend data to: {output_file}")
    
    return threshold_value, output_file

def visualize_trend(csv_file, threshold_value=None):
    """
    可视化原始数据和趋势
    Args:
        csv_file: 包含趋势数据的CSV文件路径
        threshold_value: 阈值线值（可选）
    """
    df = pd.read_csv(csv_file)
    
    # 确保数据已排序
    df = df.sort_values('f1')
    
    # 计算导数用于找拐点
    trend_real = df['trend_f9'].values
    first_derivative = np.gradient(trend_real, df['f1'])
    second_derivative = np.gradient(first_derivative, df['f1'])
    
    # 找到拐点
    peak_indices, _ = find_peaks(np.abs(second_derivative), distance=5000, prominence=1e-10)
    peak_times = df['f1'].iloc[peak_indices].values
    peak_f9s = trend_real[peak_indices]
    
    # 可视化
    plt.figure(figsize=(12,6))
    plt.plot(df['f1'], df['f9'], label='Original f9', alpha=0.6)
    plt.plot(df['f1'], df['trend_f9'], label='Moving Average Trend', color='red')
    
    # 标记拐点
    for t, f in zip(peak_times, peak_f9s):
        plt.axvline(x=t, color='purple', linestyle='--', alpha=0.7)
        plt.text(t, f, f'{t:.1e}', rotation=90, verticalalignment='bottom', fontsize=8)
    
    # 添加阈值线
    if threshold_value is not None:
        plt.axhline(y=threshold_value, color='green', linestyle='--', 
                    label=f'Threshold f9 = {threshold_value:.4f}')
    
    plt.xlabel("Time (f1)")
    plt.ylabel("f9")
    plt.title("Moving Average Trend Extraction of f9")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 生成图像文件名
    if '.' in csv_file:
        base_name, ext = csv_file.rsplit('.', 1)
        image_file = f"{base_name}_trend_plot.png"
    else:
        image_file = f"{csv_file}_trend_plot.png"
    
    plt.savefig(image_file)
    print(f"Saved visualization to: {image_file}")
    plt.show()

# ---------------------- 执行入口 ----------------------
if __name__ == '__main__':
    input_file = "tube3_mode_1-0-0.875.csv"
    
    # 步骤1: 提取趋势并保存到新CSV
    threshold, output_file = extract_and_save_trend(
        input_file, 
        window_size=4001,
        output_suffix="_with_trend"
    )
    
    # 步骤2: 可视化结果
    visualize_trend(output_file, threshold_value=threshold)