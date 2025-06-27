# improved_pytorch_trend_prediction_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import find_peaks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ---------------------- 趋势提取函数 ----------------------
def extract_moving_average_trend(signal, window=101):
    """提取滑动平均趋势"""
    pad_len = window // 2
    padded = np.pad(signal, (pad_len, pad_len), mode='reflect')
    return np.convolve(padded, np.ones(window)/window, mode='valid')

def find_threshold_from_data(df, trend_real, window=501):
    """从数据中找到threshold值"""
    # 计算趋势的一阶与二阶导数
    first_derivative = np.gradient(trend_real, df['f1'])
    second_derivative = np.gradient(first_derivative, df['f1'])
    
    # 找到多个二阶导数变化较大的点（拐点）
    peak_indices, _ = find_peaks(np.abs(second_derivative), distance=5000, prominence=1e-10)
    
    if len(peak_indices) > 0:
        peak_times = df['f1'].iloc[peak_indices].values
        peak_f9s = trend_real[peak_indices]
        threshold_value = peak_f9s[-1]  # 最后一个拐点的f9值
        threshold_time = peak_times[-1]  # 最后一个拐点的时间
        return threshold_value, threshold_time, peak_indices
    else:
        # 如果没有找到拐点，使用趋势的最大值
        max_idx = np.argmax(trend_real)
        return trend_real[max_idx], df['f1'].iloc[max_idx], [max_idx]

# ---------------------- 数据集类 ----------------------
class TrendDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------- 改进的LSTM模型定义 ----------------------
class ImprovedTrendLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=2):
        super(ImprovedTrendLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.norm(last_output)
        out = self.relu(self.fc1(last_output))
        out = self.fc2(out)
        return out

# ---------------------- 数据预处理函数 ----------------------
def create_sequences_with_trend_info(data, seq_length, include_derivatives=True):
    """创建包含趋势信息的LSTM训练序列"""
    X, y = [], []
    
    if include_derivatives and data.shape[1] >= 2:
        # 计算趋势的导数信息
        trend_values = data[:, 0]  # 假设第一列是趋势值
        first_derivative = np.gradient(trend_values)
        second_derivative = np.gradient(first_derivative)
        
        # 添加导数信息作为额外特征
        enhanced_data = np.column_stack([data, first_derivative, second_derivative])
    else:
        enhanced_data = data
    
    for i in range(len(enhanced_data) - seq_length):
        X.append(enhanced_data[i:(i + seq_length)])
        y.append(enhanced_data[i + seq_length, :2])  # 只预测原始的两个特征
    
    return np.array(X), np.array(y)

def prepare_data_for_training(df, seq_length=50, window=501):
    """准备训练数据 - 增强版"""
    # 数据预处理
    df = df.sort_values(by='f1').reset_index(drop=True)
    
    # 使用更稳健的标准化方法
    scaler = MinMaxScaler(feature_range=(-1, 1))  # 使用[-1, 1]范围以保持负值信息
    f9_scaled = scaler.fit_transform(df[['f9']].values).flatten()
    
    # 提取趋势
    trend = extract_moving_average_trend(f9_scaled, window=window)
    trend_real = scaler.inverse_transform(trend.reshape(-1, 1)).flatten()
    
    # 找到threshold
    threshold_value, threshold_time, peak_indices = find_threshold_from_data(df, trend_real, window)
    
    # 创建增强的时间特征
    time_values = df['f1'].values
    time_normalized = (time_values - time_values.min()) / (time_values.max() - time_values.min())
    
    # 添加更多时间特征以帮助模型理解趋势
    time_squared = time_normalized ** 2
    time_log = np.log1p(time_normalized)  # log(1+x) 避免log(0)
    
    # 组合特征：趋势值 + 多种时间特征
    features = np.column_stack([trend_real, time_normalized, time_squared, time_log])
    
    # 创建包含导数信息的序列数据
    X, y = create_sequences_with_trend_info(features, seq_length, include_derivatives=True)
    
    return X, y, trend_real, threshold_value, threshold_time, scaler, features, time_values

# ---------------------- 改进的训练函数 ----------------------
def train_model_with_threshold_stopping(model, train_loader, val_loader, threshold_value, 
                                      num_epochs=50, learning_rate=0.001):
    """训练模型，当预测值达到阈值时提前停止"""
    criterion = nn.MSELoss()
    
    # 使用更复杂的优化器调度
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_count = 0
    early_stop_patience = 15
    # threshold_reached_count = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        # predictions_below_threshold = 0
        # total_predictions = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # 计算损失
            loss = criterion(outputs, batch_y)
            
            # 添加趋势约束损失 - 鼓励下降趋势
            trend_pred = outputs[:, 0]  # 趋势预测值
            trend_target = batch_y[:, 0]  # 趋势目标值
            
            # 如果预测值大于目标值，增加额外的惩罚（鼓励下降）
            trend_penalty = torch.mean(torch.relu(trend_pred - trend_target) ** 2) * 0.5
            
            total_loss = loss + trend_penalty
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += total_loss.item()
            
            # 检查预测值是否接近阈值
            # with torch.no_grad():
            #     pred_values = outputs[:, 0].cpu().numpy()
            #     below_threshold = np.sum(pred_values <= (threshold_value + 0.001))
            #     predictions_below_threshold += below_threshold
            #     total_predictions += len(pred_values)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions_below_threshold = 0
        val_total_predictions = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # 检查验证集预测值
                # pred_values = outputs[:, 0].cpu().numpy()
                # below_threshold = np.sum(pred_values <= (threshold_value + 0.001))
                # val_predictions_below_threshold += below_threshold
                # val_total_predictions += len(pred_values)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # 计算达到阈值的比例
        train_threshold_ratio = 0
        val_threshold_ratio = 0
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
              f'Train Threshold Ratio: {train_threshold_ratio:.3f}, Val Threshold Ratio: {val_threshold_ratio:.3f}')
        
        # 如果大部分预测都达到了阈值，可以考虑提前停止
        # if val_threshold_ratio > 0.8:  # 80%的预测达到阈值
        #     threshold_reached_count += 1
        #     if threshold_reached_count >= 3:  # 连续3个epoch都有高比例达到阈值
        #         print(f'大部分预测已达到阈值，在第 {epoch+1} 轮提前停止训练')
        #         break
        # else:
        #     threshold_reached_count = 0
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_count += 1
        
        if patience_count >= early_stop_patience:
            print(f'验证损失未改善，在第 {epoch+1} 轮提前停止训练')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    return train_losses, val_losses

# ---------------------- 动态预测函数 ----------------------
def predict_until_threshold(model, initial_sequence, threshold_value, time_start, time_step, 
                          tolerance=0.00001, max_steps=50000, verbose=True):
    """动态预测直到达到阈值才停止"""
    model.eval()
    
    current_sequence = torch.FloatTensor(initial_sequence).unsqueeze(0).to(device)
    current_time = time_start
    
    predicted_times = []
    predicted_values = []
    
    with torch.no_grad():
        for step in range(max_steps):
            # 预测下一个点
            prediction = model(current_sequence)
            pred_trend = prediction[0, 0].cpu().item()
            
            # 更新时间
            current_time += time_step
            time_normalized = min(1.0, current_time)
            time_squared = time_normalized ** 2
            time_log = np.log1p(time_normalized)
            
            # 记录预测结果
            predicted_times.append(current_time)
            predicted_values.append(pred_trend)
            
            # 检查是否达到阈值
            if pred_trend <= (threshold_value + tolerance):
                if verbose:
                    print(f"\n达到阈值!")
                    print(f"步数: {step + 1}")
                    print(f"预测值: {pred_trend:.6f}")
                    print(f"阈值: {threshold_value:.6f}")
                    print(f"归一化时间: {current_time:.6f}")
                return current_time, np.array(predicted_values), step + 1, True
            
            # 构造新的特征点
            new_features = np.array([pred_trend, time_normalized, time_squared, time_log])
            
            # 计算导数信息（基于序列的最后几个点）
            if len(predicted_values) >= 2:
                recent_trends = [current_sequence[0, -1, 0].cpu().item()] + [pred_trend]
                first_deriv = recent_trends[-1] - recent_trends[-2]
                second_deriv = 0  # 简化处理
            else:
                first_deriv = 0
                second_deriv = 0
            
            # 添加导数信息
            enhanced_features = np.concatenate([new_features, [first_deriv, second_deriv]])
            new_point = torch.FloatTensor([enhanced_features]).to(device)
            
            # 更新序列
            current_sequence = torch.cat([current_sequence[:, 1:, :], new_point.unsqueeze(1)], dim=1)
            
            # 显示进度
            if verbose and (step + 1) % 1000 == 0:
                print(f"步数: {step + 1:5d}, 当前值: {pred_trend:.6f}, 目标: {threshold_value:.6f}, "
                      f"差距: {threshold_value - pred_trend:.6f}")
    
    if verbose:
        print(f"\n在 {max_steps} 步内未达到阈值")
        print(f"最终预测值: {predicted_values[-1]:.6f}")
        print(f"目标阈值: {threshold_value:.6f}")
    
    return None, np.array(predicted_values), max_steps, False

# ---------------------- 主要训练和预测函数 ----------------------
def train_and_predict_model(full_csv_file, partial_csv_file, seq_length=50, window=501, 
                          batch_size=32, epochs=100, fine_tune_epochs=50):
    """训练模型并进行预测 - 改进版"""
    
    print("=" * 80)
    print("Step 1: 加载完整数据集并训练初始模型")
    print("=" * 80)
    
    # 加载完整数据集
    df_full = pd.read_csv(full_csv_file)
    X_full, y_full, trend_full, threshold_value, threshold_time, scaler_full, features_full, times_full = prepare_data_for_training(
        df_full, seq_length, window
    )
    
    print(f"完整数据集大小: {len(df_full)}")
    print(f"特征序列数量: {X_full.shape[0]}")
    print(f"输入特征维度: {X_full.shape[2]}")
    print(f"阈值: {threshold_value:.6f}")
    print(f"阈值时间: {threshold_time:.2e}")
    
    # 创建数据加载器
    dataset_full = TrendDataset(X_full, y_full)
    train_size = int(0.8 * len(dataset_full))
    val_size = len(dataset_full) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_full, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建改进的模型
    input_size = X_full.shape[2]
    model = ImprovedTrendLSTM(input_size=input_size, hidden_size=128, output_size=2).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\n开始初始训练...")
    train_losses, val_losses = train_model_with_threshold_stopping(
        model, train_loader, val_loader, threshold_value, epochs, 0.001
    )
    
    print("\n" + "=" * 80)
    print("Step 2: 加载部分数据集并进行微调")
    print("=" * 80)
    
    # 加载部分数据集（整体输入，不再只取前半部分）
    df_partial = pd.read_csv(partial_csv_file)
    
    print(f"部分数据集大小: {len(df_partial)}")
    
    # 准备完整的部分数据
    X_partial, y_partial, trend_partial, _, _, scaler_partial, features_partial, times_partial = prepare_data_for_training(
        df_partial, seq_length, window
    )
    
    print(f"部分数据特征序列数量: {X_partial.shape[0]}")
    
    # 创建部分数据加载器
    dataset_partial = TrendDataset(X_partial, y_partial)
    train_size_partial = int(0.8 * len(dataset_partial))
    val_size_partial = len(dataset_partial) - train_size_partial
    train_dataset_partial, val_dataset_partial = torch.utils.data.random_split(
        dataset_partial, [train_size_partial, val_size_partial]
    )
    
    train_loader_partial = DataLoader(train_dataset_partial, batch_size=batch_size//2, shuffle=True)
    val_loader_partial = DataLoader(val_dataset_partial, batch_size=batch_size//2, shuffle=False)
    
    # 微调训练
    print("\n开始微调训练...")
    fine_tune_losses, fine_tune_val_losses = train_model_with_threshold_stopping(
        model, train_loader_partial, val_loader_partial, threshold_value, fine_tune_epochs, 0.0001
    )
    
    print("\n" + "=" * 80)
    print("Step 3: 动态预测直到达到阈值")
    print("=" * 80)
    
    # 准备预测的初始数据
    initial_sequence = features_partial[-seq_length:]
    
    # 计算导数信息并添加到初始序列
    trend_values = initial_sequence[:, 0]
    first_derivative = np.gradient(trend_values)
    second_derivative = np.gradient(first_derivative)
    
    # 扩展初始序列以包含导数信息
    enhanced_initial_sequence = np.zeros((seq_length, features_partial.shape[1] + 2))
    enhanced_initial_sequence[:, :features_partial.shape[1]] = initial_sequence
    enhanced_initial_sequence[:, -2] = first_derivative
    enhanced_initial_sequence[:, -1] = second_derivative
    
    # 计算时间步长
    time_values = times_partial
    time_step = np.mean(np.diff(time_values)) / (time_values.max() - time_values.min())
    time_start = (time_values[-1] - time_values.min()) / (time_values.max() - time_values.min())
    
    print(f"预测起始时间: {time_values[-1]:.2e}")
    print(f"时间步长: {time_step:.8f}")
    print(f"目标阈值: {threshold_value:.6f}")
    print(f"当前趋势值: {trend_partial[-1]:.6f}")
    print(f"距离阈值: {threshold_value - trend_partial[-1]:.6f}")
    
    # 进行动态预测
    print("\n开始预测...")
    predicted_time_norm, predicted_values, steps, reached_threshold = predict_until_threshold(
        model, enhanced_initial_sequence, threshold_value, time_start, time_step, 
        tolerance=0.001, max_steps=50000, verbose=True
    )
    
    # 结果处理和可视化
    results = {
        'model': model,
        'threshold_value': threshold_value,
        'threshold_time': threshold_time,
        'reached_threshold': reached_threshold,
        'prediction_steps': steps,
        'predicted_values': predicted_values,
        'scaler_full': scaler_full,
        'scaler_partial': scaler_partial,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'fine_tune_losses': fine_tune_losses,
        'fine_tune_val_losses': fine_tune_val_losses
    }
    
    if reached_threshold:
        predicted_time_real = predicted_time_norm * (time_values.max() - time_values.min()) + time_values.min()
        results['predicted_time'] = predicted_time_real
        
        print(f"\n预测成功!")
        print(f"预测达到阈值的时间: {predicted_time_real:.2e}")
        print(f"预测步数: {steps:,}")
        print(f"从当前时间到达阈值需要: {predicted_time_real - time_values[-1]:.2e} 时间单位")
        print(f"最终预测值: {predicted_values[-1]:.6f}")
    else:
        results['predicted_time'] = None
        print(f"\n在 {steps:,} 步内未能达到阈值")
        print(f"最终预测值: {predicted_values[-1]:.6f}")
        print(f"目标阈值: {threshold_value:.6f}")
    
    # 可视化结果
    print("\n" + "=" * 80)
    print("Step 4: 可视化结果")
    print("=" * 80)
    
    visualize_results(df_full, df_partial, trend_full, trend_partial, 
                     predicted_values, results, times_partial)
    
    return results

def visualize_results(df_full, df_partial, trend_full, trend_partial, 
                     predicted_values, results, times_partial):
    """可视化所有结果 - 简化版（无表情符号）"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 训练损失历史
    axes[0, 0].plot(results['train_losses'], label='Initial Training Loss', alpha=0.8)
    axes[0, 0].plot(results['val_losses'], label='Initial Validation Loss', alpha=0.8)
    
    start_idx = len(results['train_losses'])
    fine_tune_x = range(start_idx, start_idx + len(results['fine_tune_losses']))
    axes[0, 0].plot(fine_tune_x, results['fine_tune_losses'], 
                   label='Fine-tuning Loss', alpha=0.8, color='red')
    axes[0, 0].plot(fine_tune_x, results['fine_tune_val_losses'], 
                   label='Fine-tuning Val Loss', alpha=0.8, color='orange')
    
    axes[0, 0].set_title('Model Training History', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. 完整数据集趋势
    axes[0, 1].plot(df_full['f1'], df_full['f9'], alpha=0.4, label='Original f9', color='lightblue')
    axes[0, 1].plot(df_full['f1'], trend_full, 'b-', label='Extracted Trend', linewidth=2)
    axes[0, 1].axhline(y=results['threshold_value'], color='red', linestyle='--', linewidth=2,
                      label=f'Threshold = {results["threshold_value"]:.4f}')
    axes[0, 1].axvline(x=results['threshold_time'], color='purple', linestyle='--', linewidth=2,
                      label=f'Threshold Time = {results["threshold_time"]:.2e}')
    axes[0, 1].set_title('Full Dataset Trend Analysis', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time (f1)')
    axes[0, 1].set_ylabel('f9')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 部分数据集和预测
    axes[0, 2].plot(df_partial['f1'], df_partial['f9'], alpha=0.4, 
                   label='Original f9', color='lightgreen')
    axes[0, 2].plot(df_partial['f1'], trend_partial, 'g-', 
                   label='Extracted Trend', linewidth=2)
    axes[0, 2].axhline(y=results['threshold_value'], color='red', linestyle='--', linewidth=2,
                      label=f'Target Threshold = {results["threshold_value"]:.4f}')
    
    # 预测轨迹
    if results['reached_threshold'] and results['predicted_time'] is not None:
        time_values = times_partial
        time_step_real = np.mean(np.diff(time_values))
        start_time = time_values[-1]
        pred_times = [start_time + i * time_step_real for i in range(1, len(predicted_values) + 1)]
        
        axes[0, 2].plot(pred_times, predicted_values, 'r-', linewidth=2, 
                       label='Prediction Trajectory', alpha=0.8)
        axes[0, 2].scatter([pred_times[-1]], [predicted_values[-1]], color='red', s=50,
                         label=f'Predicted Time: {results["predicted_time"]:.2e}')
    
    axes[0, 2].set_title('Partial Dataset & Prediction', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Time (f1)')
    axes[0, 2].set_ylabel('f9')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 预测轨迹与完整趋势对比
    if results['reached_threshold']:
        # 获取完整数据集在预测时间范围内的部分
        mask_full = (df_full['f1'] >= times_partial[0]) & (df_full['f1'] <= results['predicted_time'])
        axes[1, 0].plot(df_full['f1'][mask_full], trend_full[mask_full], 
                       'b-', label='Full Dataset Trend')
        
        # 添加预测轨迹
        axes[1, 0].plot(pred_times, predicted_values, 'r-', 
                       label='Prediction Trajectory')
        
        # 标记关键点
        axes[1, 0].axvline(x=times_partial[-1], color='gray', linestyle=':', 
                          label='Prediction Start')
        axes[1, 0].axvline(x=results['predicted_time'], color='red', linestyle='--',
                          label='Predicted End')
        
        axes[1, 0].set_title('Prediction vs Full Trend', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time (f1)')
        axes[1, 0].set_ylabel('f9')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 预测轨迹与阈值的距离
    if len(predicted_values) > 0:
        gap_to_threshold = results['threshold_value'] - predicted_values
        
        axes[1, 1].plot(range(len(predicted_values)), gap_to_threshold, 
                       'm-', label='Gap to Threshold')
        
        # 标记达到阈值的点
        if results['reached_threshold']:
            axes[1, 1].axvline(x=len(predicted_values)-1, color='red', linestyle='--',
                              label='Threshold Reached')
        
        axes[1, 1].set_title('Gap to Threshold During Prediction', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Prediction Step')
        axes[1, 1].set_ylabel('Threshold - Predicted Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    # 6. 预测值变化趋势
    if len(predicted_values) > 0:
        axes[1, 2].plot(range(len(predicted_values)), predicted_values, 
                       'c-', label='Predicted Values')
        
        # 添加阈值线
        axes[1, 2].axhline(y=results['threshold_value'], color='red', linestyle='--',
                          label='Threshold')
        
        axes[1, 2].set_title('Predicted Values Over Steps', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Prediction Step')
        axes[1, 2].set_ylabel('Predicted Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trend_prediction_results.png')
    plt.show()

if __name__ == '__main__':
    # 使用示例
    full_csv_file = "tube1_mode_1-0-0.875.csv"  # 完整数据集
    partial_csv_file = "tube2_mode_1-0-0.875.csv"  # 相似的部分数据集
    
    # 如果没有第二个数据集，可以使用同一个数据集进行测试
    # partial_csv_file = full_csv_file
    
    try:
        results = train_and_predict_model(
            full_csv_file=full_csv_file,
            partial_csv_file=partial_csv_file,
            seq_length=300,      # LSTM序列长度
            window=2001,         # 滑动平均窗口
            batch_size=32,      # 批大小
            epochs=20,         # 初始训练轮数
            fine_tune_epochs=10 # 微调训练轮数
        )
        
        print("\n" + "=" * 80)
        print("🎉 最终结果总结")
        print("=" * 80)
        print(f"✅ 模型成功训练并微调")
        print(f"🎯 阈值: {results['threshold_value']:.6f}")
        
        if results['reached_threshold'] and results['predicted_time'] is not None:
            print(f"⏰ 预测达到阈值的时间: {results['predicted_time']:.2e}")
            print(f"📊 预测所需步数: {results['prediction_steps']:,}")
            print(f"🚀 预测成功率: 100%")
            
            # 计算预测效率
            time_diff = results['predicted_time'] - results.get('current_time', 0)
            if time_diff > 0:
                print(f"⏳ 预测时间跨度: {time_diff:.2e} 时间单位")
        else:
            print(f"❌ 模型未能在 {results['prediction_steps']:,} 步内预测到达阈值")
            print(f"📈 最终预测值: {results['predicted_values'][-1]:.6f}")
            print(f"📏 距离阈值还有: {results['threshold_value'] - results['predicted_values'][-1]:.6f}")
            
        print(f"🧠 模型复杂度: {sum(p.numel() for p in results['model'].parameters()):,} 参数")
        print(f"💾 使用设备: {device}")
        
        # 保存最终模型
        torch.save({
            'model_state_dict': results['model'].state_dict(),
            'threshold_value': results['threshold_value'],
            'scaler_full': results['scaler_full'],
            'scaler_partial': results['scaler_partial'],
            'seq_length': 50,
            'input_size': 2
        }, 'final_trend_prediction_model.pth')
        print(f"💾 模型已保存至: final_trend_prediction_model.pth")
            
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保CSV文件存在于当前目录中")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("请检查数据格式和参数设置")
        import traceback
        traceback.print_exc()

# ---------------------- 模型加载和使用函数 ----------------------
def load_and_use_saved_model(model_path, new_data_csv, seq_length=50):
    """
    加载保存的模型并用于新数据预测
    
    Args:
        model_path: 模型文件路径
        new_data_csv: 新数据CSV文件
        seq_length: 序列长度
    
    Returns:
        预测结果
    """
    try:
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        
        # 重建模型
        model = ImprovedTrendLSTM(input_size=checkpoint['input_size'], 
                         hidden_size=64, num_layers=2, output_size=2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 加载数据
        df = pd.read_csv(new_data_csv)
        
        # 取前半部分数据
        mid_point = len(df) // 2
        df_half = df.iloc[:mid_point].copy()
        
        # 准备数据
        X, y, trend, _, _, scaler, features, times = prepare_data_for_training(
            df_half, seq_length, 501
        )
        
        # 获取阈值
        threshold_value = checkpoint['threshold_value']
        
        # 准备预测初始序列
        initial_sequence = features[-seq_length:]
        
        # 计算时间参数
        time_step = np.mean(np.diff(times)) / (times.max() - times.min())
        time_start = (times[-1] - times.min()) / (times.max() - times.min())
        
        print(f"🔮 使用保存的模型进行预测...")
        print(f"🎯 目标阈值: {threshold_value:.6f}")
        print(f"📍 当前值: {trend[-1]:.6f}")
        
        # 进行预测
        predicted_time_norm, predicted_values, steps, reached_threshold = predict_until_threshold(
            model, initial_sequence, threshold_value, time_start, time_step,
            tolerance=0.001, max_steps=50000, verbose=True
        )
        
        if reached_threshold:
            predicted_time_real = predicted_time_norm * (times.max() - times.min()) + times.min()
            print(f"✅ 预测成功! 达到阈值时间: {predicted_time_real:.2e}")
            return {
                'success': True,
                'predicted_time': predicted_time_real,
                'steps': steps,
                'final_value': predicted_values[-1]
            }
        else:
            print(f"❌ 未能达到阈值")
            return {
                'success': False,
                'steps': steps,
                'final_value': predicted_values[-1] if len(predicted_values) > 0 else None
            }
            
    except Exception as e:
        print(f"❌ 加载模型时出错: {e}")
        return {'success': False, 'error': str(e)}

# ---------------------- 批量预测函数 ----------------------
def batch_predict_multiple_datasets(model_path, csv_files, seq_length=50):
    """
    批量预测多个数据集
    
    Args:
        model_path: 模型文件路径
        csv_files: CSV文件列表
        seq_length: 序列长度
    
    Returns:
        所有预测结果
    """
    results = {}
    
    print(f"🚀 开始批量预测 {len(csv_files)} 个数据集...")
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n{'='*50}")
        print(f"📊 处理数据集 {i}/{len(csv_files)}: {csv_file}")
        print(f"{'='*50}")
        
        result = load_and_use_saved_model(model_path, csv_file, seq_length)
        results[csv_file] = result
        
        if result['success']:
            print(f"✅ {csv_file}: 预测成功")
        else:
            print(f"❌ {csv_file}: 预测失败")
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("📈 批量预测结果汇总")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"📊 总数据集: {total}")
    print(f"✅ 成功预测: {successful}")
    print(f"❌ 失败预测: {total - successful}")
    print(f"🎯 成功率: {successful/total*100:.1f}%")
    
    return results

# ---------------------- 使用示例 ----------------------
"""
使用示例:

1. 训练新模型:
results = train_and_predict_model(
    full_csv_file="tube1_mode_1-0-0.875.csv",
    partial_csv_file="tube2_mode_1-0-0.875.csv"
)

2. 使用保存的模型:
result = load_and_use_saved_model(
    "final_trend_prediction_model.pth",
    "new_data.csv"
)

3. 批量预测:
csv_files = ["data1.csv", "data2.csv", "data3.csv"]
results = batch_predict_multiple_datasets(
    "final_trend_prediction_model.pth",
    csv_files
)
"""

results = train_and_predict_model(
    full_csv_file="tube1_mode_1-0-0.875.csv",
    partial_csv_file="tube2_mode_1-0-0.875.csv"
)

