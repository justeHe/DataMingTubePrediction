import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_and_preprocess_data(full_csv_file, partial_csv_file):
    """
    Loads and preprocesses the data, focusing on the f1 (time) and trend_f9 (feature) columns.
    """
    # Read data
    full_data = pd.read_csv(full_csv_file)
    partial_data = pd.read_csv(partial_csv_file)
    
    print(f"Original shape of the large dataset: {full_data.shape}")
    print(f"Original shape of the small dataset: {partial_data.shape}")
    
    # Check if required columns exist
    required_cols = ['f1', 'trend_f9']
    for col in required_cols:
        if col not in full_data.columns:
            raise ValueError(f"Column missing in the large dataset: {col}")
        if col not in partial_data.columns:
            raise ValueError(f"Column missing in the small dataset: {col}")
    
    # Keep only f1 and trend_f9 columns
    full_data = full_data[['f1', 'trend_f9']].copy()
    partial_data = partial_data[['f1', 'trend_f9']].copy()
    
    # Sort by f1 (time)
    full_data = full_data.sort_values('f1').reset_index(drop=True)
    partial_data = partial_data.sort_values('f1').reset_index(drop=True)
    
    print(f"Shape of the large dataset after processing: {full_data.shape}")
    print(f"Shape of the small dataset after processing: {partial_data.shape}")
    
    return full_data, partial_data

def create_mapping_features(full_data, window_size=1001):
    """
    Creates mapping features based on f1 and f9.
    """
    features = []
    feature_names = []
    
    time_data = full_data['f1'].values
    feature_data = full_data['trend_f9'].values
    
    features.append(feature_data.reshape(-1, 1))
    feature_names.append("f9_original")
    
    # Rolling mean of f9
    rolling_mean = pd.Series(feature_data).rolling(window=window_size, min_periods=1).mean().values
    features.append(rolling_mean.reshape(-1, 1))
    feature_names.append("f9_rolling_mean")
    
    # Trend of f9
    f9_trend = np.diff(feature_data, prepend=feature_data[0])
    features.append(f9_trend.reshape(-1, 1))
    feature_names.append("f9_trend")
    
    # Relative position feature
    relative_position = np.arange(len(full_data)) / (len(full_data) - 1)
    features.append(relative_position.reshape(-1, 1))
    feature_names.append("relative_position")
    
    # Combine all features
    X = np.hstack(features)
    
    return X, feature_names

# --- PyTorch Model Definition ---
class SimpleNN(nn.Module):
    """
    A simple Feed-Forward Neural Network (MLP) for regression.
    """
    def __init__(self, input_size, hidden_size=128, num_hidden_layers=3, dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        
        layers = []
        # Input Layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden Layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            
        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_and_predict_model(full_csv_file, partial_csv_file, test_ratio=0.3, save_plots=True):
    """
    Trains an f1 -> f9 mapping model and makes predictions.
    
    Args:
        full_csv_file (str): Path to the large dataset file.
        partial_csv_file (str): Path to the small dataset file.
        test_ratio (float): The proportion of the dataset to be used for prediction.
        save_plots (bool): Whether to save the plots.
        
    Returns:
        dict: A dictionary containing prediction results and evaluation metrics.
    """
    # Create a directory to save plots
    if save_plots:
        os.makedirs('prediction_plots_pytorch', exist_ok=True)
        
    # 1. Load data
    print("Loading data...")
    full_data, partial_data = load_and_preprocess_data(full_csv_file, partial_csv_file)
    
    # 2. Align data - use the size of the smaller dataset to determine the training size
    train_size = int(len(partial_data) * (1 - test_ratio))
    
    print(f"Total length of the small dataset: {len(partial_data)}")
    print(f"Total length of the large dataset: {len(full_data)}")
    print(f"Length of data for training: {train_size}")
    print(f"Remaining length of the small dataset: {len(partial_data) - train_size}")
    print(f"Remaining length of the large dataset: {len(full_data) - train_size}")
    
    # 3. Prepare training data
    full_train = full_data.iloc[:train_size].copy()
    partial_train = partial_data.iloc[:train_size].copy()
    
    # Create mapping features
    X_train, feature_names = create_mapping_features(full_train)
    y_train = partial_train['trend_f9'].values  # The target is the f9 feature from the small dataset
    
    print(f"Training feature dimensions: {X_train.shape}")
    print(f"Training target dimensions: {y_train.shape}")
    
    # 4. Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    
    # 5. Train the PyTorch model
    print("Training the PyTorch mapping model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create validation split to monitor training
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.15, random_state=42
    )

    # Convert data to PyTorch Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train_final), torch.FloatTensor(y_train_final))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model, loss, and optimizer
    model = SimpleNN(input_size=X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    num_epochs = 100
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        train_losses.append(np.mean(batch_losses))
        
        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_batch_losses.append(loss.item())
        
        val_losses.append(np.mean(val_batch_losses))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

    print("PyTorch model training completed.")
    
    # 6. Predict using the entire remaining part of the large dataset
    full_test = full_data.iloc[train_size:].copy()  # Use all remaining data
    X_test, _ = create_mapping_features(full_test)
    X_test_scaled = scaler_X.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    # Prediction
    print(f"Predicting using {len(full_test)} remaining data points from the large dataset...")
    model.eval()
    with torch.no_grad():
        predictions_scaled_tensor = model(X_test_tensor)
    
    predictions_scaled = predictions_scaled_tensor.cpu().numpy()
    predictions = scaler_y.inverse_transform(predictions_scaled).ravel()
    
    # 7. Evaluate predictions (only for the part where the small dataset has corresponding true values)
    actual_test = partial_data.iloc[train_size:]
    evaluation = None
    evaluation_length = min(len(actual_test), len(predictions))
    
    if evaluation_length > 0:
        actual_values = actual_test['trend_f9'].values[:evaluation_length]
        pred_values_for_eval = predictions[:evaluation_length]
        
        # Calculate evaluation metrics
        mse = mean_squared_error(actual_values, pred_values_for_eval)
        mae = mean_absolute_error(actual_values, pred_values_for_eval)
        r2 = r2_score(actual_values, pred_values_for_eval)
        
        evaluation = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'actual_values': actual_values,
            'evaluation_length': evaluation_length
        }
        
        print(f"\n=== Prediction Evaluation Results (based on {evaluation_length} comparison points) ===")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"R² Score: {r2:.6f}")
    
    # 8. Create a DataFrame for the prediction results
    prediction_df = pd.DataFrame({
        'f1': full_test['f1'].values,
        'trend_f9': predictions
    })
    
    # 9. Visualize the results
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Time Series Mapping Prediction Results (PyTorch Model)', fontsize=16, fontweight='bold')
    
    # Subplot 1: Complete time series comparison
    ax1 = axes[0, 0]
    train_time = partial_train['f1'].values
    train_values = partial_train['trend_f9'].values
    ax1.plot(train_time, train_values, 'b-', label='Training Data (Small Dataset)', linewidth=2, alpha=0.8)
    
    pred_time = full_test['f1'].values
    ax1.plot(pred_time, predictions, 'r-', label=f'Predicted Results ({len(predictions)} points)', linewidth=2)
    
    if evaluation_length > 0:
        test_time = actual_test['f1'].values[:evaluation_length]
        test_values = actual_test['trend_f9'].values[:evaluation_length]
        ax1.plot(test_time, test_values, 'g--', label=f'Actual Values ({evaluation_length} points)', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Time (f1)')
    ax1.set_ylabel('Feature Value (f9)')
    ax1.set_title('Complete Time Series Prediction Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Detailed comparison of the prediction part
    ax2 = axes[0, 1]
    if evaluation_length > 0:
        eval_pred_time = pred_time[:evaluation_length]
        eval_predictions = predictions[:evaluation_length]
        ax2.plot(eval_pred_time, eval_predictions, 'r-', label='Predicted Results (with Actuals)', linewidth=3)
        ax2.plot(test_time, test_values, 'g--', label='Actual Values', linewidth=2, alpha=0.8)
    
    if len(predictions) > evaluation_length:
        extend_pred_time = pred_time[evaluation_length:]
        extend_predictions = predictions[evaluation_length:]
        ax2.plot(extend_pred_time, extend_predictions, 'orange', linestyle='-', label=f'Extended Prediction ({len(extend_predictions)} points)', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Time (f1)')
    ax2.set_ylabel('Feature Value (f9)')
    ax2.set_title('Detailed Comparison of Prediction Part')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Training and Validation Loss
    ax3 = axes[1, 0]
    ax3.plot(range(num_epochs), train_losses, label='Training Loss')
    ax3.plot(range(num_epochs), val_losses, label='Validation Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('Model Training History')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Prediction error analysis
    ax4 = axes[1, 1]
    if evaluation_length > 0:
        eval_pred_values = predictions[:evaluation_length]
        eval_actual_values = actual_test['trend_f9'].values[:evaluation_length]
        errors = eval_actual_values - eval_pred_values
        eval_time = pred_time[:evaluation_length]
        
        ax4.plot(eval_time, errors, 'purple', linewidth=2, label='Prediction Error')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.fill_between(eval_time, errors, alpha=0.3, color='purple')
        ax4.set_xlabel('Time (f1)')
        ax4.set_ylabel('Error (Actual - Predicted)')
        ax4.set_title(f'Prediction Error Analysis ({evaluation_length} comparison points)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        error_text = f'Mean: {np.mean(errors):.4f}\nStd Dev: {np.std(errors):.4f}\nMax Error: {np.max(np.abs(errors)):.4f}'
        ax4.text(0.02, 0.98, error_text, transform=ax4.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No actual test data\nCannot calculate error', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Error Analysis')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_plots:
        plot_file = 'prediction_plots_pytorch/tube3_pytorch_prediction_results.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Main result plot saved to: {plot_file}")
    
    plt.show()
    
    # Additional plots and summary can be added here as needed
    
    # Return results
    results = {
        'predictions': prediction_df,
        'model': model,
        'scalers': {'X': scaler_X, 'y': scaler_y},
        'feature_names': feature_names,
        'training_info': {
            'train_size': train_size,
            'total_predictions': len(predictions),
            'evaluation_length': evaluation_length if evaluation_length else 0,
            'extended_predictions': len(predictions) - (evaluation_length if evaluation_length else 0),
            'n_features': X_train.shape[1],
            'full_data_shape': full_data.shape,
            'partial_data_shape': partial_data.shape,
            'model_type': 'PyTorch NN'
        },
        'training_loss': {'train': train_losses, 'validation': val_losses}
    }
    
    if evaluation:
        results['evaluation'] = evaluation
    
    print(f"\n=== Prediction Complete ===")
    print(f"Total f9 features predicted for {len(predictions)} time steps")
    if evaluation_length > 0:
        print(f"{evaluation_length} of these points had actual values for comparison")
        print(f"{len(predictions) - evaluation_length} additional time points were predicted")
    else:
        print(f"All {len(predictions)} points are extended predictions")
    print(f"Used {X_train.shape[1]} mapping features")
    print(f"Model type: PyTorch NN")
    if save_plots:
        print(f"All plots saved to the prediction_plots_pytorch/ directory")
    
    return results

def save_predictions(results, output_file="f9_predictions_pytorch.csv"):
    """
    Saves prediction results to a CSV file.
    """
    results['predictions'].to_csv(output_file, index=False)
    print(f"Prediction results saved to: {output_file}")

# Example usage
if __name__ == "__main__":
    # Main call
    results = train_and_predict_model(
        full_csv_file="tube1_mode_1-0-0.875_with_trend.csv",
        partial_csv_file="tube2_mode_1-0-0.875_with_trend.csv",
        save_plots=True  # Save plots
    )
    
    # Save prediction results
    save_predictions(results, "f9_predictions_pytorch.csv")
    
    # Print model training information
    print("\n=== Model Training Information ===")
    for key, value in results['training_info'].items():
        print(f"{key}: {value}")