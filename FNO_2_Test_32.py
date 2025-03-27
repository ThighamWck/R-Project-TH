# pde_operator_testing_higher_res.py

# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import model definition from training script - ensure this matches your FNO_1 or updated implementation
from FNO_2 import FNO2D

# ===============================
# DEVICE SETUP (MPS or CPU)
# ===============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# CUSTOM DATASET FOR HIGHER RESOLUTION
# ===============================
class PDEOperatorDatasetHighRes(Dataset):
    def __init__(self, grouped_data, grid_size=32):
        self.grouped_data = grouped_data
        self.grid_size = grid_size

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        # Get the group (PDE solution) based on index
        group_df = self.grouped_data.get_group(idx)

        # Extract input features (x, y, b1, b2, finv) and target (rho)
        x = torch.tensor(group_df[['x', 'y', 'b1', 'b2', 'finv']].values, dtype=torch.float32)
        
        # Extract 'rho' as the target variable
        if 'rho' in group_df.columns:
            target = torch.tensor(group_df['rho'].values, dtype=torch.float32)
        else:
            raise ValueError("'rho' column not found in group_df")

        # Reshape to grid format (32x32)
        x = x.view(self.grid_size, self.grid_size, -1).permute(2, 0, 1)  # [5, 32, 32]
        target = target.view(self.grid_size, self.grid_size)

        return x, target

# ===============================
# LOAD HIGHER RESOLUTION TEST DATA
# ===============================
def load_test_data_high_res(csv_path, grid_size=32, batch_size=16):
    # Load CSV
    data = pd.read_csv(csv_path)
    
    # Group by solution_id
    grouped_data = data.groupby('solution_id')
    
    # Create dataset and dataloader for 32x32 data
    test_dataset = PDEOperatorDatasetHighRes(grouped_data, grid_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_dataloader

# ===============================
# INFERENCE FUNCTION FOR HIGHER RESOLUTION
# ===============================
def model_inference_high_res(model, x_high_res):
    """
    Process higher resolution input with a model trained on lower resolution.
    Handles 32×32 input for a model trained on 16×16.
    """
    batch_size, channels, height, width = x_high_res.shape
    
    # Option 1: Downsample input to 16×16, process, then upsample result to 32×32
    x_down = F.interpolate(x_high_res, size=(16, 16), mode='bilinear', align_corners=False)
    y_down = model(x_down)
    y_up = F.interpolate(y_down, size=(height, width), mode='bilinear', align_corners=False)
    return y_up

# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate_model_high_res(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            
            # Forward pass with resolution handling
            outputs = model_inference_high_res(model, x)
            loss = criterion(outputs, y)
            test_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_preds.append(outputs.squeeze(1).cpu().numpy())
            all_targets.append(y.squeeze(1).cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    all_targets = np.concatenate([t.reshape(-1) for t in all_targets])
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    avg_loss = test_loss / len(dataloader)
    
    metrics = {
        'test_loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics, all_preds, all_targets

# ===============================
# VISUALIZATION FUNCTIONS
# ===============================
def plot_high_res_comparison(model, dataloader, num_samples=3):
    """Visualize predictions vs ground truth for high resolution samples"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6*num_samples))
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            x, y = x.to(device), y.to(device)
            output = model_inference_high_res(model, x)
            
            # Get the first sample from batch
            input_grid = x[0, :, :, :].cpu().numpy()
            true_field = y[0, :, :].cpu().numpy()
            pred_field = output[0, 0, :, :].cpu().numpy()
            
            # Plot input (b1 parameter as an example)
            im1 = axes[i, 0].imshow(input_grid[2, :, :])  # Showing b1 parameter
            axes[i, 0].set_title(f"Input (b1 parameter) - 32×32 Resolution")
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Plot ground truth
            im2 = axes[i, 1].imshow(true_field)
            axes[i, 1].set_title(f"Ground Truth (32×32)")
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Plot prediction
            im3 = axes[i, 2].imshow(pred_field)
            axes[i, 2].set_title(f"Prediction (32×32)")
            plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig('high_res_prediction_comparison.png')
    plt.close()

def plot_error_distribution(predictions, targets):
    """Plot the distribution of prediction errors"""
    errors = predictions - targets
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Prediction Errors (32×32 Resolution)')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('high_res_error_distribution.png')
    plt.close()
    
    # Also plot scatter of predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.1)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.title('Predicted vs Actual Values (32×32 Resolution)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    plt.savefig('high_res_pred_vs_actual.png')
    plt.close()

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    # Path to your test CSV file
    test_csv_path = "test_data_32_not_square.csv"  # Higher resolution test data
    model_path = "model_16_not_square.pth"  # Your existing 16×16 model
    
    # Load higher resolution test data (32×32)
    test_dataloader = load_test_data_high_res(test_csv_path, grid_size=32)
    
    # Initialize and load model (keep original architecture for 16×16)
    model = FNO2D(in_channels=5, out_channels=1, modes1=16, modes2=16, width=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate model on higher resolution data
    metrics, all_preds, all_targets = evaluate_model_high_res(model, test_dataloader)
    
    # Print metrics
    print("\n===== Model Evaluation Results (32×32 Resolution) =====")
    print(f"Test Loss: {metrics['test_loss']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R² Score: {metrics['r2']:.6f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_high_res_comparison(model, test_dataloader)
    plot_error_distribution(all_preds, all_targets)
    print("Visualizations saved.")
    
    # Save predictions to CSV
    print("\nSaving detailed results to CSV...")
    results_df = pd.DataFrame({
        'actual': all_targets,
        'predicted': all_preds,
        'error': all_preds - all_targets
    })
    results_df.to_csv('high_res_model_predictions.csv', index=False)
    print("Results saved to high_res_model_predictions.csv")

# Run main if script is executed
if __name__ == "__main__":
    main()
