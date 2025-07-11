"""
#################### 
# Fourier Neural Operator trainer for non-square domains - for Fourth Year Project
# Author - Thomas Higham
# Date - 31/03/25
# University of Warwick
#####################
Initialisations is in main function at bottom
# """

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import model definition from training script
from FNO_Train_Curved import FNO2D, PDEOperatorDataset

# ===============================
# DEVICE SETUP (MPS or CPU)
# ===============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# LOAD TEST DATA
# ===============================
def load_test_data(csv_path, grid_size=32, batch_size=128):
    # Load CSV
    data = pd.read_csv(csv_path)
    
    # Group by solution_id
    grouped_data = data.groupby('solution_id')
    
    # Create dataset and dataloader
    test_dataset = PDEOperatorDataset(grouped_data, grid_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_dataloader

# ===============================
# CUSTOM METRIC: Relative L2 Test Error
# ===============================
def relative_l2_test_error(y_true, y_pred, eps=1e-8):
    """
    Compute the relative L2 test error defined as:
        ||y_pred - y_true||_2 / ||y_true||_2
    """
    error_norm = np.linalg.norm(y_pred - y_true)
    true_norm = np.linalg.norm(y_true)
    return error_norm / (true_norm + eps)

# ===============================
# EVALUATION FUNCTION
# ===============================
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()
            
            # Store predictions and targets for metrics
            all_preds.append(outputs.squeeze(1).cpu().numpy())
            all_targets.append(y.squeeze(1).cpu().numpy())
    
    # Concatenate all batches and flatten
    all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    all_targets = np.concatenate([t.reshape(-1) for t in all_targets])
    
    # Calculate standard metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # Calculate relative L2 test error
    rel_l2_error = relative_l2_test_error(all_targets, all_preds)
    
    avg_loss = test_loss / len(dataloader)
    
    metrics = {
        'test_loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'relative_l2_test_error': rel_l2_error
    }
    
    return metrics, all_preds, all_targets

# ===============================
# VISUALIZATION FUNCTIONS
# ===============================
def plot_prediction_comparison(model, dataloader, num_samples=3):
    """Visualize predictions vs ground truth for a few samples"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            x, y = x.to(device), y.to(device)
            output = model(x)
            
            # Get the first sample from batch 1
            input_grid = x[0, :, :, :].cpu().numpy()
            true_field = y[0, :, :].cpu().numpy()
            pred_field = output[0, 0, :, :].cpu().numpy()
            coefficient_field = np.sqrt(input_grid[2, :, :]**2 + input_grid[3, :, :]**2)
            
            # Plot input vector field
            im1 = axes[i, 0].imshow(coefficient_field, cmap='viridis', extent=[
                input_grid[4, :, :].min(), input_grid[4, :, :].max(),
                input_grid[1, :, :].min(), input_grid[1, :, :].max()
            ], origin='lower', aspect='auto')
            axes[i, 0].set_title("Magnitude of Velocity Field b")
            plt.colorbar(im1, ax=axes[i, 0])
            
            finv_vals = input_grid[4, 0, :]  # unique finv values along x-axis
            y_vals = input_grid[1, :, 0]     # unique y values along y-axis
            F, Y = np.meshgrid(finv_vals, y_vals)

            axes[i, 0].streamplot(
                        F, Y,
                        input_grid[2, :, :],  # U component (b1)
                        input_grid[3, :, :],  # V component (b2)
                        color="black", linewidth=0.6, density=1.2
                    )
            axes[i, 0].set_xlabel("finv")
            axes[i, 0].set_ylabel("y")
            
            # Plot ground truth
            im2 = axes[i, 1].imshow(true_field,cmap='coolwarm')
            axes[i, 1].set_title("Ground Truth")
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Plot prediction
            im3 = axes[i, 2].imshow(pred_field,cmap='coolwarm')
            axes[i, 2].set_title("Prediction")
            plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig('100725_prediction_comparison_1500_epoch_28_modes_4_layers.png')
    plt.close()


def plot_error_distribution(predictions, targets):
    """Plot the distribution of prediction errors"""
    errors = predictions - targets
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('error_distribution.png')
    plt.close()
    
    # Also plot scatter of predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.1)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    plt.savefig('pred_vs_actual.png')
    plt.close()

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    # Path to test CSV file 
    test_csv_path = "300625_test_data_32_transform_1000.csv"  
    model_path = "100725_FNO_32_15000_epochs_1500_24modes_4layers.pth"  
    
    # Load test data
    test_dataloader = load_test_data(test_csv_path)
    
    # Initialize and load model
    model = FNO2D(in_channels=5, out_channels=1, modes1=24 , modes2=24, width=256, num_fourier_layers= 4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

  
    
    # Evaluate model
    metrics, all_preds, all_targets = evaluate_model(model, test_dataloader)
    
    # Print metrics
    print("\n===== Model Evaluation Results =====")
    print(f"Test Loss: {metrics['test_loss']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R² Score: {metrics['r2']:.6f}")
    print(f"Relative L2 Test Error: {metrics['relative_l2_test_error']:.6f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_prediction_comparison(model, test_dataloader)
    plot_error_distribution(all_preds, all_targets)
    print("Visualizations saved.")
    
    # Save predictions to CSV with all metadata
    print("\nSaving detailed results to CSV...")

    # Reload the test CSV to get all metadata
    test_df = pd.read_csv(test_csv_path)

    # Sanity check: ensure the number of predictions matches the number of rows
    assert len(test_df) == len(all_preds), "Mismatch between test data and predictions!"

    # Add predictions and error columns
    test_df['rho_actual'] = all_targets
    test_df['rho_predicted'] = all_preds
    test_df['error'] = all_preds - all_targets

    # Save only the desired columns
    cols_to_save = [
        'solution_id', 'x_ij', 'y_ij', 'b1', 'b2', 'finv_val',
        'rho_actual', 'rho_predicted', 'error'
    ]
    test_df[cols_to_save].to_csv('100725_model_predictions_1500_24modes_4layers.csv', index=False)
    print("Results saved to 100725_model_predictions_1000.csv")

# Run main if script is executed
if __name__ == "__main__":
    main()

