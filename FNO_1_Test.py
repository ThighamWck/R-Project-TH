# pde_operator_testing.py

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

# Import model definition from training script
from FNO_1 import FNO2D, PDEOperatorDataset

# ===============================
# DEVICE SETUP (MPS or CPU)
# ===============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# LOAD TEST DATA
# ===============================
def load_test_data(csv_path, grid_size=63, batch_size=32):
    # Load CSV
    data = pd.read_csv(csv_path)
    
    # Group by solution_id
    grouped_data = data.groupby('solution_id')
    
    # Create dataset and dataloader
    test_dataset = PDEOperatorDataset(grouped_data, grid_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_dataloader

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
def plot_prediction_comparison(model, dataloader, num_samples=3):
    """Visualize predictions vs ground truth for a few samples"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6*num_samples))
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            x, y = x.to(device), y.to(device)
            output = model(x)
            
            # Get the first sample from batch
            input_grid = x[0, :, :, :].cpu().numpy()
            true_field = y[0, :, :].cpu().numpy()
            pred_field = output[0, 0, :, :].cpu().numpy()
            
            # Plot input (b1 parameter as an example)
            im1 = axes[i, 0].imshow(input_grid[2, :, :], cmap='viridis')  # Showing b1 parameter
            axes[i, 0].set_title(f"Input (b1 parameter)")
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Plot ground truth
            im2 = axes[i, 1].imshow(true_field, cmap='coolwarm')
            axes[i, 1].set_title(f"Ground Truth")
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Plot prediction
            im3 = axes[i, 2].imshow(pred_field, cmap='coolwarm')
            axes[i, 2].set_title(f"Prediction")
            plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
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
    # Path to your test CSV file
    test_csv_path = "rho_63_square_test_100.csv"
    model_path = "rho_63_square_10000.pth"
    
    # Load test data
    test_dataloader = load_test_data(test_csv_path)
    
    # Initialize and load model
    model = FNO2D(in_channels=4, out_channels=1, modes1=10, modes2=10, width=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate model
    metrics, all_preds, all_targets = evaluate_model(model, test_dataloader)
    
    # Print metrics
    print("\n===== Model Evaluation Results =====")
    print(f"Test Loss: {metrics['test_loss']:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R² Score: {metrics['r2']:.6f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_prediction_comparison(model, test_dataloader)
    plot_error_distribution(all_preds, all_targets)
    print("Visualizations saved.")
    
    # Save predictions to CSV
    print("\nSaving detailed results to CSV...")
    results_df = pd.DataFrame({
        'actual': all_targets,
        'predicted': all_preds,
        'error': all_preds - all_targets
    })
    results_df.to_csv('model_predictions.csv', index=False)
    print("Results saved to model_predictions.csv")

# Run main if script is executed
if __name__ == "__main__":
    main()
