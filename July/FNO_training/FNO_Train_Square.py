"""
#################### 
# Fourier Neural Operator trainer for square domains - for Fourth Year Project
# Author - Thomas Higham
# Date - 18/06/2025
# University of Warwick
#####################
I use "config_square_FNO.json" to specify my hyperparameters
# """


"""
If everything is set up in the config file correctly, you just need to run this code to generate an FNO.
"""

# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torch.fft import fft2, ifft2
import matplotlib.pyplot as plt
import json
import os


# ===============================
# DEVICE SETUP (MPS or CPU)  -- This allows the GPU to be used on my MAC for pytorch
# ===============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ===============================
# LOAD CONFIGURATION
# ===============================
def load_config(config_path="config_square_FNO.json"):
    """Load hyperparameters from JSON configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Loaded configuration:")
    print(json.dumps(config, indent=2))
    return config


# ===============================
# LOAD AND PREPARE DATA
"""Specify gridsize here """
# ===============================
class PDEOperatorDataset(Dataset):
    def __init__(self, grouped_data, grid_size):
        self.grouped_data = grouped_data
        self.grid_size = grid_size

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        # Get the group (PDE solution) based on index
        group_df = self.grouped_data.get_group(idx)


        # Extract input features (x, y, b1, b2) and target (rho)
        x = torch.tensor(group_df[['x', 'y', 'b1', 'b2']].values, dtype=torch.float32)

        # Extract 'rho' as the target variable
        if 'rho' in group_df.columns:
            target = torch.tensor(group_df['rho'].values, dtype=torch.float32)
        else:
            raise ValueError("'rho' column not found in group_df")

        # Reshape to grid format 
        x = x.view(self.grid_size, self.grid_size, -1).permute(2, 0, 1)  # [4, 16, 16]
        target = target.view(self.grid_size, self.grid_size)

        return x, target

class NPZOperatorDataset(Dataset):
    def __init__(self, data, grid_size):
        self.grid_size = grid_size
        # data is a dict of arrays, each with shape (N, grid_size, grid_size)
        self.solution_ids = data['solution_id']
        self.x = data['x']
        self.y = data['y']
        self.b1 = data['b1']
        self.b2 = data['b2']
        self.rho = data['rho']
        self.N = self.x.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Each field is (grid_size, grid_size)
        x = np.stack([
            self.x[idx],
            self.y[idx],
            self.b1[idx],
            self.b2[idx]
        ], axis=0)  # [4, grid_size, grid_size]
        target = self.rho[idx]  # [grid_size, grid_size]
        x = torch.tensor(x, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

# ===============================
# DEFINE FOURIER LAYER
# ===============================
class FourierLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FourierLayer2D, self).__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(torch.randn(1, out_channels, modes1, modes2)  # Shape [1, 64, 12, 12] or similar
        )

    def forward(self, x):
            # Fourier transform processing (FFT)
            x_ft = torch.fft.fftn(x, dim=(-2, -1))
            
            # Apply weights (make sure the shapes match)
            out_ft = torch.zeros_like(x_ft)
            out_ft[:, :, :self.modes1, :self.modes2] = x_ft[:, :, :self.modes1, :self.modes2] * self.weights
            
            # Inverse Fourier transform
            out = torch.fft.ifftn(out_ft, dim=(-2, -1)).real
            
            return out

# ===============================
# DEFINE FNO MODEL
# ===============================
class FNO2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width, num_fourier_layers):
        super(FNO2D, self).__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.fourier_layers = nn.ModuleList([
            FourierLayer2D(width, width, modes1, modes2) for _ in range(num_fourier_layers)
        ])
        self.fc1 = nn.Linear(width, 128) #This is the penultimate fourier layer which has dim 128
        self.fc2 = nn.Linear(128, out_channels) #This reduces the dimension from 128 to 1 for the output layer

    def forward(self, x):
        # Initial linear transform
        x = self.fc0(x.permute(0, 2, 3, 1))  # [B, 16, 16, in_channels] -> [B, 16, 16, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, 16, 16]


        # Apply Fourier layers
        for layer in self.fourier_layers:
            x = F.gelu(layer(x))
        # Final linear layers
        x = x.permute(0, 2, 3, 1)  # [B, 16, 16, width]
        x = F.gelu(self.fc1(x))  #Applying GELU

        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)  # [B, out_channels, 16, 16]


# ===============================
# LOAD CSV AND CREATE DATASET
"""Specify gridsize here """
# ===============================
def load_data(data_path, grid_size, batch_size):
    if data_path.endswith('.npz'):
        # Load NPZ
        data = np.load(data_path)
        dataset = NPZOperatorDataset(data, grid_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    else:
        # Load CSV
        data = pd.read_csv(data_path)
        grouped_data = data.groupby('solution_id')
        dataset = PDEOperatorDataset(grouped_data, grid_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader


# ===============================
# TRAINING LOOP
# ===============================

def train_model(model, dataloader, config):
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    num_epochs = config['training']['num_epochs']
    
    # Lists to store training history
    epoch_losses = []
    epochs = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        epochs.append(epoch + 1)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_losses, 'b-', label='Training Loss')
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(config['output']['loss_plot_path'])
    plt.close()

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    # Load configuration
    config = load_config()
    
    # Load data
    dataloader = load_data(
        data_path=config['data']['data_path'],
        grid_size=config['data']['grid_size'],
        batch_size=config['training']['batch_size']
    )

    # Initialize model
    model = FNO2D(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        modes1=config['model']['modes1'],
        modes2=config['model']['modes2'],
        width=config['model']['width'],
        num_fourier_layers=config['model']['num_fourier_layers']
    ).to(device)

    # Train the model
    train_model(model, dataloader, config)

    # Save the model after training
    torch.save(model.state_dict(), config['output']['model_save_path'])
    print('Model saved successfully!')

# Run main if script is executed
if __name__ == "__main__":
    main()

