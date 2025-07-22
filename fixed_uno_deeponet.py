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
import matplotlib.pyplot as plt
import json
import os

# ===============================
# DEVICE SETUP (MPS or CPU)
# ===============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# LOAD CONFIGURATION
# ===============================
def load_config(config_path="config_uno_deeponet.json"):
    """Load hyperparameters from JSON configuration file"""
    if not os.path.exists(config_path):
        # Create a default config if none exists
        default_config = {
            "data": {
                "data_path": "110725_training_data_30000_correct_energy.npz",
                "grid_size": 32
            },
            "model": {
                "in_channels": 5,
                "out_channels": 1,
                "branch_layers": [256, 256, 256, 256],
                "trunk_layers": [256, 256, 256, 256],
                "encoder_channels": [32, 64, 128, 256],
                "decoder_channels": [256, 128, 64, 32],
                "activation": "gelu",
                "dropout": 0.1
            },
            "training": {
                "batch_size": 16,  # Smaller due to more complex model
                "num_epochs": 100,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "scheduler_step": 50,
                "scheduler_gamma": 0.5
            },
            "output": {
                "loss_plot_path": "uno_deeponet_loss_plot.png",
                "model_save_path": "uno_deeponet_model.pth"
            }
        }
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default config file: {config_path}")
        return default_config
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Loaded configuration:")
    print(json.dumps(config, indent=2))
    return config

# ===============================
# DEEPONET COMPONENTS
# ===============================
class MLP(nn.Module):
    """Multi-layer perceptron with dropout and activation"""
    def __init__(self, layers, activation='gelu', dropout=0.1, final_activation=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.final_activation = final_activation
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2 or final_activation:
                self.dropouts.append(nn.Dropout(dropout))
            else:
                self.dropouts.append(nn.Identity())
        
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.gelu

    def forward(self, x):
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            x = layer(x)
            if i < len(self.layers) - 1 or self.final_activation:
                x = self.activation(x)
            x = dropout(x)
        return x

class DeepONetCore(nn.Module):
    """DeepONet implementation for operator learning"""
    def __init__(self, branch_input_dim, trunk_input_dim, branch_layers, trunk_layers, 
                 activation='gelu', dropout=0.1):
        super(DeepONetCore, self).__init__()
        branch_arch = [branch_input_dim] + branch_layers
        self.branch_net = MLP(branch_arch, activation, dropout, final_activation=True)
        trunk_arch = [trunk_input_dim] + trunk_layers
        self.trunk_net = MLP(trunk_arch, activation, dropout, final_activation=True)
        assert branch_layers[-1] == trunk_layers[-1], "Branch and trunk final layers must have same size"
        self.latent_dim = branch_layers[-1]
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        batch_size, num_points, _ = trunk_input.shape
        branch_output = self.branch_net(branch_input)  # (batch_size, latent_dim)
        trunk_input_flat = trunk_input.reshape(-1, trunk_input.shape[-1])
        trunk_output_flat = self.trunk_net(trunk_input_flat)
        trunk_output = trunk_output_flat.reshape(batch_size, num_points, -1)
        output = torch.einsum('bl,bpl->bp', branch_output, trunk_output) + self.bias
        return output

# ===============================
# U-NET ENCODER/DECODER
# ===============================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='gelu', dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.gelu
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, channels_list, activation='gelu', dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels_list:
            self.blocks.append(ConvBlock(prev_ch, ch, activation, dropout))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = ch
    def forward(self, x):
        skip_connections = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skip_connections.append(x)
            x = pool(x)
        return x, skip_connections

class Decoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels, activation='gelu', dropout=0.1):
        super().__init__()
        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        
        # Reverse encoder channels for skip connections (from deepest to shallowest)
        skip_channels = encoder_channels[::-1]
        
        # First upsample: from decoder input to deepest skip connection level
        self.upsamples.append(nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2))
        self.blocks.append(ConvBlock(decoder_channels[1] + skip_channels[0], decoder_channels[1], activation, dropout))
        
        # Subsequent upsamples
        for i in range(1, len(decoder_channels) - 1):
            self.upsamples.append(nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], kernel_size=2, stride=2))
            self.blocks.append(ConvBlock(decoder_channels[i+1] + skip_channels[i], decoder_channels[i+1], activation, dropout))
        
        # Final output layer
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)
    
    def forward(self, x, skip_connections):
        # Skip connections are in order from shallowest to deepest
        # We need them from deepest to shallowest for decoding
        skip_connections = skip_connections[::-1]
        
        for i, (upsample, block) in enumerate(zip(self.upsamples, self.blocks)):
            x = upsample(x)
            skip = skip_connections[i]
            
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        
        x = self.final_conv(x)
        return x

# ===============================
# U-NO WITH DEEPONET BACKBONE
# ===============================
class UNODeepONet(nn.Module):
    """U-shaped Neural Operator with DeepONet backbone"""
    def __init__(self, in_channels, out_channels, encoder_channels, decoder_channels,
                 branch_layers, trunk_layers, activation='gelu', dropout=0.1, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.in_channels = in_channels
        
        # Calculate the bottleneck spatial size after encoder downsampling
        self.bottleneck_size = grid_size // (2**len(encoder_channels))
        print(f"Bottleneck spatial size: {self.bottleneck_size}x{self.bottleneck_size}")
        
        # Encoder
        self.encoder = Encoder(in_channels, encoder_channels, activation, dropout)
        
        # DeepONet core in the bottleneck
        bottleneck_feature_size = encoder_channels[-1] * self.bottleneck_size**2
        trunk_input_dim = 2  # x, y coordinates
        
        self.deeponet = DeepONetCore(
            branch_input_dim=bottleneck_feature_size,
            trunk_input_dim=trunk_input_dim,
            branch_layers=branch_layers,
            trunk_layers=trunk_layers,
            activation=activation,
            dropout=dropout
        )
        
        # Projection layers
        self.deeponet_projection = nn.Conv2d(1, encoder_channels[-1], kernel_size=1)
        
        # Combined projection: concatenated features -> decoder input channels
        self.combined_proj = nn.Conv2d(encoder_channels[-1] * 2, decoder_channels[0], kernel_size=1)
        
        # Decoder
        self.decoder = Decoder(encoder_channels, decoder_channels, out_channels, activation, dropout)
        
        # Create coordinate grid for bottleneck resolution (not full resolution)
        self.register_buffer('coords', self._create_coordinate_grid(self.bottleneck_size))
    
    def _create_coordinate_grid(self, grid_size):
        """Create coordinate grid for the specified grid size"""
        x = torch.linspace(0, 1, grid_size)
        y = torch.linspace(0, 1, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return coords
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Encoder path
        encoded, skip_connections = self.encoder(x)
        
        # DeepONet in bottleneck
        branch_input = encoded.flatten(1)  # (batch_size, bottleneck_feature_size)
        coords_batch = self.coords.unsqueeze(0).expand(batch_size, -1, -1)
        
        deeponet_output = self.deeponet(branch_input, coords_batch)
        # Reshape to bottleneck spatial dimensions
        deeponet_2d = deeponet_output.view(batch_size, 1, self.bottleneck_size, self.bottleneck_size)
        
        # Project DeepONet output to match encoder channel dimension
        deeponet_projected = self.deeponet_projection(deeponet_2d)
        
        # Concatenate bottleneck features and DeepONet features
        combined = torch.cat([encoded, deeponet_projected], dim=1)
        
        # Project to decoder input channels
        decoder_input = self.combined_proj(combined)
        
        # Decoder path with skip connections
        output = self.decoder(decoder_input, skip_connections)
        
        return output

# ===============================
# DATA LOADING (Same as before)
# ===============================
class PDEOperatorDataset(Dataset):
    def __init__(self, grouped_data, grid_size):
        self.grouped_data = grouped_data
        self.grid_size = grid_size
    def __len__(self):
        return len(self.grouped_data)
    def __getitem__(self, idx):
        group_df = self.grouped_data.get_group(idx)
        x = torch.tensor(group_df[['x_ij', 'y_ij', 'b1', 'b2', 'finv_val']].values, dtype=torch.float32)
        if 'rho_val' in group_df.columns:
            target = torch.tensor(group_df['rho_val'].values, dtype=torch.float32)
        else:
            raise ValueError("'rho_val' column not found in group_df")
        x = x.view(self.grid_size, self.grid_size, -1).permute(2, 0, 1)
        target = target.view(self.grid_size, self.grid_size)
        return x, target

class NPZOperatorDataset(Dataset):
    def __init__(self, data, grid_size):
        columns = data['columns']
        data_array = data['data']
        if isinstance(columns, np.ndarray):
            columns = columns.tolist()
        columns = [str(col) for col in columns]
        self.df = pd.DataFrame(data_array, columns=columns)
        self.grid_size = grid_size
        self.grouped_data = self.df.groupby('solution_id')
        self.indices = list(self.grouped_data.groups.keys())
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        group_df = self.grouped_data.get_group(self.indices[idx])
        x = torch.tensor(group_df[['x_ij', 'y_ij', 'b1', 'b2', 'finv_val']].values, dtype=torch.float32)
        target = torch.tensor(group_df['rho_val'].values, dtype=torch.float32)
        x = x.view(self.grid_size, self.grid_size, -1).permute(2, 0, 1)
        target = target.view(self.grid_size, self.grid_size)
        return x, target

def load_data(data_path, grid_size, batch_size):
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        dataset = NPZOperatorDataset(data, grid_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    else:
        data = pd.read_csv(data_path)
        grouped_data = data.groupby('solution_id')
        dataset = PDEOperatorDataset(grouped_data, grid_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

# ===============================
# LOSS FUNCTIONS
# ===============================
def relative_l2_loss(pred, target):
    return torch.norm(pred - target, 2) / torch.norm(target, 2)

class RelativeL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        batch_size = pred.shape[0]
        pred_flat = pred.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)
        loss = 0.0
        for i in range(batch_size):
            loss += relative_l2_loss(pred_flat[i], target_flat[i])
        return loss / batch_size

# ===============================
# TRAINING LOOP
# ===============================
def train_model(model, dataloader, config):
    criterion = RelativeL2Loss()
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['training']['scheduler_step'], 
        gamma=config['training']['scheduler_gamma']
    )
    num_epochs = config['training']['num_epochs']
    epoch_losses = []
    rel_l2_losses = []
    epochs = []
    print("Starting U-NO with DeepONet training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rel_l2 = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            y = y.unsqueeze(1)
            mse_loss = mse_criterion(outputs, y)
            rel_l2 = criterion(outputs, y)
            loss = mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += mse_loss.item()
            epoch_rel_l2 += rel_l2.item()
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_rel_l2 = epoch_rel_l2 / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        rel_l2_losses.append(avg_rel_l2)
        epochs.append(epoch + 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], MSE Loss: {avg_epoch_loss:.6f}, "
                  f"Rel L2: {avg_rel_l2:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_losses, 'b-', label='MSE Loss')
    plt.title('MSE Training Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, rel_l2_losses, 'r-', label='Relative L2 Loss')
    plt.title('Relative L2 Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Relative L2 Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(config['output']['loss_plot_path'], dpi=300, bbox_inches='tight')
    plt.close()
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'final_mse_loss': avg_epoch_loss,
        'final_rel_l2_loss': avg_rel_l2
    }
    torch.save(checkpoint, config['output']['model_save_path'])
    print(f'Model saved to {config["output"]["model_save_path"]}')
    model.eval()
    total_rel_l2 = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = model(x)
            total_rel_l2 += criterion(pred, y).item()
    final_rel_l2 = total_rel_l2 / len(dataloader)
    print(f"Final Relative L2 loss on training data: {final_rel_l2:.6f}")

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    config = load_config()
    dataloader = load_data(
        data_path=config['data']['data_path'],
        grid_size=config['data']['grid_size'],
        batch_size=config['training']['batch_size']
    )
    model = UNODeepONet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        encoder_channels=config['model']['encoder_channels'],
        decoder_channels=config['model']['decoder_channels'],
        branch_layers=config['model']['branch_layers'],
        trunk_layers=config['model']['trunk_layers'],
        activation=config['model']['activation'],
        dropout=config['model']['dropout'],
        grid_size=config['data']['grid_size']
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"U-NO DeepONet model created with {total_params:,} parameters")
    print(f"Branch layers: {config['model']['branch_layers']}")
    print(f"Trunk layers: {config['model']['trunk_layers']}")
    print(f"Encoder channels: {config['model']['encoder_channels']}")
    print(f"Decoder channels: {config['model']['decoder_channels']}")
    print(f"Bottleneck spatial size: {model.bottleneck_size}x{model.bottleneck_size}")
    train_model(model, dataloader, config)

if __name__ == "__main__":
    main()