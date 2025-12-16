"""
Monolithic GRU-based policy architecture.
Processes scandots (Phase 1) or depth (Phase 2) directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DepthConvNet(nn.Module):
    """Convolutional network for processing depth images."""
    
    def __init__(self, config: dict):
        """
        Initialize depth processing network.
        
        Args:
            config: Configuration with channels, kernel_sizes, strides, output_dim
        """
        super().__init__()
        
        channels = config.get('channels', [1, 16, 32])
        kernel_sizes = config.get('kernel_sizes', [3, 3])
        strides = config.get('strides', [2, 2])
        output_dim = config.get('output_dim', 128)
        
        layers = []
        in_channels = channels[0]
        
        for out_channels, kernel_size, stride in zip(channels[1:], kernel_sizes, strides):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size (approximate)
        # For 58x87 input with stride 2 and pool 2: ~15x22 -> ~7x11 -> ~3x5
        # With 32 channels: ~3*5*32 = 480, but we'll use adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten_size = 4 * 4 * channels[-1]
        
        self.fc = nn.Linear(self.flatten_size, output_dim)
    
    def forward(self, depth_images: torch.Tensor) -> torch.Tensor:
        """
        Process depth images.
        
        Args:
            depth_images: (batch, height, width) or (batch, 1, height, width)
        
        Returns:
            features: (batch, output_dim) compressed depth features
        """
        if depth_images.dim() == 3:
            depth_images = depth_images.unsqueeze(1)  # Add channel dimension
        
        x = self.conv_layers(depth_images)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x


class ScandotsMLP(nn.Module):
    """MLP for compressing scandots (Phase 1)."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        """
        Initialize scandots compression network.
        
        Args:
            input_dim: Input dimension (num_scandots * 3 for x,y,z)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (gamma_t)
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, scandots: torch.Tensor) -> torch.Tensor:
        """
        Compress scandots to latent representation.
        
        Args:
            scandots: (batch, num_scandots, 3) or (batch, num_scandots*3)
        
        Returns:
            latent: (batch, output_dim)
        """
        if scandots.dim() == 3:
            scandots = scandots.flatten(1)
        
        return self.network(scandots)


class MonolithicPolicy(nn.Module):
    """
    Monolithic GRU-based policy.
    For Phase 1: Uses scandots
    For Phase 2: Uses depth images
    """
    
    def __init__(self, config: dict, phase: int = 1):
        """
        Initialize monolithic policy.
        
        Args:
            config: Policy configuration
            phase: Training phase (1 or 2)
        """
        super().__init__()
        
        self.phase = phase
        self.config = config
        
        if phase == 1:
            # Phase 1: Process scandots
            scandots_config = config.get('scandots_mlp', {})
            num_scandots = config.get('num_scandots', 16)
            scandots_input_dim = num_scandots * 3  # x, y, z
            
            self.scandots_processor = ScandotsMLP(
                input_dim=scandots_input_dim,
                hidden_dims=scandots_config.get('hidden_dims', [64, 32]),
                output_dim=scandots_config.get('output_dim', 16)
            )
            vision_dim = scandots_config.get('output_dim', 16)
        else:
            # Phase 2: Process depth
            depth_config = config.get('depth_convnet', {})
            self.depth_processor = DepthConvNet(depth_config)
            vision_dim = depth_config.get('output_dim', 128)
        
        # GRU configuration
        gru_config = config.get('gru', {})
        # Use input_dim from config if specified, otherwise compute
        if 'input_dim' in gru_config:
            gru_input_dim = gru_config['input_dim']
        else:
            # Compute from components
            # Proprioception: 12 (joint pos) + 12 (joint vel) + 3 (base lin vel) + 3 (base ang vel) + 3 (projected gravity) + 12 (prev actions) = 45
            proprio_dim = 45
            cmd_dim = 3
            gru_input_dim = proprio_dim + vision_dim + cmd_dim
        
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_config.get('hidden_dim', 256),
            num_layers=gru_config.get('num_layers', 1),
            batch_first=False  # Isaac Gym uses (seq_len, batch, features)
        )
        
        # Store expected dimension for validation
        self.expected_gru_input_dim = gru_input_dim
        
        # Action MLP
        action_config = config.get('action_mlp', {})
        hidden_dims = action_config.get('hidden_dims', [128, 64])
        activation = action_config.get('activation', 'tanh')
        
        layers = []
        prev_dim = gru_config.get('hidden_dim', 256)
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 12))  # 12 joint actions
        
        if activation == 'tanh':
            layers.append(nn.Tanh())
        
        self.action_mlp = nn.Sequential(*layers)
        
        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(gru_config.get('hidden_dim', 256), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.hidden_state = None
    
    def reset_hidden(self, batch_size: int, device: torch.device):
        """Reset GRU hidden state."""
        hidden_dim = self.gru.hidden_size
        num_layers = self.gru.num_layers
        self.hidden_state = torch.zeros(
            num_layers, batch_size, hidden_dim, device=device
        )
    
    def forward(
        self,
        observations: dict,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            observations: Dictionary with:
                - proprioception: (batch, 48)
                - scandots (Phase 1): (batch, num_scandots, 3) or depth (Phase 2): (batch, H, W)
                - commands: (batch, 3)
            hidden_state: Optional GRU hidden state
        
        Returns:
            actions: (batch, 12) joint target angles
            values: (batch, 1) value estimates
            new_hidden: (num_layers, batch, hidden_dim) new hidden state
        """
        proprio = observations['proprioception']
        commands = observations['commands']
        
        # Process vision input
        if self.phase == 1:
            scandots = observations['scandots']
            vision_features = self.scandots_processor(scandots)
        else:
            depth = observations['depth']
            vision_features = self.depth_processor(depth)
        
        # Debug: Check dimensions (can be removed after fixing)
        # print(f"Debug - proprio: {proprio.shape}, vision: {vision_features.shape}, commands: {commands.shape}")
        
        # Concatenate inputs
        gru_input = torch.cat([proprio, vision_features, commands], dim=-1)
        
        # Validate and fix dimension mismatch if needed
        expected_dim = self.expected_gru_input_dim
        actual_dim = gru_input.shape[-1]
        
        if actual_dim != expected_dim:
            # Try to fix common mismatches
            if actual_dim == expected_dim - 3:
                # Missing commands - pad with zeros
                missing = expected_dim - actual_dim
                gru_input = torch.cat([gru_input, torch.zeros(*gru_input.shape[:-1], missing, device=gru_input.device)], dim=-1)
            elif actual_dim == expected_dim + 3:
                # Extra dimensions - truncate
                gru_input = gru_input[..., :expected_dim]
            else:
                # Cannot auto-fix, raise error with details
                raise RuntimeError(
                    f"GRU input dimension mismatch: expected {expected_dim}, got {actual_dim}. "
                    f"Breakdown - proprio: {proprio.shape[-1]}, vision: {vision_features.shape[-1]}, "
                    f"commands: {commands.shape[-1]}, total: {actual_dim}. "
                    f"Please check environment observation dimensions match policy configuration."
                )
        
        # Reshape for GRU: (batch, features) -> (1, batch, features)
        if gru_input.dim() == 2:
            gru_input = gru_input.unsqueeze(0)
        
        # GRU forward
        if hidden_state is None:
            hidden_state = self.hidden_state
        
        gru_out, new_hidden = self.gru(gru_input, hidden_state)
        
        # Extract last output
        gru_out = gru_out.squeeze(0)  # (batch, hidden_dim)
        
        # Predict actions
        actions = self.action_mlp(gru_out)
        
        # Predict values
        values = self.value_head(gru_out)
        
        return actions, values, new_hidden
