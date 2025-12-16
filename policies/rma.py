"""
RMA (Rapid Motor Adaptation) policy architecture.
Separates terrain estimation (gamma_t) and extrinsics (z_t).
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from .monolithic import DepthConvNet, ScandotsMLP


class RMAPolicy(nn.Module):
    """
    RMA policy with separate terrain and extrinsics estimation.
    Phase 1: Uses scandots for terrain, privileged info for extrinsics
    Phase 2: Uses depth for terrain, proprioception history for extrinsics
    """
    
    def __init__(self, config: dict, phase: int = 1):
        """
        Initialize RMA policy.
        
        Args:
            config: Policy configuration
            phase: Training phase (1 or 2)
        """
        super().__init__()
        
        self.phase = phase
        self.config = config
        
        # Terrain estimation (gamma_t)
        if phase == 1:
            # Phase 1: Process scandots
            terrain_config = config.get('terrain_gru', {})
            num_scandots = config.get('num_scandots', 16)
            scandots_input_dim = num_scandots * 3
            
            self.terrain_processor = nn.GRU(
                input_size=scandots_input_dim,
                hidden_size=terrain_config.get('hidden_dim', 128),
                num_layers=terrain_config.get('num_layers', 1),
                batch_first=False
            )
            terrain_output_dim = terrain_config.get('output_dim', 16)
            self.terrain_projection = nn.Linear(
                terrain_config.get('hidden_dim', 128),
                terrain_output_dim
            )
        else:
            # Phase 2: Process depth with vision GRU
            depth_config = config.get('depth_convnet', {})
            self.depth_processor = DepthConvNet(depth_config)
            depth_dim = depth_config.get('output_dim', 128)
            
            vision_gru_config = config.get('vision_gru', {})
            proprio_dim = 48
            cmd_dim = 3
            vision_gru_input_dim = proprio_dim + cmd_dim + depth_dim
            
            self.vision_gru = nn.GRU(
                input_size=vision_gru_input_dim,
                hidden_size=vision_gru_config.get('hidden_dim', 128),
                num_layers=vision_gru_config.get('num_layers', 1),
                batch_first=False
            )
            terrain_output_dim = vision_gru_config.get('output_dim', 16)
            self.terrain_projection = nn.Linear(
                vision_gru_config.get('hidden_dim', 128),
                terrain_output_dim
            )
        
        # Extrinsics estimation (z_t)
        if phase == 1:
            # Phase 1: Process privileged information
            extrinsics_config = config.get('extrinsics_mlp', {})
            privileged_dim = config.get('num_privileged', 20)
            
            layers = []
            prev_dim = privileged_dim
            for hidden_dim in extrinsics_config.get('hidden_dims', [64, 32]):
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU()
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, extrinsics_config.get('output_dim', 8)))
            self.extrinsics_mlp = nn.Sequential(*layers)
        else:
            # Phase 2: Estimate from proprioception history
            proprio_gru_config = config.get('proprio_gru', {})
            proprio_dim = 48
            cmd_dim = 3
            
            self.proprio_gru = nn.GRU(
                input_size=proprio_dim + cmd_dim,
                hidden_size=proprio_gru_config.get('hidden_dim', 64),
                num_layers=proprio_gru_config.get('num_layers', 1),
                batch_first=False
            )
            extrinsics_output_dim = proprio_gru_config.get('output_dim', 8)
            self.extrinsics_projection = nn.Linear(
                proprio_gru_config.get('hidden_dim', 64),
                extrinsics_output_dim
            )
        
        # Base policy (MLP)
        base_config = config.get('base_mlp', {})
        proprio_dim = 48
        cmd_dim = 3
        terrain_dim = terrain_output_dim
        extrinsics_dim = config.get('extrinsics_mlp', {}).get('output_dim', 8) if phase == 1 else extrinsics_output_dim
        
        base_input_dim = proprio_dim + cmd_dim + terrain_dim + extrinsics_dim
        
        layers = []
        prev_dim = base_input_dim
        for hidden_dim in base_config.get('hidden_dims', [256, 128]):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 12))  # 12 joint actions
        
        if base_config.get('activation', 'tanh') == 'tanh':
            layers.append(nn.Tanh())
        
        self.base_policy = nn.Sequential(*layers)
        
        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(base_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Hidden states
        self.terrain_hidden = None
        self.vision_hidden = None
        self.proprio_hidden = None
        
        # Freeze base policy in Phase 2 if specified
        if phase == 2 and base_config.get('frozen', False):
            for param in self.base_policy.parameters():
                param.requires_grad = False
    
    def reset_hidden(self, batch_size: int, device: torch.device):
        """Reset all GRU hidden states."""
        if self.phase == 1:
            terrain_hidden_dim = self.terrain_processor.hidden_size
            terrain_num_layers = self.terrain_processor.num_layers
            self.terrain_hidden = torch.zeros(
                terrain_num_layers, batch_size, terrain_hidden_dim, device=device
            )
        else:
            vision_hidden_dim = self.vision_gru.hidden_size
            vision_num_layers = self.vision_gru.num_layers
            self.vision_hidden = torch.zeros(
                vision_num_layers, batch_size, vision_hidden_dim, device=device
            )
            
            proprio_hidden_dim = self.proprio_gru.hidden_size
            proprio_num_layers = self.proprio_gru.num_layers
            self.proprio_hidden = torch.zeros(
                proprio_num_layers, batch_size, proprio_hidden_dim, device=device
            )
    
    def forward(
        self,
        observations: dict,
        terrain_hidden: Optional[torch.Tensor] = None,
        vision_hidden: Optional[torch.Tensor] = None,
        proprio_hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            observations: Dictionary with:
                - proprioception: (batch, 48)
                - scandots (Phase 1) or depth (Phase 2)
                - commands: (batch, 3)
                - privileged (Phase 1 only): (batch, num_privileged)
            terrain_hidden: Optional terrain GRU hidden state
            vision_hidden: Optional vision GRU hidden state (Phase 2)
            proprio_hidden: Optional proprio GRU hidden state (Phase 2)
        
        Returns:
            actions: (batch, 12) joint target angles
            values: (batch, 1) value estimates
            latents: Dictionary with gamma_t and z_t
        """
        proprio = observations['proprioception']
        commands = observations['commands']
        
        # Estimate terrain latent (gamma_t)
        if self.phase == 1:
            scandots = observations['scandots']
            if scandots.dim() == 3:
                scandots = scandots.flatten(1)  # (batch, num_scandots*3)
            
            if scandots.dim() == 2:
                scandots = scandots.unsqueeze(0)  # (1, batch, features)
            
            if terrain_hidden is None:
                terrain_hidden = self.terrain_hidden
            
            terrain_gru_out, new_terrain_hidden = self.terrain_processor(
                scandots, terrain_hidden
            )
            terrain_gru_out = terrain_gru_out.squeeze(0)  # (batch, hidden_dim)
            gamma_t = self.terrain_projection(terrain_gru_out)
            
            new_vision_hidden = None
        else:
            depth = observations['depth']
            depth_features = self.depth_processor(depth)  # (batch, depth_dim)
            
            vision_input = torch.cat([proprio, commands, depth_features], dim=-1)
            if vision_input.dim() == 2:
                vision_input = vision_input.unsqueeze(0)
            
            if vision_hidden is None:
                vision_hidden = self.vision_hidden
            
            vision_gru_out, new_vision_hidden = self.vision_gru(
                vision_input, vision_hidden
            )
            vision_gru_out = vision_gru_out.squeeze(0)
            gamma_t = self.terrain_projection(vision_gru_out)
            
            new_terrain_hidden = None
        
        # Estimate extrinsics latent (z_t)
        if self.phase == 1:
            privileged = observations['privileged']
            z_t = self.extrinsics_mlp(privileged)
            new_proprio_hidden = None
        else:
            proprio_input = torch.cat([proprio, commands], dim=-1)
            if proprio_input.dim() == 2:
                proprio_input = proprio_input.unsqueeze(0)
            
            if proprio_hidden is None:
                proprio_hidden = self.proprio_hidden
            
            proprio_gru_out, new_proprio_hidden = self.proprio_gru(
                proprio_input, proprio_hidden
            )
            proprio_gru_out = proprio_gru_out.squeeze(0)
            z_t = self.extrinsics_projection(proprio_gru_out)
        
        # Base policy forward
        base_input = torch.cat([proprio, commands, gamma_t, z_t], dim=-1)
        actions = self.base_policy(base_input)
        values = self.value_head(base_input)
        
        latents = {
            'gamma_t': gamma_t,
            'z_t': z_t
        }
        
        hidden_states = {
            'terrain_hidden': new_terrain_hidden,
            'vision_hidden': new_vision_hidden,
            'proprio_hidden': new_proprio_hidden
        }
        
        return actions, values, latents, hidden_states
