"""
Domain randomization utilities for sim-to-real transfer.
Based on Table 3 from the paper.
"""
import torch
import numpy as np
from typing import Dict, Tuple


class DomainRandomizer:
    """Applies domain randomization to simulation parameters."""
    
    def __init__(self, config: Dict):
        """
        Initialize randomizer with config ranges.
        
        Args:
            config: Dictionary with randomization ranges
        """
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset randomization state."""
        self.mass_shift = None
        self.com_shift = None
        self.friction = None
        self.motor_strength = None
        self.pd_stiffness = None
        self.pd_damping = None
        self.last_push_time = None
    
    def sample_parameters(self, num_envs: int, device: str = 'cuda:0'):
        """
        Sample random parameters for all environments.
        
        Args:
            num_envs: Number of parallel environments
            device: Device to place tensors on
        
        Returns:
            params: Dictionary of sampled parameters
        """
        params = {}
        
        # Mass
        if 'mass_range' in self.config:
            mass_range = self.config['mass_range']
            params['mass_shift'] = torch.empty(
                num_envs, device=device
            ).uniform_(*mass_range)
        
        # Center of mass shift
        if 'com_shift_range' in self.config:
            com_range = self.config['com_shift_range']
            params['com_shift'] = torch.empty(
                num_envs, 3, device=device
            ).uniform_(*com_range)
        
        # Friction
        if 'friction_range' in self.config:
            friction_range = self.config['friction_range']
            params['friction'] = torch.empty(
                num_envs, device=device
            ).uniform_(*friction_range)
        
        # Motor strength
        if 'motor_strength_range' in self.config:
            strength_range = self.config['motor_strength_range']
            params['motor_strength'] = torch.empty(
                num_envs, 12, device=device
            ).uniform_(*strength_range)
        
        # PD controller parameters
        if 'pd_stiffness_range' in self.config:
            stiffness_range = self.config['pd_stiffness_range']
            params['pd_stiffness'] = torch.empty(
                num_envs, 12, device=device
            ).uniform_(*stiffness_range)
        
        if 'pd_damping_range' in self.config:
            damping_range = self.config['pd_damping_range']
            params['pd_damping'] = torch.empty(
                num_envs, 12, device=device
            ).uniform_(*damping_range)
        
        return params
    
    def apply_observation_noise(
        self,
        observations: Dict[str, torch.Tensor],
        noise_config: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Apply noise to observations based on Table 2.
        
        Args:
            observations: Dictionary of observation tensors
            noise_config: Configuration for each observation type
        
        Returns:
            noisy_observations: Dictionary with noise applied
        """
        noisy_obs = {}
        
        for obs_name, obs_tensor in observations.items():
            if obs_name in noise_config:
                config = noise_config[obs_name]
                scale = config.get('scale', 1.0)
                bias = config.get('bias', 0.0)
                sigma = config.get('sigma', 0.0)
                
                # Apply scaling and bias
                noisy = scale * (obs_tensor - bias)
                
                # Add Gaussian noise
                if sigma > 0:
                    noise = torch.randn_like(noisy) * sigma
                    noisy = noisy + noise
                
                noisy_obs[obs_name] = noisy
            else:
                noisy_obs[obs_name] = obs_tensor
        
        return noisy_obs
    
    def should_apply_push(self, current_time: float) -> bool:
        """
        Check if a random push should be applied.
        
        Args:
            current_time: Current simulation time
        
        Returns:
            should_push: Whether to apply push
        """
        if 'push_interval' not in self.config:
            return False
        
        if self.last_push_time is None:
            self.last_push_time = current_time
            return True
        
        if current_time - self.last_push_time >= self.config['push_interval']:
            self.last_push_time = current_time
            return True
        
        return False
    
    def sample_push(self, num_envs: int, device: str = 'cuda:0') -> torch.Tensor:
        """
        Sample random push force.
        
        Args:
            num_envs: Number of environments
            device: Device
        
        Returns:
            push_force: (num_envs, 3) push force vector
        """
        velocity = self.config.get('push_velocity', 0.3)
        
        # Random direction in horizontal plane
        angle = torch.empty(num_envs, device=device).uniform_(0, 2 * np.pi)
        push_force = torch.zeros(num_envs, 3, device=device)
        push_force[:, 0] = velocity * torch.cos(angle)
        push_force[:, 1] = velocity * torch.sin(angle)
        
        return push_force
