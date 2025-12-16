#!/usr/bin/env python3
"""
Phase 1 Training Script: Reinforcement Learning with Scandots
Updated for Windows and Isaac Sim compatibility.
"""
import argparse
import yaml
import torch
import os
import sys
from pathlib import Path

# Add project root to path (Windows-compatible)
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from envs.a1_vision_env import A1VisionEnv
from policies.monolithic import MonolithicPolicy
from policies.rma import RMAPolicy
from trainers.ppo_trainer import PPOTrainer

# Try to import legged_gym, but make it optional
try:
    from legged_gym.envs import A1RoughCfg
    LEGGED_GYM_AVAILABLE = True
except ImportError:
    LEGGED_GYM_AVAILABLE = False
    print("Warning: legged_gym not found. Using fallback configuration.")
    # Create fallback A1RoughCfg class
    class A1RoughCfg:
        def __init__(self):
            self.terrain = type('obj', (object,), {
                'num_rows': 20,
                'num_cols': 10,
                'terrain_size': 8.0,
                'fractal_height_range': [0.02, 0.04],
                'types': ['flat']
            })()
            self.num_envs = 4096


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file (Windows-compatible paths)."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = project_root / config_path
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_policy(config: dict, phase: int = 1):
    """Create policy based on architecture."""
    architecture = config['policy']['architecture']
    policy_config = config['policy']
    
    if architecture == 'monolithic':
        policy = MonolithicPolicy(policy_config, phase=phase)
    elif architecture == 'rma':
        policy = RMAPolicy(policy_config, phase=phase)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return policy


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Training')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (cuda:0, cpu, etc.)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device with CUDA availability check
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. PyTorch was not compiled with CUDA support.")
        print("Please install PyTorch with CUDA support:")
        print("  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        print("  OR")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("Falling back to CPU...")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    # Create environment configuration
    env_cfg = A1RoughCfg()
    if hasattr(env_cfg, 'terrain'):
        terrain_config = config['env']['terrain']
        env_cfg.terrain.num_rows = terrain_config.get('num_rows', 20)
        env_cfg.terrain.num_cols = terrain_config.get('num_cols', 10)
        env_cfg.terrain.terrain_size = terrain_config.get('terrain_size', 8.0)
        # Set additional terrain attributes if they exist
        if 'fractal_height_range' in terrain_config:
            env_cfg.terrain.fractal_height_range = terrain_config['fractal_height_range']
        if 'types' in terrain_config:
            env_cfg.terrain.types = terrain_config['types']
    if hasattr(env_cfg, 'num_envs'):
        env_cfg.num_envs = config['env'].get('num_envs', 4096)
    
    # Create environment
    print("Creating environment...")
    env = A1VisionEnv(
        cfg=env_cfg,
        phase=1,
        num_scandots=config['env'].get('num_scandots', 16),
        sim_device=args.device,
        headless=True
    )
    
    # Create policy
    print("Creating policy...")
    policy = create_policy(config, phase=1)
    policy = policy.to(device)
    
    # Create trainer
    print("Creating trainer...")
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        config=config,
        device=args.device
    )
    
    # Load checkpoint if resuming (Windows-compatible paths)
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = project_root / args.resume
        print(f"Resuming training from {resume_path}...")
        trainer.load_checkpoint(str(resume_path))
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final checkpoint (Windows-compatible paths)
    save_dir = config.get('logging', {}).get('save_dir', './logs')
    checkpoint_dir = Path(save_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / save_dir
    checkpoint_dir = checkpoint_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = checkpoint_dir / 'phase1_final.pth'
    trainer.save_checkpoint(str(final_checkpoint_path))
    
    print("Training complete!")


if __name__ == '__main__':
    main()
