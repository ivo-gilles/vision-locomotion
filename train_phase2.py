#!/usr/bin/env python3
"""
Phase 2 Training Script: Distillation with Depth using DAgger
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
from trainers.dagger_trainer import DAggerTrainer

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
    parser = argparse.ArgumentParser(description='Phase 2 Training')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to Phase 2 configuration YAML file'
    )
    parser.add_argument(
        '--phase1_checkpoint',
        type=str,
        required=True,
        help='Path to Phase 1 checkpoint'
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
        help='Path to Phase 2 checkpoint to resume from'
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
    
    # Create environment (Phase 2 with depth)
    print("Creating environment...")
    depth_resolution = tuple(config['env']['depth_resolution'])
    env = A1VisionEnv(
        cfg=env_cfg,
        phase=2,
        depth_resolution=depth_resolution,
        sim_device=args.device,
        headless=True
    )
    
    # Load Phase 1 teacher policy (Windows-compatible paths)
    phase1_checkpoint_path = Path(args.phase1_checkpoint)
    if not phase1_checkpoint_path.is_absolute():
        phase1_checkpoint_path = project_root / args.phase1_checkpoint
    print(f"Loading Phase 1 teacher from {phase1_checkpoint_path}...")
    phase1_checkpoint = torch.load(str(phase1_checkpoint_path), map_location=device)
    
    # Create teacher policy (same architecture as Phase 1)
    # Note: Need to load Phase 1 config to get architecture
    # For now, assume monolithic
    teacher_config = config['policy'].copy()
    teacher_config['architecture'] = 'monolithic'  # Or load from Phase 1 config
    teacher_policy = create_policy({'policy': teacher_config}, phase=1)
    teacher_policy.load_state_dict(phase1_checkpoint['policy_state_dict'])
    teacher_policy = teacher_policy.to(device)
    teacher_policy.eval()
    
    # Create student policy (Phase 2)
    print("Creating student policy...")
    student_policy = create_policy(config, phase=2)
    student_policy = student_policy.to(device)
    
    # Initialize student with teacher weights where applicable
    # Note: GRU weights cannot be copied directly because input dimensions differ:
    #   Phase 1: 48 (proprio) + 16 (scandots) + 3 (cmd) = 67
    #   Phase 2: 48 (proprio) + 128 (depth) + 3 (cmd) = 179
    # Only action MLP weights are compatible since they both use GRU hidden state
    if config['policy']['architecture'] == 'monolithic':
        try:
            # Copy action MLP weights (compatible since both use GRU hidden state)
            student_policy.action_mlp.load_state_dict(teacher_policy.action_mlp.state_dict())
            print("Successfully initialized action MLP with teacher weights.")
        except Exception as e:
            print(f"Warning: Could not copy action MLP weights: {e}")
            print("Continuing with randomly initialized action MLP...")
    
    # Load checkpoint if resuming (Windows-compatible paths)
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = project_root / args.resume
        print(f"Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(str(resume_path), map_location=device)
        student_policy.load_state_dict(checkpoint['student_state_dict'])
    
    # Create trainer
    print("Creating DAgger trainer...")
    trainer = DAggerTrainer(
        env=env,
        teacher_policy=teacher_policy,
        student_policy=student_policy,
        config=config,
        device=args.device
    )
    
    # Train
    print("Starting Phase 2 training...")
    trainer.train()
    
    # Save final checkpoint (Windows-compatible paths)
    save_dir = config.get('logging', {}).get('save_dir', './logs')
    checkpoint_dir = Path(save_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / save_dir
    checkpoint_dir = checkpoint_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = checkpoint_dir / 'phase2_final.pth'
    trainer.save_checkpoint(str(final_checkpoint_path))
    
    print("Phase 2 training complete!")


if __name__ == '__main__':
    main()
