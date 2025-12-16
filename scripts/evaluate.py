#!/usr/bin/env python3
"""
Evaluation script for trained policies.
Updated for Windows compatibility.
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from envs.a1_vision_env import A1VisionEnv
from policies.monolithic import MonolithicPolicy
from policies.rma import RMAPolicy

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
            self.env = type('obj', (object,), {
                'max_episode_length': 1000
            })()


def evaluate_policy(env, policy, num_episodes=10):
    """Evaluate policy performance."""
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        hidden_state = None
        episode_reward = 0.0
        episode_length = 0
        
        done = False
        while not done:
            with torch.no_grad():
                action, _, hidden_state = policy(obs, hidden_state)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward.mean().item()
            episode_length += 1
            
            if done.any():
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check success (simplified - would check terrain completion)
        max_episode_length = 1000  # Default
        if hasattr(env, 'cfg') and hasattr(env.cfg, 'env') and hasattr(env.cfg.env, 'max_episode_length'):
            max_episode_length = env.cfg.env.max_episode_length
        elif hasattr(env, 'cfg') and hasattr(env.cfg, 'max_episode_length'):
            max_episode_length = env.cfg.max_episode_length
        
        if episode_length >= max_episode_length * 0.8:
            success_count += 1
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / num_episodes
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained policy')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--phase', type=int, default=2, help='Phase (1 or 2)')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    
    args = parser.parse_args()
    
    # Load checkpoint (Windows-compatible paths)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / args.checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=args.device)
    config = checkpoint.get('config', {})
    
    # Create environment
    env_cfg = A1RoughCfg()
    env = A1VisionEnv(
        cfg=env_cfg,
        phase=args.phase,
        sim_device=args.device,
        headless=True
    )
    
    # Create policy
    if config.get('policy', {}).get('architecture') == 'rma':
        policy = RMAPolicy(config['policy'], phase=args.phase)
    else:
        policy = MonolithicPolicy(config['policy'], phase=args.phase)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy = policy.to(args.device)
    policy.eval()
    
    # Evaluate
    print(f"Evaluating policy for {args.num_episodes} episodes...")
    metrics = evaluate_policy(env, policy, args.num_episodes)
    
    print("\nEvaluation Results:")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Episode Length: {metrics['mean_length']:.1f}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")


if __name__ == '__main__':
    main()
