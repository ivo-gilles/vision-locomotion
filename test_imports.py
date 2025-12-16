#!/usr/bin/env python3
"""
Test script to verify all imports and dependencies are correctly installed.
Run this after setting up your environment to ensure everything works.
"""
import sys
import os

print("=" * 60)
print("Vision-Locomotion Environment Test")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print()

# Test PyTorch
print("Testing PyTorch...")
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except Exception as e:
    print(f"✗ PyTorch failed: {e}")
print()

# Test NumPy
print("Testing NumPy...")
try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy failed: {e}")
print()

# Test Isaac Sim packages
print("Testing Isaac Sim packages...")
try:
    from omni.isaac.core import World
    print("✓ omni.isaac.core imported")
except Exception as e:
    print(f"✗ omni.isaac.core failed: {e}")
    print("  → Make sure Isaac Sim is installed and accessible")

try:
    from omni.isaac.sensor import Camera
    print("✓ omni.isaac.sensor imported")
except Exception as e:
    print(f"✗ omni.isaac.sensor failed: {e}")
    print("  → Install: pip install omni-isaac-sensor")
print()

# Test legged_gym (optional)
print("Testing legged_gym (optional)...")
try:
    from legged_gym.envs import A1RoughCfg
    print("✓ legged_gym imported")
except Exception as e:
    print(f"⚠ legged_gym not available (optional): {e}")
    print("  → Clone from: https://github.com/leggedrobotics/legged_gym")
print()

# Test project modules
print("Testing project modules...")
try:
    from envs.a1_vision_env import A1VisionEnv
    print("✓ A1VisionEnv imported")
except Exception as e:
    print(f"✗ A1VisionEnv failed: {e}")

try:
    from policies.monolithic import MonolithicPolicy
    print("✓ MonolithicPolicy imported")
except Exception as e:
    print(f"✗ MonolithicPolicy failed: {e}")

try:
    from policies.rma import RMAPolicy
    print("✓ RMAPolicy imported")
except Exception as e:
    print(f"✗ RMAPolicy failed: {e}")

try:
    from trainers.ppo_trainer import PPOTrainer
    print("✓ PPOTrainer imported")
except Exception as e:
    print(f"✗ PPOTrainer failed: {e}")

try:
    from trainers.dagger_trainer import DAggerTrainer
    print("✓ DAggerTrainer imported")
except Exception as e:
    print(f"✗ DAggerTrainer failed: {e}")
print()

# Test other dependencies
print("Testing other dependencies...")
dependencies = [
    ("yaml", "PyYAML"),
    ("cv2", "opencv-python"),
    ("scipy", "scipy"),
    ("stable_baselines3", "stable-baselines3"),
    ("imitation", "imitation"),
    ("gym", "gym"),
    ("wandb", "wandb"),
    ("tensorboard", "tensorboard"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
]

for module_name, package_name in dependencies:
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
    except ImportError:
        print(f"✗ {package_name} not found")
        print(f"  → Install: pip install {package_name}")

print()
print("=" * 60)
print("Test complete!")
print("=" * 60)
