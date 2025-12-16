# Legged Locomotion in Challenging Terrains using Egocentric Vision

This repository implements the method from the paper "Legged Locomotion in Challenging Terrains using Egocentric Vision" (Agarwal et al., CoRL 2022).

## Overview

The implementation follows a two-phase training approach:
- **Phase 1**: Train a teacher policy using PPO with scandots (terrain height queries) as a proxy for depth
- **Phase 2**: Distill the teacher policy to a student that uses real depth images via DAgger

## Prerequisites

### Hardware
- **Windows 10/11** (or Linux)
- NVIDIA GPU (RTX 4080 or better recommended, minimum RTX 3090) with 16GB+ VRAM
- Intel Core i7 (7th Generation) or AMD Ryzen 5 CPU
- 32GB+ RAM (recommended)
- CPU for control (optional: separate machine for deployment)

### Software Dependencies
- Python 3.8+ (3.9 or 3.10 recommended)
- PyTorch 1.10+ (with CUDA 12.x support for Isaac Sim 2024+)
- **Isaac Sim 2024+** (NVIDIA Omniverse) - [Download](https://www.nvidia.com/en-us/omniverse/isaac-sim/)
- Conda (recommended for Windows environment management)
- stable-baselines3 for PPO
- wandb for experiment tracking (with offline mode support)
- NumPy, SciPy, OpenCV, PyYAML, tqdm

**Note**: This code has been updated to work with Isaac Sim (Omniverse) on Windows. Legacy Isaac Gym Preview support is also included for backward compatibility. The codebase includes Windows-specific PowerShell scripts for easy setup and execution.

## Installation

### For Windows with Isaac Sim (Recommended)

#### Option 1: Using Conda Environment (Recommended)

1. **Set up Conda environment**:
   ```powershell
   # Run the setup script (includes PyTorch CUDA installation)
   .\setup_conda_env.ps1
   
   # Or manually:
   conda create -n isaaclab-new python=3.10
   conda activate isaaclab-new
   .\install_pytorch_cuda.ps1
   ```

2. **Install Isaac Sim**:
   - Download Isaac Sim from [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/isaac-sim/)
   - Extract to `C:\isaacsim` (or your preferred location)
   - Run `post_install.bat` in the Isaac Sim directory
   - Launch Isaac Sim at least once to complete setup

3. **Install Python dependencies**:
   ```powershell
   # Activate conda environment
   conda activate isaaclab-new
   
   # Navigate to project directory
   cd C:\Users\tindu\Desktop\isaaclab\vision-locomotion
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install in development mode
   pip install -e .
   ```

4. **Fix Isaac Sim Python path** (if needed):
   ```powershell
   .\fix_isaac_sim_path.ps1
   ```

#### Option 2: Using Isaac Sim's Python Environment

1. **Install Isaac Sim**:
   - Download Isaac Sim from [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/isaac-sim/)
   - Extract to `C:\isaacsim` (or your preferred location)
   - Run `post_install.bat` in the Isaac Sim directory
   - Launch Isaac Sim at least once to complete setup

2. **Install Python dependencies** (in Isaac Sim's Python environment):
   ```powershell
   # Navigate to vision-locomotion directory
   cd C:\Users\tindu\Desktop\isaaclab\vision-locomotion
   
   # Install dependencies using Isaac Sim's Python
   C:\isaacsim\python.bat -m pip install -r requirements.txt
   C:\isaacsim\python.bat -m pip install -e .
   ```

3. **Install Isaac Sim Python packages** (if not already included):
   ```powershell
   C:\isaacsim\python.bat -m pip install omni-isaac-core
   C:\isaacsim\python.bat -m pip install omni-isaac-sensor
   ```

### For Linux with Isaac Sim

1. **Install Isaac Sim**:
   - Download and extract Isaac Sim
   - Run the setup script
   - Set environment variable (optional):
   ```bash
   export ISAAC_SIM_PATH=/path/to/isaacsim
   ```

2. **Install dependencies**:
   ```bash
   cd vision-locomotion
   # Use Isaac Sim's Python
   /path/to/isaacsim/python.sh -m pip install -r requirements.txt
   /path/to/isaacsim/python.sh -m pip install -e .
   ```

### Legacy: Isaac Gym Preview (Optional)

If you need to use the legacy Isaac Gym Preview:

1. Install Isaac Gym Preview and set environment variable:
   ```bash
   # Windows (PowerShell)
   $env:ISAAC_GYM_PATH="C:\path\to\isaacgym"
   
   # Linux
   export ISAAC_GYM_PATH=/path/to/isaacgym
   ```

2. Install legged_gym (if needed):
   ```bash
   git clone https://github.com/leggedrobotics/legged_gym.git
   cd legged_gym
   pip install -e .
   ```

## Project Structure

```
vision-locomotion/
├── envs/                    # Custom Isaac Sim/Gym environments
│   └── a1_vision_env.py     # A1 robot environment with vision support
├── policies/                 # Policy architectures
│   ├── monolithic.py         # Monolithic GRU-based policy
│   └── rma.py                # RMA (Rapid Motor Adaptation) policy
├── trainers/                 # Training implementations
│   ├── ppo_trainer.py       # PPO trainer for Phase 1 (with wandb error handling)
│   └── dagger_trainer.py    # DAgger trainer for Phase 2 (with wandb error handling)
├── utils/                    # Utility modules
│   ├── rewards.py           # Reward computation functions
│   ├── randomization.py     # Domain randomization utilities
│   ├── terrain.py           # Terrain generation (stairs, gaps, etc.)
│   └── curriculum.py        # Curriculum learning manager
├── configs/                  # Configuration files (YAML)
│   ├── phase1_monolithic.yaml
│   ├── phase1_rma.yaml
│   ├── phase2_monolithic.yaml
│   └── phase2_rma.yaml
├── scripts/                  # Helper scripts
│   ├── evaluate.py          # Policy evaluation script
│   └── export_onnx.py       # ONNX export for deployment
├── train_phase1.py          # Phase 1 training entry point
├── train_phase2.py          # Phase 2 training entry point
├── run_phase1.ps1           # PowerShell script for Phase 1 training (Windows)
├── setup_conda_env.ps1      # Conda environment setup script
├── install_pytorch_cuda.ps1 # PyTorch CUDA installation script
├── fix_isaac_sim_path.ps1   # Isaac Sim path configuration script
├── test_imports.py          # Import testing script
├── setup.py                 # Package setup file
├── requirements.txt         # Python dependencies
└── logs/                    # Training logs and checkpoints
    └── checkpoints/         # Saved model checkpoints
```

## Usage

### Phase 1: Reinforcement Learning with Scandots

#### Windows (PowerShell - Recommended)
```powershell
# Using the provided PowerShell script
.\run_phase1.ps1 --config configs/phase1_monolithic.yaml --device cuda:0

# Or with RMA architecture
.\run_phase1.ps1 --config configs/phase1_rma.yaml --device cuda:0
```

#### Direct Python Execution
```bash
# Monolithic architecture
python train_phase1.py --config configs/phase1_monolithic.yaml --device cuda:0

# RMA architecture
python train_phase1.py --config configs/phase1_rma.yaml --device cuda:0

# Resume from checkpoint
python train_phase1.py --config configs/phase1_monolithic.yaml --resume logs/checkpoints/phase1_checkpoint.pth
```

### Phase 2: Distillation with Depth

```bash
# Distill Phase 1 policy to use depth images
python train_phase2.py \
    --config configs/phase2_monolithic.yaml \
    --phase1_checkpoint logs/checkpoints/phase1_final.pth \
    --device cuda:0

# Resume Phase 2 training
python train_phase2.py \
    --config configs/phase2_monolithic.yaml \
    --phase1_checkpoint logs/checkpoints/phase1_final.pth \
    --resume logs/checkpoints/phase2_checkpoint.pth
```

### Evaluation

Evaluate a trained policy:
```bash
# Evaluate Phase 1 policy
python scripts/evaluate.py \
    --checkpoint logs/checkpoints/phase1_final.pth \
    --phase 1 \
    --num_episodes 10

# Evaluate Phase 2 policy
python scripts/evaluate.py \
    --checkpoint logs/checkpoints/phase2_final.pth \
    --phase 2 \
    --num_episodes 10
```

### Export for Deployment

Export policy to ONNX format for deployment:
```bash
# Export Phase 2 policy (recommended for deployment)
python scripts/export_onnx.py \
    --checkpoint logs/checkpoints/phase2_final.pth \
    --output model.onnx \
    --phase 2 \
    --device cuda:0
```

## Configuration

Edit YAML files in `configs/` to adjust training parameters. Each config file contains:

### Environment Settings
- `num_envs`: Number of parallel environments (default: 4096)
- `num_scandots`: Number of terrain height queries for Phase 1 (default: 16)
- `episode_length`: Maximum episode length in steps (default: 1000)
- `terrain`: Terrain generation parameters (types, size, difficulty)

### Policy Architecture
- `architecture`: "monolithic" or "rma"
- Network dimensions (GRU hidden size, MLP layers, etc.)
- Phase-specific settings (scandots MLP for Phase 1, depth ConvNet for Phase 2)
- Value head automatically added for PPO training

### Training Hyperparameters (PPO)
- `total_timesteps`: Total training timesteps (default: 15B for Phase 1)
- `learning_rate`: Initial learning rate (default: 3e-4, with cosine annealing)
- `batch_size`: Number of sequences per batch (default: 64)
- `n_epochs`: Number of PPO update epochs per rollout (default: 10)
- `bptt_length`: Backpropagation through time length (default: 24)
- `clip_range`: PPO clipping parameter (default: 0.2)
- `entropy_coef`: Entropy bonus coefficient (default: 0.01)
- `value_loss_coef`: Value function loss coefficient (default: 0.5)
- `max_grad_norm`: Gradient clipping threshold (default: 1.0)
- `checkpoint_freq`: Checkpoint saving frequency in steps (default: 1M)

### Training Hyperparameters (DAgger - Phase 2)
- `total_timesteps`: Total training timesteps (default: 6M for Phase 2)
- `learning_rate`: Learning rate (default: 1e-4)
- `bc_initial_samples`: Initial behavioral cloning samples (default: 1M)
- `dagger_iterations`: Number of DAgger iterations (default: 100)

### Domain Randomization
- Mass, COM shift, friction ranges
- Motor strength and PD gains
- External push forces

### Logging
- `use_wandb`: Enable/disable Weights & Biases logging (default: true, uses offline mode)
- `project_name`: Wandb project name
- `log_freq`: Logging frequency in steps (default: 1000)
- `save_dir`: Directory for checkpoints and logs

**Note**: Wandb is configured to use offline mode by default to avoid connection issues. Logs are saved locally and can be synced later with `wandb sync wandb/offline-run-*`.

## Key Features

### Architecture Options
- **Monolithic Architecture**: Simple GRU-based policy that processes vision (scandots or depth) directly
- **RMA Architecture**: Modular architecture with separate terrain estimation (γ_t) and extrinsics estimation (z_t)

### Training Features
- **Two-Phase Training**: Phase 1 uses cheap scandots, Phase 2 distills to real depth images
- **Curriculum Learning**: Automatic terrain difficulty progression based on performance (`utils/curriculum.py`)
- **Domain Randomization**: Comprehensive randomization for robust sim-to-real transfer (`utils/randomization.py`)
- **BPTT Support**: Backpropagation through time for recurrent policies
- **Checkpointing**: Automatic checkpoint saving and resuming support

### Platform Support
- **Windows Compatibility**: Full Windows support with PowerShell scripts and pathlib-based path handling
- **Isaac Sim Integration**: Native support for Isaac Sim 2024+ (Omniverse)
- **Legacy Support**: Backward compatibility with Isaac Gym Preview
- **Conda Support**: Easy environment setup with provided PowerShell scripts

### Logging and Monitoring
- **Wandb Integration**: Experiment tracking with offline mode (handles connection errors gracefully)
- **Error Handling**: Robust error handling for wandb connection issues - training continues even if logging fails
- **Checkpoint Management**: Automatic checkpoint saving with configurable frequency

### Utilities
- **Terrain Generation**: Multiple terrain types (flat, stairs, curbs, gaps, stepping stones)
- **Reward Shaping**: Configurable reward functions for locomotion
- **ONNX Export**: Deploy trained policies to ONNX format
- **Evaluation Scripts**: Comprehensive policy evaluation tools

## Architecture Details

### Phase 1 (Scandots)
- Uses terrain height queries (scandots) as a cheap proxy for depth
- Enables high-throughput training (~200k steps/sec with 4096 parallel envs)
- Trains robust locomotion policies without depth rendering overhead
- Scandots are processed through an MLP to create a latent representation
- Combined with proprioception (48 dims) and commands (3 dims) → 67 dim input to GRU
- **GRU-based recurrent policy** with value head for PPO
- Supports BPTT (Backpropagation Through Time) for temporal credit assignment

### Phase 2 (Depth)
- Distills Phase 1 policy to use real depth images
- Uses DAgger algorithm for sim-to-real transfer
- Processes depth through ConvNet + GRU to estimate terrain geometry
- Depth images (58x87) are processed by convolutional layers
- Student policy initialized with teacher's action MLP weights (GRU trained from scratch)

## Recent Improvements

### Complete PPO Implementation (Latest)
- **Production-Ready PPO**: Full implementation with recurrent policy support
- **GAE (Generalized Advantage Estimation)**: Variance reduction for stable training
- **BPTT Support**: Proper backpropagation through time for GRU policies
- **Value Function Clipping**: Prevents value function explosions
- **Advantage Normalization**: Stabilizes training across different reward scales
- **Learning Rate Scheduling**: Cosine annealing for better convergence
- **AdamW Optimizer**: Better than Adam with weight decay
- **Gradient Clipping**: Prevents gradient explosions
- **Rollout Buffer**: Efficient experience collection with parallel environments
- **Episode Statistics**: Tracks rewards, lengths, and training metrics
- **Automatic Checkpointing**: Periodic saving with full training state
- **Resume Training**: Can resume from checkpoints with optimizer and scheduler state

### Wandb Error Handling
- **Offline Mode**: Wandb now runs in offline mode by default to prevent connection errors
- **Graceful Degradation**: Training continues even if wandb logging fails
- **Error Recovery**: Connection errors (ConnectionResetError, ConnectionError) are caught and handled silently
- **Stabilization Delay**: Added delay after wandb initialization to prevent race conditions
- **Comprehensive Logging**: Tracks policy loss, value loss, explained variance, KL divergence

### Windows Support Enhancements
- **PowerShell Scripts**: Convenient scripts for training (`run_phase1.ps1`, `run_phase2.ps1`)
- **Conda Integration**: Easy environment setup with `setup_conda_env.ps1`
- **Path Handling**: All paths use `pathlib.Path` for cross-platform compatibility
- **CUDA Installation**: Automated PyTorch CUDA installation script

### Code Quality
- **Optional Dependencies**: Graceful fallback when `legged_gym` is not available
- **Device Detection**: Automatic CUDA availability checking with fallback to CPU
- **Checkpoint Management**: Improved checkpoint saving/loading with full training state
- **Environment Compatibility**: Added `reset()` and improved `step()` methods

## Training Tips

### Phase 1 (PPO with Scandots)
1. **Environment Scaling**: Start with 1024-2048 envs for debugging, scale to 4096+ for full training
2. **BPTT Length**: 24 timesteps works well for locomotion (balances memory and credit assignment)
3. **Learning Rate**: Start with 3e-4, use cosine annealing for better convergence
4. **Batch Size**: 64 sequences works well, increase if you have more GPU memory
5. **Checkpointing**: Save every 1M steps to enable recovery from crashes
6. **Monitor Metrics**: Watch explained variance (should be >0.7) and KL divergence (should stay <0.05)
7. **Curriculum**: Monitor terrain success rates to ensure proper progression
8. **Domain Randomization**: Gradually increase randomization ranges during training

### Phase 2 (DAgger with Depth)
1. **Teacher Quality**: Ensure Phase 1 policy is well-trained (mean reward >500)
2. **Initial BC**: Collect 1M samples of behavioral cloning before DAgger iterations
3. **DAgger Iterations**: 100 iterations usually sufficient for good distillation
4. **Depth Resolution**: 58x87 is a good balance between detail and computation

### General Best Practices
1. **Resume Training**: Use `--resume` flag to continue from checkpoints
2. **Wandb Sync**: Sync offline logs with `wandb sync wandb/offline-run-*`
3. **GPU Memory**: Reduce `num_envs` if running out of memory
4. **Training Time**: Phase 1 takes ~24-48 hours on RTX 4080, Phase 2 takes ~12-24 hours

## Troubleshooting

### Windows-Specific Issues

- **Path Issues**: The code uses `pathlib.Path` for Windows-compatible path handling. All paths are automatically handled correctly.
- **Isaac Sim Not Found**: 
  - Make sure Isaac Sim is installed and path is configured
  - Run `.\fix_isaac_sim_path.ps1` to configure paths
  - Check that `omni.isaac.core` can be imported
- **CUDA Issues**: 
  - Ensure you have CUDA 12.x (required for Isaac Sim 2024+)
  - Run `.\install_pytorch_cuda.ps1` to install PyTorch with CUDA support
  - Update GPU drivers if needed
- **Conda Environment**: 
  - Use `.\setup_conda_env.ps1` to set up the environment
  - Activate with `conda activate isaaclab-new` before running scripts

### Wandb Connection Issues

- **Connection Errors**: The code now handles wandb connection errors gracefully:
  - Wandb runs in offline mode by default to avoid connection issues
  - If connection fails, training continues without logging
  - Logs are saved locally in `wandb/` directory
  - Sync logs later with `wandb sync wandb/offline-run-*`
- **Offline Mode**: Wandb is configured to use offline mode automatically. This prevents connection errors during training.

### General Issues

- **Isaac Sim/Isaac Gym Issues**: 
  - For Isaac Sim: Ensure Isaac Sim is properly installed and you're using the correct Python environment
  - For legacy Isaac Gym Preview: Ensure `ISAAC_GYM_PATH` is set correctly
  - The code automatically detects and uses available simulation backend
- **Memory Issues**: 
  - Reduce `num_envs` in config if running out of GPU memory
  - Start with 1024 or 2048 environments for testing
- **Training Instability**: 
  - Check reward scales in config
  - Reduce learning rate if needed
  - Monitor wandb logs for training metrics
- **Import Errors**: 
  - Make sure you're using the correct Python environment (conda or Isaac Sim's Python)
  - Run `python test_imports.py` to verify all imports work
- **Camera/Depth Issues**: 
  - If depth images aren't working, check that `omni.isaac.sensor` is properly installed
  - Verify camera configuration in environment setup

### Running Scripts

**Windows (Recommended)**:
```powershell
# Activate conda environment
conda activate isaaclab-new

# Use PowerShell script
.\run_phase1.ps1 --config configs/phase1_monolithic.yaml --device cuda:0

# Or run Python directly
python train_phase1.py --config configs/phase1_monolithic.yaml --device cuda:0
```

**Linux/Isaac Sim Python**:
```bash
# Use Isaac Sim's Python
/path/to/isaacsim/python.sh train_phase1.py --config configs/phase1_monolithic.yaml
```

For more detailed troubleshooting, see `TROUBLESHOOTING.md` and `WINDOWS_CONDA_SETUP.md`.

For detailed information about the PPO implementation, see `PPO_IMPLEMENTATION.md`.

## Citation

If you use this code, please cite:
```bibtex
@article{agarwal2022legged,
  title={Legged Locomotion in Challenging Terrains using Egocentric Vision},
  author={Agarwal, Ananye and Kumar, Ashish and Malik, Jitendra and Pathak, Deepak},
  journal={CoRL},
  year={2022}
}
```

## License

This implementation is provided for research purposes. Please refer to the original paper and legged_gym repository for licensing information.

## Acknowledgments

This implementation is based on the paper "Legged Locomotion in Challenging Terrains using Egocentric Vision" and uses the legged_gym framework from ETH Zurich's legged robotics lab.
