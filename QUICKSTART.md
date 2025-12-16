# Quick Start Guide

This guide will help you get started with training vision-based legged locomotion policies.

## Prerequisites Checklist

- [ ] NVIDIA GPU with CUDA support
- [ ] Python 3.8+ installed
- [ ] Isaac Gym Preview downloaded and extracted
- [ ] legged_gym repository cloned

## Step 1: Environment Setup

```bash
# 1. Set Isaac Gym path
export ISAAC_GYM_PATH=/path/to/isaacgym

# 2. Clone legged_gym (if not already done)
git clone https://github.com/leggedrobotics/legged_gym.git
cd legged_gym
pip install -e .
cd ..

# 3. Install this package
cd vision-locomotion
pip install -r requirements.txt
```

## Step 2: Verify Installation

```bash
# Test that Isaac Gym is accessible
python -c "from isaacgym import gymapi; print('Isaac Gym OK')"

# Test that legged_gym is accessible
python -c "from legged_gym.envs import A1RoughCfg; print('legged_gym OK')"
```

## Step 3: Phase 1 Training (Scandots)

Start with a small test run:

```bash
# Edit configs/phase1_monolithic.yaml to reduce num_envs for testing
# Change num_envs: 4096 to num_envs: 64

# Run Phase 1 training
python train_phase1.py --config configs/phase1_monolithic.yaml --device cuda:0
```

**Expected Output:**
- Training should start without errors
- You should see reward metrics being logged
- Checkpoints will be saved in `logs/checkpoints/`

**Training Time:**
- Full training: ~13 hours with 4096 envs on RTX 3090
- Test run: ~30 minutes with 64 envs

## Step 4: Phase 2 Training (Depth)

After Phase 1 completes:

```bash
# Find your Phase 1 checkpoint
ls logs/checkpoints/phase1_final.pth

# Run Phase 2 training
python train_phase2.py \
    --config configs/phase2_monolithic.yaml \
    --phase1_checkpoint logs/checkpoints/phase1_final.pth \
    --device cuda:0
```

**Expected Output:**
- Phase 2 training with depth rendering
- Slower than Phase 1 (~500 steps/sec vs ~200k steps/sec)
- Final checkpoint saved as `phase2_final.pth`

## Step 5: Evaluation

Test your trained policy:

```bash
python scripts/evaluate.py \
    --checkpoint logs/checkpoints/phase2_final.pth \
    --phase 2 \
    --num_episodes 10
```

## Common Issues and Solutions

### Issue: "Isaac Gym not found"
**Solution:** Ensure `ISAAC_GYM_PATH` is set correctly:
```bash
export ISAAC_GYM_PATH=/path/to/isaacgym
```

### Issue: "CUDA out of memory"
**Solution:** Reduce `num_envs` in config file:
```yaml
env:
  num_envs: 256  # Reduce from 4096
```

### Issue: "legged_gym import error"
**Solution:** Make sure legged_gym is installed:
```bash
cd legged_gym
pip install -e .
```

### Issue: Training is very slow
**Solution:** 
- Check GPU utilization: `nvidia-smi`
- Reduce `num_envs` if memory constrained
- Ensure using GPU: `--device cuda:0`

## Next Steps

1. **Experiment with architectures**: Try RMA vs Monolithic
2. **Tune hyperparameters**: Adjust learning rates, reward scales
3. **Modify terrain**: Add custom terrain types
4. **Deploy to hardware**: Export to ONNX and deploy on robot

## Tips for Success

1. **Start small**: Begin with fewer environments to debug
2. **Monitor metrics**: Watch reward curves and success rates
3. **Save frequently**: Checkpoints are your friend
4. **Use curriculum**: Let the system automatically progress difficulty
5. **Domain randomization**: Don't skip it - crucial for sim-to-real

## Getting Help

- Check the main README.md for detailed documentation
- Review config files for all available options
- Examine the paper for theoretical background

Good luck with your training!
