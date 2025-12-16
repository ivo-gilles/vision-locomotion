# Windows Setup Guide for Vision-Locomotion with Conda Environment

This guide will help you set up and run the vision-locomotion code on Windows using your conda environment `isaaclab-new` with Isaac Sim installed at `c:/isaacsim/`.

## Prerequisites

- ✅ Isaac Sim installed at: `c:/isaacsim/`
- ✅ Conda environment: `isaaclab-new`
- ✅ NVIDIA GPU with CUDA support
- ✅ Windows 10/11

## Step 1: Activate Your Conda Environment

```powershell
# Activate your conda environment
conda activate isaaclab-new

# Verify Python version (should be 3.8+)
python --version
```

## Step 2: Install Core Python Dependencies

Navigate to the project directory and install the required packages:

```powershell
# Navigate to project directory
cd C:\Users\tindu\Desktop\isaaclab\vision-locomotion

# Upgrade pip first
python -m pip install --upgrade pip

# Install core dependencies from requirements.txt
python -m pip install -r requirements.txt

# Install the project in development mode
python -m pip install -e .
```

## Step 3: Install Isaac Sim Python Packages

Isaac Sim packages need to be installed in a way that your conda environment can access them. There are two approaches:

### Option A: Install Isaac Sim Packages in Conda Environment (Recommended)

Isaac Sim packages can be installed directly into your conda environment if they're available on PyPI or if you can access Isaac Sim's Python packages:

```powershell
# Make sure conda environment is activated
conda activate isaaclab-new

# Try installing Isaac Sim packages
# Note: These may need to be installed via Isaac Sim's Python, then linked
python -m pip install omni-isaac-core
python -m pip install omni-isaac-sensor
python -m pip install omni-isaac-gym
```

If the above doesn't work (packages not found), use Option B.

### Option B: Link Isaac Sim Python Packages to Conda Environment

If Isaac Sim packages aren't directly installable, you can add Isaac Sim's Python site-packages to your conda environment's Python path:

1. **Find Isaac Sim's Python site-packages path:**
   ```powershell
   # Run this to find the path
   c:\isaacsim\python.bat -c "import site; print(site.getsitepackages()[0])"
   ```

2. **Create a `.pth` file in your conda environment:**
   ```powershell
   # Get your conda environment's site-packages path
   python -c "import site; print(site.getsitepackages()[0])"
   
   # Create a .pth file (replace with actual paths from above commands)
   # Example: Create isaacsim.pth in your conda env's site-packages
   ```

3. **Or set PYTHONPATH environment variable:**
   ```powershell
   # Add Isaac Sim's Python packages to PYTHONPATH
   $env:PYTHONPATH = "C:\isaacsim\python\lib\site-packages;$env:PYTHONPATH"
   
   # To make it permanent, add to your conda environment activation script
   ```

### Option C: Use Isaac Sim's Python with Conda Packages (Alternative)

If linking is complex, you can install your project packages into Isaac Sim's Python environment:

```powershell
# Install packages using Isaac Sim's Python
c:\isaacsim\python.bat -m pip install -r requirements.txt
c:\isaacsim\python.bat -m pip install -e .
```

Then run scripts using Isaac Sim's Python:
```powershell
c:\isaacsim\python.bat train_phase1.py --config configs/phase1_monolithic.yaml
```

## Step 4: Install legged_gym (Optional but Recommended)

The code can work without `legged_gym`, but full functionality requires it. To install:

```powershell
# Navigate to a directory where you want to clone legged_gym
cd C:\Users\tindu\Desktop

# Clone legged_gym repository
git clone https://github.com/leggedrobotics/legged_gym.git
cd legged_gym

# Install in your conda environment
conda activate isaaclab-new
pip install -e .
```

**Note:** `legged_gym` may require Isaac Gym Preview. If you're using Isaac Sim, the code has been updated to work with Isaac Sim's API, so `legged_gym` may not be strictly necessary but provides useful utilities.

## Step 5: Verify Installation

Create a test script to verify everything works:

```powershell
# Create test_imports.py
@"
import sys
print(f"Python: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch failed: {e}")

# Test Isaac Sim packages
try:
    from omni.isaac.core import World
    print("✓ omni.isaac.core imported")
except Exception as e:
    print(f"✗ omni.isaac.core failed: {e}")

try:
    from omni.isaac.sensor import Camera
    print("✓ omni.isaac.sensor imported")
except Exception as e:
    print(f"✗ omni.isaac.sensor failed: {e}")

# Test legged_gym (optional)
try:
    from legged_gym.envs import A1RoughCfg
    print("✓ legged_gym imported")
except Exception as e:
    print(f"⚠ legged_gym not available (optional): {e}")

# Test project modules
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
"@ | Out-File -FilePath test_imports.py -Encoding utf8

# Run test
python test_imports.py
```

## Step 6: Configure for Testing

Before running full training, reduce the number of environments for testing:

```powershell
# Edit configs/phase1_monolithic.yaml
# Change: num_envs: 4096  →  num_envs: 64  (for testing)
```

You can use a text editor or PowerShell:
```powershell
# Backup original config
Copy-Item configs\phase1_monolithic.yaml configs\phase1_monolithic.yaml.bak

# Edit the file (use your preferred editor)
notepad configs\phase1_monolithic.yaml
# Or use: code configs\phase1_monolithic.yaml
```

## Step 7: Run Training

### Phase 1 Training (Scandots)

```powershell
# Make sure conda environment is activated
conda activate isaaclab-new

# Navigate to project directory
cd C:\Users\tindu\Desktop\isaaclab\vision-locomotion

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

### Phase 2 Training (Depth)

After Phase 1 completes:

```powershell
# Run Phase 2 training
python train_phase2.py `
    --config configs/phase2_monolithic.yaml `
    --phase1_checkpoint logs/checkpoints/phase1_final.pth `
    --device cuda:0
```

### Evaluation

Test your trained policy:

```powershell
python scripts/evaluate.py `
    --checkpoint logs/checkpoints/phase2_final.pth `
    --phase 2 `
    --num_episodes 10
```

## Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'omni.isaac.core'"

**Solution:** Isaac Sim packages aren't accessible from your conda environment.

**Option A:** Add Isaac Sim's site-packages to PYTHONPATH:
```powershell
# Find Isaac Sim's site-packages
c:\isaacsim\python.bat -c "import site; print(site.getsitepackages()[0])"

# Add to PYTHONPATH (replace with actual path from above)
$env:PYTHONPATH = "C:\isaacsim\python\lib\site-packages;$env:PYTHONPATH"

# Or create a .pth file in your conda env's site-packages
```

**Option B:** Use Isaac Sim's Python directly:
```powershell
c:\isaacsim\python.bat train_phase1.py --config configs/phase1_monolithic.yaml
```

### Issue 2: "ModuleNotFoundError: No module named 'legged_gym'"

**Solution:** Install legged_gym (see Step 4) or the code will work with limited functionality. The code is designed to work without `legged_gym`, but some features may be limited.

### Issue 3: "CUDA out of memory"

**Solution:** Reduce `num_envs` in config files:
```yaml
env:
  num_envs: 64  # Reduce from 4096
```

### Issue 4: "Wrong Python interpreter"

**Solution:** Make sure you're using the correct Python:
```powershell
# Check which Python you're using
where python

# Should point to your conda environment
# Example: C:\Users\tindu\anaconda3\envs\isaaclab-new\python.exe

# If not, activate conda environment
conda activate isaaclab-new
```

### Issue 5: Path issues on Windows

**Solution:** The code uses `pathlib.Path` which should handle Windows paths. If issues persist:
- Use forward slashes: `logs/checkpoints/model.pth`
- Or use raw strings: `r"C:\path\to\file"`

### Issue 6: Isaac Sim packages not found even with PYTHONPATH

**Solution:** Some Isaac Sim packages may need to be imported differently or may require additional setup. Try:

```powershell
# Verify Isaac Sim installation
c:\isaacsim\python.bat -c "from omni.isaac.core import World; print('OK')"

# If that works, the issue is linking. Try creating a startup script:
```

Create `run_with_isaac.py`:
```python
import sys
import os

# Add Isaac Sim's site-packages
isaac_sim_path = r"C:\isaacsim\python\lib\site-packages"
if os.path.exists(isaac_sim_path):
    sys.path.insert(0, isaac_sim_path)

# Now import and run your script
if __name__ == "__main__":
    import train_phase1
    train_phase1.main()
```

## Quick Start Script

Create a PowerShell script to automate setup:

```powershell
# Create setup_and_run.ps1
@"
# Activate conda environment
conda activate isaaclab-new

# Navigate to project
cd C:\Users\tindu\Desktop\isaaclab\vision-locomotion

# Add Isaac Sim to PYTHONPATH (adjust path as needed)
$isaacSimPath = "C:\isaacsim\python\lib\site-packages"
if (Test-Path $isaacSimPath) {
    $env:PYTHONPATH = "$isaacSimPath;$env:PYTHONPATH"
}

# Run training
python train_phase1.py --config configs/phase1_monolithic.yaml --device cuda:0
"@ | Out-File -FilePath setup_and_run.ps1 -Encoding utf8

# Run it
.\setup_and_run.ps1
```

## Alternative: Using Isaac Sim's Python Directly

If linking packages is too complex, you can use Isaac Sim's Python directly:

```powershell
# Install packages in Isaac Sim's Python environment
c:\isaacsim\python.bat -m pip install -r requirements.txt
c:\isaacsim\python.bat -m pip install -e .

# Run scripts with Isaac Sim's Python
cd C:\Users\tindu\Desktop\isaaclab\vision-locomotion
c:\isaacsim\python.bat train_phase1.py --config configs/phase1_monolithic.yaml --device cuda:0
```

This approach uses Isaac Sim's Python environment but allows you to manage your project dependencies separately.

## Next Steps

1. **Start with a small test run** (64 envs) to verify everything works
2. **Monitor GPU usage** with `nvidia-smi` in a separate terminal
3. **Check logs** in `./logs/` directory
4. **Gradually increase `num_envs`** as you verify everything works
5. **Experiment with architectures**: Try RMA vs Monolithic
6. **Tune hyperparameters**: Adjust learning rates, reward scales

## Additional Resources

- Main README: `README.md`
- Quick Start Guide: `QUICKSTART.md`
- Configuration files: `configs/`
- Isaac Sim documentation: https://docs.omniverse.nvidia.com/app_isaacsim/

Good luck with your training!
