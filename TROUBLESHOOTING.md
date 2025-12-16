# Troubleshooting Guide

Common issues and solutions for running vision-locomotion on Windows.

## Issue 1: "Torch not compiled with CUDA enabled"

**Error:**
```
AssertionError: Torch not compiled with CUDA enabled
```

**Solution:**
Your PyTorch installation doesn't have CUDA support. Install PyTorch with CUDA:

```powershell
# Option 1: Use the automated script
.\install_pytorch_cuda.ps1

# Option 2: Manual installation via Conda (recommended)
conda activate isaaclab-new
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Option 3: Manual installation via Pip
conda activate isaaclab-new
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify installation:**
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

**Note:** If you don't have a CUDA-capable GPU or want to test on CPU, use `--device cpu`:
```powershell
python train_phase1.py --config configs/phase1_monolithic.yaml --device cpu
```

---

## Issue 2: "Neither Isaac Sim nor Isaac Gym Preview found!"

**Error:**
```
Warning: Neither Isaac Sim nor Isaac Gym Preview found!
```

**Solution:**
Isaac Sim packages aren't accessible from your conda environment. Use one of these methods:

### Method 1: Link Isaac Sim packages (Recommended)

```powershell
# Run the helper script
.\fix_isaac_sim_path.ps1

# Or manually add to PYTHONPATH
$env:PYTHONPATH = "C:\isaacsim\python\lib\site-packages;$env:PYTHONPATH"
```

### Method 2: Create .pth file (Permanent)

```powershell
conda activate isaaclab-new
$condaPath = python -c "import site; print(site.getsitepackages()[0])"
$isaacPath = "C:\isaacsim\python\lib\site-packages"
"$isaacPath" | Out-File -FilePath "$condaPath\isaacsim.pth" -Encoding ASCII
```

### Method 3: Use Isaac Sim's Python directly

```powershell
# Install packages in Isaac Sim's Python environment
c:\isaacsim\python.bat -m pip install -r requirements.txt
c:\isaacsim\python.bat -m pip install -e .

# Run scripts with Isaac Sim's Python
c:\isaacsim\python.bat train_phase1.py --config configs/phase1_monolithic.yaml
```

**Verify:**
```powershell
python -c "from omni.isaac.core import World; print('OK')"
```

---

## Issue 3: "legged_gym not found"

**Error:**
```
Warning: legged_gym not found. Some features may not work.
```

**Solution:**
This is **optional**. The code can work without `legged_gym`, but full functionality requires it.

To install:
```powershell
cd C:\Users\tindu\Desktop
git clone https://github.com/leggedrobotics/legged_gym.git
cd legged_gym
conda activate isaaclab-new
pip install -e .
```

**Note:** `legged_gym` may require Isaac Gym Preview (legacy). If you're using Isaac Sim, the code has been updated to work with Isaac Sim's API, so `legged_gym` may not be strictly necessary.

---

## Issue 4: "CUDA out of memory"

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
Reduce the number of parallel environments in the config file:

1. Edit `configs/phase1_monolithic.yaml`
2. Change `num_envs: 4096` to a smaller value:
   - For testing: `num_envs: 64`
   - For RTX 3090: `num_envs: 2048`
   - For RTX 4090: `num_envs: 4096`

---

## Issue 5: "Gym has been unmaintained" warning

**Warning:**
```
Gym has been unmaintained since 2022 and does not support NumPy 2.0
Please upgrade to Gymnasium
```

**Solution:**
This is a deprecation warning. The code should still work, but you can:

1. **Ignore it** - The code will work with the current `gym` package
2. **Upgrade to Gymnasium** (if dependencies support it):
   ```powershell
   pip install gymnasium
   # Then update imports in code from 'import gym' to 'import gymnasium as gym'
   ```

---

## Issue 6: Wrong Python interpreter

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
Make sure you're using the correct Python from your conda environment:

```powershell
# Check which Python you're using
where python

# Should show: C:\Users\tindu\.conda\envs\isaaclab-new\python.exe
# If not, activate conda environment:
conda activate isaaclab-new
```

---

## Issue 7: Path issues on Windows

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**
The code uses `pathlib.Path` which should handle Windows paths. If issues persist:

- Use forward slashes: `logs/checkpoints/model.pth`
- Or use raw strings: `r"C:\path\to\file"`
- Use relative paths when possible

---

## Issue 8: Isaac Sim not found at expected location

**Error:**
```
Isaac Sim not found at: C:\isaacsim
```

**Solution:**
If Isaac Sim is installed elsewhere:

1. Update the path in scripts:
   - `setup_conda_env.ps1` - line 88
   - `run_phase1.ps1` - line 29
   - `fix_isaac_sim_path.ps1` - line 8

2. Or use the interactive script:
   ```powershell
   .\fix_isaac_sim_path.ps1
   ```

---

## Quick Diagnostic Commands

Run these to diagnose issues:

```powershell
# 1. Check Python and environment
conda activate isaaclab-new
python --version
where python

# 2. Check PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# 3. Check Isaac Sim
c:\isaacsim\python.bat -c "from omni.isaac.core import World; print('OK')"

# 4. Check project imports
python test_imports.py

# 5. Check GPU
nvidia-smi
```

---

## Getting More Help

1. **Check logs:** Look in `./logs/` directory for detailed error messages
2. **Review configuration:** Check `configs/` files for settings
3. **Test imports:** Run `python test_imports.py` to see what's missing
4. **Check documentation:** See `WINDOWS_CONDA_SETUP.md` for detailed setup instructions

---

## Common Setup Checklist

Before running training, verify:

- [ ] Conda environment `isaaclab-new` is activated
- [ ] PyTorch installed with CUDA support (if using GPU)
- [ ] Isaac Sim packages accessible (check with `test_imports.py`)
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Project installed (`pip install -e .`)
- [ ] Config file edited (reduce `num_envs` for testing)
- [ ] GPU available and drivers installed (if using CUDA)

Run this to check everything:
```powershell
python test_imports.py
```
