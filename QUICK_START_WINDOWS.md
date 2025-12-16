# Quick Start Guide - Windows with Conda

**Quick setup for running vision-locomotion on Windows with conda environment `isaaclab-new` and Isaac Sim at `c:/isaacsim/`**

## 🚀 Quick Setup (3 Steps)

### Step 1: Run Setup Script
```powershell
cd C:\Users\tindu\Desktop\isaaclab\vision-locomotion
.\setup_conda_env.ps1
```

This will:
- Activate your `isaaclab-new` conda environment
- Install all Python dependencies
- Attempt to link Isaac Sim packages
- Run import tests

### Step 2: Verify Installation
```powershell
conda activate isaaclab-new
python test_imports.py
```

Check that:
- ✓ PyTorch and CUDA work
- ✓ Isaac Sim packages are accessible (may show warnings - see troubleshooting)
- ✓ Project modules can be imported

### Step 3: Run Training
```powershell
# Option A: Use the run script
.\run_phase1.ps1

# Option B: Run manually
conda activate isaaclab-new
python train_phase1.py --config configs/phase1_monolithic.yaml --device cuda:0
```

## ⚙️ Manual Setup (If Script Fails)

### 1. Install Dependencies
```powershell
conda activate isaaclab-new
cd C:\Users\tindu\Desktop\isaaclab\vision-locomotion
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Link Isaac Sim Packages

**Option A: Add to PYTHONPATH (Temporary)**
```powershell
$env:PYTHONPATH = "C:\isaacsim\python\lib\site-packages;$env:PYTHONPATH"
```

**Option B: Use Isaac Sim's Python**
```powershell
c:\isaacsim\python.bat -m pip install -r requirements.txt
c:\isaacsim\python.bat -m pip install -e .
c:\isaacsim\python.bat train_phase1.py --config configs/phase1_monolithic.yaml
```

### 3. Test
```powershell
python test_imports.py
```

## 🔧 Common Issues

### "Torch not compiled with CUDA enabled"
**Solution:** Install PyTorch with CUDA support:
```powershell
.\install_pytorch_cuda.ps1
```
Or see `TROUBLESHOOTING.md` for manual installation steps.

### "omni.isaac.core not found"
**Solution:** Link Isaac Sim packages:
```powershell
.\fix_isaac_sim_path.ps1
```
Or see `TROUBLESHOOTING.md` for manual methods.

### "CUDA out of memory"
**Solution:** Edit `configs/phase1_monolithic.yaml` and change `num_envs: 4096` to `num_envs: 64`

### "legged_gym not found"
**Solution:** This is optional. The code works without it, but install if needed:
```powershell
cd C:\Users\tindu\Desktop
git clone https://github.com/leggedrobotics/legged_gym.git
cd legged_gym
pip install -e .
```

**For more issues, see `TROUBLESHOOTING.md`**

## 📝 Configuration

Before full training, reduce environments for testing:
- Edit `configs/phase1_monolithic.yaml`
- Change `num_envs: 4096` → `num_envs: 64`

## 🎯 Training Commands

### Phase 1 (Scandots)
```powershell
conda activate isaaclab-new
python train_phase1.py --config configs/phase1_monolithic.yaml --device cuda:0
```

### Phase 2 (Depth)
```powershell
python train_phase2.py `
    --config configs/phase2_monolithic.yaml `
    --phase1_checkpoint logs/checkpoints/phase1_final.pth `
    --device cuda:0
```

### Evaluation
```powershell
python scripts/evaluate.py `
    --checkpoint logs/checkpoints/phase2_final.pth `
    --phase 2 `
    --num_episodes 10
```

## 📚 More Information

- **Detailed Setup:** See `WINDOWS_CONDA_SETUP.md`
- **Troubleshooting:** See `TROUBLESHOOTING.md` for common issues and solutions
- **General Guide:** See `README.md` and `QUICKSTART.md`
- **Configuration:** See files in `configs/` directory

## 🛠️ Helper Scripts

- **`setup_conda_env.ps1`** - Automated setup of conda environment
- **`install_pytorch_cuda.ps1`** - Install PyTorch with CUDA support
- **`fix_isaac_sim_path.ps1`** - Link Isaac Sim packages to conda environment
- **`run_phase1.ps1`** - Run Phase 1 training with checks
- **`test_imports.py`** - Verify all dependencies are installed

## ✅ Checklist

- [ ] Conda environment `isaaclab-new` activated
- [ ] Dependencies installed (`requirements.txt`)
- [ ] Project installed (`pip install -e .`)
- [ ] Isaac Sim packages accessible (check with `test_imports.py`)
- [ ] Config file edited (reduce `num_envs` for testing)
- [ ] GPU available and CUDA working
- [ ] Ready to train!

---

**Need help?** Check `WINDOWS_CONDA_SETUP.md` for detailed troubleshooting.
