# PowerShell script to set up conda environment for vision-locomotion
# Usage: .\setup_conda_env.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Vision-Locomotion Conda Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
try {
    $condaVersion = conda --version
    Write-Host "[OK] Conda found: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "[X] Conda not found! Please install Anaconda or Miniconda." -ForegroundColor Red
    exit 1
}

# Activate conda environment
$envName = "isaaclab-new"
Write-Host "Activating conda environment: $envName" -ForegroundColor Yellow
conda activate $envName

if ($LASTEXITCODE -ne 0) {
    Write-Host "[X] Failed to activate conda environment: $envName" -ForegroundColor Red
    Write-Host "  -> Create it first: conda create -n isaaclab-new python=3.9" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Conda environment activated" -ForegroundColor Green
Write-Host ""

# Get project directory
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir
Write-Host "Project directory: $projectDir" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version
Write-Host "[OK] $pythonVersion" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "[OK] pip upgraded" -ForegroundColor Green

# Install requirements
Write-Host ""
Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Yellow
python -m pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Requirements installed" -ForegroundColor Green
} else {
    Write-Host "[X] Failed to install requirements" -ForegroundColor Red
    exit 1
}

# Install project in development mode
Write-Host ""
Write-Host "Installing project in development mode..." -ForegroundColor Yellow
python -m pip install -e .
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Project installed" -ForegroundColor Green
} else {
    Write-Host "[X] Failed to install project" -ForegroundColor Red
    exit 1
}

# Try to install Isaac Sim packages
Write-Host ""
Write-Host "Attempting to install Isaac Sim packages..." -ForegroundColor Yellow
Write-Host "  (These may not be available via pip - see WINDOWS_CONDA_SETUP.md for alternatives)" -ForegroundColor Gray

$isaacPackages = @("omni-isaac-core", "omni-isaac-sensor", "omni-isaac-gym")
foreach ($package in $isaacPackages) {
    python -m pip install $package 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] $package installed" -ForegroundColor Green
    } else {
        Write-Host "[!] $package not available via pip" -ForegroundColor Yellow
        Write-Host "  -> You may need to link Isaac Sim packages manually (see WINDOWS_CONDA_SETUP.md)" -ForegroundColor Gray
    }
}

# Add Isaac Sim to PYTHONPATH if it exists
$isaacSimPath = "C:\isaacsim\python\lib\site-packages"
if (Test-Path $isaacSimPath) {
    Write-Host ""
    Write-Host "Found Isaac Sim at: $isaacSimPath" -ForegroundColor Cyan
    Write-Host "Adding to PYTHONPATH for this session..." -ForegroundColor Yellow
    
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$isaacSimPath;$env:PYTHONPATH"
    } else {
        $env:PYTHONPATH = $isaacSimPath
    }
    Write-Host "[OK] PYTHONPATH updated" -ForegroundColor Green
    Write-Host "  Note: This is temporary. To make permanent, add to your conda env activation script" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "[!] Isaac Sim not found at expected location: $isaacSimPath" -ForegroundColor Yellow
    Write-Host "  -> Update the path in this script if Isaac Sim is installed elsewhere" -ForegroundColor Gray
}

# Run test script
Write-Host ""
Write-Host "Running import tests..." -ForegroundColor Yellow
Write-Host ""
python test_imports.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review test_imports.py output above" -ForegroundColor White
Write-Host "2. If Isaac Sim packages are missing, see WINDOWS_CONDA_SETUP.md" -ForegroundColor White
Write-Host "3. Edit configs/phase1_monolithic.yaml to reduce num_envs for testing" -ForegroundColor White
Write-Host "4. Run: python train_phase1.py --config configs/phase1_monolithic.yaml --device cuda:0" -ForegroundColor White
Write-Host ""
