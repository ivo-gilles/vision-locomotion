# PowerShell script to run Phase 1 training
# Usage: .\run_phase1.ps1 [--config config_file] [--device cuda:0]

param(
    [string]$Config = "configs/phase1_monolithic.yaml",
    [string]$Device = "cuda:0"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Vision-Locomotion Phase 1 Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate conda environment
$envName = "isaaclab-new"
Write-Host "Activating conda environment: $envName" -ForegroundColor Yellow
conda activate $envName

if ($LASTEXITCODE -ne 0) {
    Write-Host "[X] Failed to activate conda environment: $envName" -ForegroundColor Red
    exit 1
}

# Get project directory
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

# Add Isaac Sim to PYTHONPATH if it exists
$isaacSimPath = "C:\isaacsim\python\lib\site-packages"
if (Test-Path $isaacSimPath) {
    if ($env:PYTHONPATH) {
        $env:PYTHONPATH = "$isaacSimPath;$env:PYTHONPATH"
    } else {
        $env:PYTHONPATH = $isaacSimPath
    }
}

# Check if config file exists
if (-not (Test-Path $Config)) {
    Write-Host "[X] Config file not found: $Config" -ForegroundColor Red
    exit 1
}

Write-Host "Configuration: $Config" -ForegroundColor Cyan
Write-Host "Device: $Device" -ForegroundColor Cyan
Write-Host ""

# Check CUDA availability if CUDA device requested
if ($Device -like "cuda*") {
    Write-Host "Checking CUDA availability..." -ForegroundColor Yellow
    $cudaCheck = python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>&1
    Write-Host $cudaCheck
    
    if ($cudaCheck -match "CUDA available: False") {
        Write-Host ""
        Write-Host "[!] WARNING: CUDA requested but not available!" -ForegroundColor Yellow
        Write-Host "  -> PyTorch was not compiled with CUDA support" -ForegroundColor Gray
        Write-Host "  -> Run .\install_pytorch_cuda.ps1 to install PyTorch with CUDA" -ForegroundColor Gray
        Write-Host "  -> Or use --device cpu to run on CPU (very slow)" -ForegroundColor Gray
        Write-Host ""
        $continue = Read-Host "Continue anyway? (y/n)"
        if ($continue -ne "y" -and $continue -ne "Y") {
            exit 1
        }
    }
}

Write-Host ""

# Run training
Write-Host "Starting Phase 1 training..." -ForegroundColor Yellow
Write-Host ""
python train_phase1.py --config $Config --device $Device

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Training completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[X] Training failed. Check error messages above." -ForegroundColor Red
    exit 1
}
