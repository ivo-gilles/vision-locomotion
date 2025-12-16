# PowerShell script to run Phase 2 training
# Usage: .\run_phase2.ps1 [--config config_file] [--phase1_checkpoint checkpoint_path] [--device cuda:0] [--resume checkpoint_path]

param(
    [string]$Config = "configs/phase2_monolithic.yaml",
    
    [string]$Phase1Checkpoint = "logs/checkpoints/phase1_final.pth",
    
    [string]$Device = "cuda:0",
    
    [string]$Resume = $null
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Vision-Locomotion Phase 2 Training" -ForegroundColor Cyan
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

# Check if Phase 1 checkpoint exists
if (-not (Test-Path $Phase1Checkpoint)) {
    Write-Host "[X] Phase 1 checkpoint not found: $Phase1Checkpoint" -ForegroundColor Red
    Write-Host "  -> Make sure Phase 1 training is complete first" -ForegroundColor Gray
    Write-Host "  -> Default location: logs/checkpoints/phase1_final.pth" -ForegroundColor Gray
    exit 1
}

# Check if resume checkpoint exists (if provided)
if ($Resume -and -not (Test-Path $Resume)) {
    Write-Host "[!] WARNING: Resume checkpoint not found: $Resume" -ForegroundColor Yellow
    Write-Host "  -> Continuing without resume (starting fresh)" -ForegroundColor Gray
    $Resume = $null
}

Write-Host "Configuration: $Config" -ForegroundColor Cyan
Write-Host "Phase 1 Checkpoint: $Phase1Checkpoint" -ForegroundColor Cyan
Write-Host "Device: $Device" -ForegroundColor Cyan
if ($Resume) {
    Write-Host "Resume from: $Resume" -ForegroundColor Cyan
}
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

# Build command arguments
$argsList = @(
    "train_phase2.py",
    "--config", $Config,
    "--phase1_checkpoint", $Phase1Checkpoint,
    "--device", $Device
)

# Add resume argument if provided
if ($Resume) {
    $argsList += "--resume"
    $argsList += $Resume
}

# Run training
Write-Host "Starting Phase 2 training..." -ForegroundColor Yellow
Write-Host ""
python $argsList

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Phase 2 training completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[X] Training failed. Check error messages above." -ForegroundColor Red
    exit 1
}

