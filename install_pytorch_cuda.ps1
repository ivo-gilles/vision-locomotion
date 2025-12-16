# PowerShell script to install PyTorch with CUDA support
# Usage: .\install_pytorch_cuda.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PyTorch CUDA Installation Script" -ForegroundColor Cyan
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

Write-Host "[OK] Conda environment activated" -ForegroundColor Green
Write-Host ""

# Check CUDA version
Write-Host "Checking CUDA availability..." -ForegroundColor Yellow
$cudaVersion = nvcc --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] CUDA compiler found" -ForegroundColor Green
    Write-Host $cudaVersion
} else {
    Write-Host "[!] CUDA compiler not found in PATH" -ForegroundColor Yellow
    Write-Host "  -> This is OK if CUDA is installed but not in PATH" -ForegroundColor Gray
}

# Check NVIDIA driver
Write-Host ""
Write-Host "Checking NVIDIA driver..." -ForegroundColor Yellow
$nvidiaSmi = nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] NVIDIA driver found" -ForegroundColor Green
    $driverLine = ($nvidiaSmi | Select-String "Driver Version").ToString()
    Write-Host $driverLine
} else {
    Write-Host "[X] NVIDIA driver not found!" -ForegroundColor Red
    Write-Host "  -> Please install NVIDIA drivers first" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Choose installation method:" -ForegroundColor Cyan
Write-Host "1. Conda (recommended)" -ForegroundColor White
Write-Host "2. Pip" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host ""
    Write-Host "Installing via Conda..." -ForegroundColor Yellow
    Write-Host "This will install PyTorch with CUDA 12.1 support" -ForegroundColor Gray
    Write-Host ""
    
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[OK] PyTorch installed via Conda" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "[X] Installation failed" -ForegroundColor Red
        exit 1
    }
} elseif ($choice -eq "2") {
    Write-Host ""
    Write-Host "Installing via Pip..." -ForegroundColor Yellow
    Write-Host "This will install PyTorch with CUDA 12.1 support" -ForegroundColor Gray
    Write-Host ""
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[OK] PyTorch installed via Pip" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "[X] Installation failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[X] Invalid choice" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "Verifying PyTorch CUDA installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] PyTorch CUDA installation verified!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[!] Verification failed - check output above" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
