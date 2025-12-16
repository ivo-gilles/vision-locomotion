# PowerShell script to help set up Isaac Sim Python path
# Usage: .\fix_isaac_sim_path.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Isaac Sim Path Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Isaac Sim exists at default location
$isaacSimPath = "C:\isaacsim"
if (-not (Test-Path $isaacSimPath)) {
    Write-Host "[!] Isaac Sim not found at: $isaacSimPath" -ForegroundColor Yellow
    $customPath = Read-Host "Enter Isaac Sim installation path (or press Enter to skip)"
    if ($customPath -and (Test-Path $customPath)) {
        $isaacSimPath = $customPath
    } else {
        Write-Host "[X] Isaac Sim path not found. Exiting." -ForegroundColor Red
        exit 1
    }
}

Write-Host "[OK] Isaac Sim found at: $isaacSimPath" -ForegroundColor Green
Write-Host ""

# Find Python site-packages
$pythonBat = Join-Path $isaacSimPath "python.bat"
if (-not (Test-Path $pythonBat)) {
    Write-Host "[!] python.bat not found. Trying to locate Python..." -ForegroundColor Yellow
    
    # Try common locations
    $possiblePaths = @(
        (Join-Path $isaacSimPath "python\lib\site-packages"),
        (Join-Path $isaacSimPath "exts\omni.isaac.core\omni\isaac"),
        (Join-Path $isaacSimPath "kit\python\lib\site-packages")
    )
    
    $foundPath = $null
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $foundPath = $path
            Write-Host "[OK] Found Python packages at: $path" -ForegroundColor Green
            break
        }
    }
    
    if (-not $foundPath) {
        Write-Host "[X] Could not locate Isaac Sim Python packages" -ForegroundColor Red
        Write-Host "  -> You may need to run Isaac Sim at least once to initialize" -ForegroundColor Yellow
        exit 1
    }
    
    $sitePackagesPath = $foundPath
} else {
    Write-Host "Finding Isaac Sim Python site-packages..." -ForegroundColor Yellow
    $sitePackagesOutput = & $pythonBat -c "import site; print(site.getsitepackages()[0])" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        $sitePackagesPath = $sitePackagesOutput.Trim()
        Write-Host "[OK] Site-packages found at: $sitePackagesPath" -ForegroundColor Green
    } else {
        Write-Host "[!] Could not query site-packages. Trying default location..." -ForegroundColor Yellow
        $sitePackagesPath = Join-Path $isaacSimPath "python\lib\site-packages"
    }
}

if (-not (Test-Path $sitePackagesPath)) {
    Write-Host "[X] Site-packages path not found: $sitePackagesPath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Testing Isaac Sim imports..." -ForegroundColor Yellow
& $pythonBat -c "from omni.isaac.core import World; print('OK')" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Isaac Sim packages are accessible" -ForegroundColor Green
} else {
    Write-Host "[!] Isaac Sim packages test failed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Options" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To make Isaac Sim packages accessible in your conda environment:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Option 1: Add to PYTHONPATH (temporary, for this session)" -ForegroundColor Cyan
Write-Host "  Run this command:" -ForegroundColor White
Write-Host "  `$env:PYTHONPATH = `"$sitePackagesPath;`$env:PYTHONPATH`"" -ForegroundColor Gray
Write-Host ""
Write-Host "Option 2: Create .pth file (permanent for conda env)" -ForegroundColor Cyan
Write-Host "  This will create a .pth file in your conda environment" -ForegroundColor White
Write-Host ""

$createPth = Read-Host "Create .pth file? (y/n)"
if ($createPth -eq "y" -or $createPth -eq "Y") {
    # Activate conda environment
    conda activate isaaclab-new
    
    # Get conda env site-packages
    $condaSitePackages = python -c "import site; print(site.getsitepackages()[0])" 2>&1
    $condaSitePackages = $condaSitePackages.Trim()
    
    if (Test-Path $condaSitePackages) {
        $pthFile = Join-Path $condaSitePackages "isaacsim.pth"
        $sitePackagesPath | Out-File -FilePath $pthFile -Encoding ASCII
        Write-Host "[OK] Created .pth file at: $pthFile" -ForegroundColor Green
        Write-Host "  -> Isaac Sim packages will be available in your conda environment" -ForegroundColor Gray
    } else {
        Write-Host "[X] Could not find conda site-packages" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Option 3: Use Isaac Sim's Python directly" -ForegroundColor Cyan
Write-Host "  Instead of using conda Python, use Isaac Sim's Python:" -ForegroundColor White
Write-Host "  $pythonBat train_phase1.py --config configs/phase1_monolithic.yaml" -ForegroundColor Gray
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
