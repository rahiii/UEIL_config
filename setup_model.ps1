<#
.SYNOPSIS
    Create a conda environment for a model and install all its dependencies.
    Auto-detects GPU and installs the correct PyTorch CUDA version.
.EXAMPLE
    .\setup_model.ps1 raft
    .\setup_model.ps1 voxelmorph -Python 3.11
#>
param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelName,
    [string]$Python = "3.10"
)

$ErrorActionPreference = "Stop"
$ModelDir = Join-Path "models" $ModelName
$SkipPackages = @('pytorch', 'torch', 'torchvision', 'torchaudio', 'cudatoolkit', 'cuda',
                  'cuda-toolkit', 'nvidia-cuda-toolkit')

# ── Preflight ────────────────────────────────────────────
if (-not (Test-Path $ModelDir)) {
    Write-Host "ERROR: '$ModelDir' not found. Clone the model into models/ first." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Setting up environment for: $ModelName ===" -ForegroundColor Cyan

# ── 0. Detect GPU and pick correct CUDA version ─────────
Write-Host "`n[0/7] Detecting GPU..." -ForegroundColor Yellow
$Cuda = "cu128"  # safe default for modern GPUs

try {
    $gpuInfo = nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0 -and $gpuInfo) {
        Write-Host "  GPU detected: $gpuInfo" -ForegroundColor Green
        $capMatch = [regex]::Match($gpuInfo, '(\d+)\.(\d+)')
        if ($capMatch.Success) {
            $major = [int]$capMatch.Groups[1].Value
            $minor = [int]$capMatch.Groups[2].Value
            $cap = $major * 10 + $minor
            Write-Host "  Compute capability: sm_$($major)$($minor)" -ForegroundColor Green

            # sm_120+ (Blackwell / RTX 5000 series) needs CUDA 12.8+
            if ($cap -ge 120) {
                $Cuda = "cu128"
                Write-Host "  -> Blackwell GPU detected, using CUDA 12.8 (cu128)" -ForegroundColor Cyan
            }
            # sm_89-90 (Ada Lovelace / Hopper, RTX 4000 series) works with CUDA 12.4+
            elseif ($cap -ge 89) {
                $Cuda = "cu124"
                Write-Host "  -> Ada/Hopper GPU detected, using CUDA 12.4 (cu124)" -ForegroundColor Cyan
            }
            # sm_80-86 (Ampere, RTX 3000 series)
            elseif ($cap -ge 80) {
                $Cuda = "cu121"
                Write-Host "  -> Ampere GPU detected, using CUDA 12.1 (cu121)" -ForegroundColor Cyan
            }
            # sm_75 (Turing, RTX 2000 series)
            elseif ($cap -ge 75) {
                $Cuda = "cu118"
                Write-Host "  -> Turing GPU detected, using CUDA 11.8 (cu118)" -ForegroundColor Cyan
            }
            else {
                $Cuda = "cu118"
                Write-Host "  -> Older GPU (sm_$($major)$($minor)), using CUDA 11.8 (cu118)" -ForegroundColor DarkYellow
            }
        }
    } else {
        Write-Host "  WARNING: nvidia-smi not found or failed. Defaulting to cu128." -ForegroundColor DarkYellow
    }
} catch {
    Write-Host "  WARNING: Could not detect GPU. Defaulting to cu128." -ForegroundColor DarkYellow
}

Write-Host "  PyTorch CUDA build: $Cuda" -ForegroundColor Green

# ── 1. Create conda env ─────────────────────────────────
Write-Host "`n[1/7] Creating conda environment '$ModelName' (Python $Python)..." -ForegroundColor Yellow

$envExists = conda env list | Select-String "^$ModelName\s"
if ($envExists) {
    Write-Host "  Environment '$ModelName' already exists, skipping creation." -ForegroundColor DarkYellow
} else {
    conda create -n $ModelName python=$Python -y
    if ($LASTEXITCODE -ne 0) { Write-Host "Failed to create env" -ForegroundColor Red; exit 1 }
}

# ── 2. Install PyTorch with correct CUDA ─────────────────
Write-Host "`n[2/7] Installing PyTorch with CUDA ($Cuda)..." -ForegroundColor Yellow
conda run -n $ModelName --no-capture-output pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$Cuda"

# ── 3. Find and install model dependencies ──────────────
Write-Host "`n[3/7] Scanning for model dependencies..." -ForegroundColor Yellow
$depsFound = $false

# requirements.txt (skip docs/ requirements)
$reqFiles = Get-ChildItem -Path $ModelDir -Filter "requirements*.txt" -Recurse -Depth 1 |
    Where-Object { $_.DirectoryName -notmatch '\\docs($|\\)' }
foreach ($f in $reqFiles) {
    Write-Host "  Found: $($f.FullName)" -ForegroundColor Green
    # Filter out torch-related packages from requirements files too
    $tempReq = Join-Path $env:TEMP "filtered_requirements.txt"
    Get-Content $f.FullName | Where-Object {
        $line = $_.Trim()
        if ($line -eq '' -or $line.StartsWith('#')) { return $true }
        $pkg = ($line -split '[=<>!\[;]')[0].Trim().ToLower()
        return ($SkipPackages -notcontains $pkg)
    } | Set-Content -Path $tempReq
    conda run -n $ModelName --no-capture-output pip install -r $tempReq
    Remove-Item $tempReq -ErrorAction SilentlyContinue
    $depsFound = $true
}

# setup.py or pyproject.toml — pip install -e (editable)
$setupPy = Join-Path $ModelDir "setup.py"
$pyproject = Join-Path $ModelDir "pyproject.toml"
if (Test-Path $setupPy) {
    Write-Host "  Found: setup.py — installing in editable mode" -ForegroundColor Green
    conda run -n $ModelName --no-capture-output pip install --no-build-isolation -e $ModelDir
    $depsFound = $true
}
elseif (Test-Path $pyproject) {
    Write-Host "  Found: pyproject.toml — installing in editable mode" -ForegroundColor Green
    conda run -n $ModelName --no-capture-output pip install --no-build-isolation -e $ModelDir
    $depsFound = $true
}

# environment.yml
$envYml = Get-ChildItem -Path $ModelDir -Filter "environment*.yml" -Depth 0
if ($envYml) {
    Write-Host "  Found: $($envYml[0].Name) (extracting pip/conda deps only)" -ForegroundColor Green
    $content = Get-Content $envYml[0].FullName -Raw
    $pipLines = [regex]::Matches($content, '(?m)^\s*-\s+(pip install .+)$')
    foreach ($m in $pipLines) {
        conda run -n $ModelName --no-capture-output pip install $m.Groups[1].Value.Replace("pip install ", "").Split(" ")
    }
    $depsFound = $true
}

# Fallback: parse README for install commands (skip torch-related)
if (-not $depsFound) {
    Write-Host "  No requirements files found, parsing README..." -ForegroundColor DarkYellow

    # Conda/shorthand names that differ from their PyPI package names
    $PipNameMap = @{
        'opencv'      = 'opencv-python'
        'cv2'         = 'opencv-python'
        'sklearn'     = 'scikit-learn'
        'skimage'     = 'scikit-image'
        'pil'         = 'Pillow'
        'yaml'        = 'pyyaml'
        'attr'        = 'attrs'
    }

    $readmes = Get-ChildItem -Path $ModelDir -Filter "README*" -Depth 0
    foreach ($readme in $readmes) {
        $lines = Get-Content $readme.FullName
        foreach ($line in $lines) {
            $trimmed = $line.Trim()
            $pkgStr = $null
            if ($trimmed -match '^pip install\s+(.+)$') {
                $pkgStr = $Matches[1] -replace '#.*$', ''
            } elseif ($trimmed -match '^conda install\s+(.+)$') {
                $pkgStr = $Matches[1] -replace '#.*$', '' -replace '-c\s+\S+', ''
            }

            if ($pkgStr) {
                $mapped = ($pkgStr.Split(" ") | Where-Object { $_.Trim() -ne '' } | ForEach-Object {
                    $base = $_ -replace '[=<>!\[].*$', ''
                    if ($SkipPackages -contains $base.ToLower()) { return }
                    # Map conda/shorthand names to PyPI names
                    if ($PipNameMap.ContainsKey($base.ToLower())) {
                        $_ -replace "^[^=<>!\[]+", $PipNameMap[$base.ToLower()]
                    } else { $_ }
                }) -join " "

                if ($mapped.Trim()) {
                    Write-Host "  README -> pip install $mapped" -ForegroundColor Green
                    conda run -n $ModelName --no-capture-output pip install $mapped.Split(" ")
                }
            }
        }
    }
}

# ── 4. Install framework dependencies ───────────────────
Write-Host "`n[4/7] Installing framework dependencies..." -ForegroundColor Yellow
conda run -n $ModelName --no-capture-output pip install pyyaml scipy opencv-python matplotlib tensorboard

# ── 5. Verify GPU + CUDA compatibility ──────────────────
Write-Host "`n[5/7] Verifying GPU and CUDA compatibility..." -ForegroundColor Yellow
$verifyScript = Join-Path $env:TEMP "verify_gpu.py"
Set-Content -Path $verifyScript -Value @'
import sys
import torch

print('  Python version: ', sys.version.split()[0])
print('  PyTorch version:', torch.__version__)
print('  CUDA built with:', torch.version.cuda)
print('  CUDA available: ', torch.cuda.is_available())

if not torch.cuda.is_available():
    print('  ERROR: CUDA not available! Check your GPU driver and PyTorch build.')
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    print('  GPU %d: %s (sm_%d%d)' % (i, name, cap[0], cap[1]))

# Actual GPU test — this catches sm_XX mismatch errors
try:
    x = torch.randn(4, 4, device='cuda')
    y = x @ x.T
    assert y.shape == (4, 4)
    print('  GPU compute test: PASSED')
except Exception as e:
    print('  GPU compute test: FAILED — %s' % e)
    print('  This usually means PyTorch CUDA version does not support your GPU.')
    print('  Try reinstalling with a newer CUDA build.')
    sys.exit(1)
'@

conda run -n $ModelName --no-capture-output python $verifyScript
$gpuOk = $LASTEXITCODE -eq 0
Remove-Item $verifyScript -ErrorAction SilentlyContinue

if (-not $gpuOk) {
    Write-Host "`n  GPU verification FAILED. Attempting to fix..." -ForegroundColor Red
    Write-Host "  Trying PyTorch with cu128 (CUDA 12.8)..." -ForegroundColor Yellow
    conda run -n $ModelName --no-capture-output pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu128" --force-reinstall

    # Re-verify
    $verifyScript2 = Join-Path $env:TEMP "verify_gpu2.py"
    Set-Content -Path $verifyScript2 -Value @'
import torch, sys
try:
    x = torch.randn(4, 4, device='cuda')
    y = x @ x.T
    print('  GPU compute test after fix: PASSED')
    print('  PyTorch version:', torch.__version__)
    print('  CUDA version:   ', torch.version.cuda)
except Exception as e:
    print('  GPU compute test after fix: STILL FAILED — %s' % e)
    sys.exit(1)
'@
    conda run -n $ModelName --no-capture-output python $verifyScript2
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Could not auto-fix. You may need to install PyTorch nightly." -ForegroundColor Red
    } else {
        Write-Host "  Fixed! Re-installed PyTorch with cu128." -ForegroundColor Green
    }
    Remove-Item $verifyScript2 -ErrorAction SilentlyContinue
}

# ── 6. Re-install model deps that PyTorch reinstall may have clobbered
if (-not $gpuOk) {
    Write-Host "`n[6/7] Re-installing model dependencies after PyTorch fix..." -ForegroundColor Yellow
    foreach ($f in $reqFiles) {
        conda run -n $ModelName --no-capture-output pip install -r $f.FullName 2>$null
    }
    if (Test-Path $setupPy) {
        conda run -n $ModelName --no-capture-output pip install --no-build-isolation -e $ModelDir 2>$null
    } elseif (Test-Path $pyproject) {
        conda run -n $ModelName --no-capture-output pip install --no-build-isolation -e $ModelDir 2>$null
    }
    conda run -n $ModelName --no-capture-output pip install pyyaml scipy opencv-python matplotlib tensorboard
} else {
    Write-Host "`n[6/7] Skipping (no fix needed)." -ForegroundColor DarkGray
}

# ── 7. Summary ──────────────────────────────────────────
Write-Host "`n[7/7] Done!" -ForegroundColor Green
Write-Host ""
Write-Host "  Environment:  $ModelName" -ForegroundColor Cyan
Write-Host "  Activate:     conda activate $ModelName" -ForegroundColor Cyan
Write-Host "  Train:        python train.py" -ForegroundColor Cyan
Write-Host ""
