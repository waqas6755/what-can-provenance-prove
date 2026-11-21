# Paper 3 GPU kit - environment setup ONLY (for G5), Windows (RTX 3070 Laptop, 8 GB).
# Run from PowerShell in this folder:
#     powershell -ExecutionPolicy Bypass -File .\setup_g5_env.ps1
# It creates a private Python environment in .venv (nothing is installed system-wide),
# installs PyTorch with CUDA plus the libraries used for G1-G4, and runs a 1-minute self-test.
# It runs no experiment. Skip it if .venv already exists from the G1-G4 run.

$ErrorActionPreference = "Continue"   # native tools (pip) write progress to stderr; exit codes are checked explicitly
Set-Location $PSScriptRoot
Start-Transcript -Path "$PSScriptRoot\setup_g5_transcript.txt" -Append | Out-Null

function Test-Py([string[]]$cmd) {
    # True if the interpreter runs and is Python 3.10-3.14. All output is discarded so that
    # messages such as "Python 3.12 not found!" cannot leak into the return value.
    try {
        $exe = $cmd[0]; $rest = @(); if ($cmd.Length -gt 1) { $rest = $cmd[1..($cmd.Length-1)] }
        & $exe @rest -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 5)" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Find-Python {
    # Prefer the versions that the G1-G4 libraries support best; 3.14 is the last resort.
    foreach ($v in @("3.11", "3.12", "3.13", "3.10")) { if (Test-Py @("py", "-$v")) { return ,@("py", "-$v") } }
    $mf = Join-Path $env:USERPROFILE "miniforge3\python.exe"
    if ((Test-Path $mf) -and (Test-Py @($mf))) { return ,@($mf) }
    if (Test-Py @("python")) { return ,@("python") }
    if (Test-Py @("py", "-3.14")) { return ,@("py", "-3.14") }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.10-3.14 not found. Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run this script again." -ForegroundColor Red
    Stop-Transcript | Out-Null; exit 1
}
Write-Host "Using Python: $($py -join ' ')"
& $py[0] @($py | Select-Object -Skip 1) --version

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if ($py.Length -gt 1) { & $py[0] $py[1] -m venv .venv } else { & $py[0] -m venv .venv }
}
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Could not create .venv with $($py -join ' '). Send setup_g5_transcript.txt" -ForegroundColor Red
    Stop-Transcript | Out-Null; exit 1
}
$vpy = "$PSScriptRoot\.venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip

# PyTorch with CUDA: try recent CUDA builds until one gives a working GPU
$ok = $false
# First choice: the exact build used for G1-G4 (torch 2.6.0 + CUDA 12.4), so G3 runs can replay bit for bit.
foreach ($spec in @(@("torch==2.6.0", "cu124"), @("torch", "cu128"), @("torch", "cu126"), @("torch", "cu124"))) {
    $cu = $spec[1]
    Write-Host "Installing PyTorch ($($spec[0]), $cu)..."
    & $vpy -m pip install $spec[0] --index-url "https://download.pytorch.org/whl/$cu"
    if ($LASTEXITCODE -eq 0) {
        & $vpy -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        & $vpy -m pip uninstall -y torch
    }
}
if (-not $ok) {
    Write-Host "PyTorch could not see the GPU. Update the NVIDIA driver (GeForce Experience or nvidia.com), reboot, and rerun." -ForegroundColor Red
    Stop-Transcript | Out-Null; exit 1
}
& $vpy -m pip install "transformers==5.22.1" "peft==0.21.2" "datasets==5.1.0" numpy cryptography
if ($LASTEXITCODE -ne 0) { Write-Host "Library install failed; see setup_g5_transcript.txt" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
& $vpy -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| CUDA', torch.version.cuda)"

Write-Host "Self-test (tiny models, about 1 minute)..."
& $vpy p3gpu_g5.py --smoke
if ($LASTEXITCODE -ne 0) { Write-Host "Self-test failed; send results_g5_smoke\log.txt" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
Write-Host "Environment ready. Now double-click run_g5.bat" -ForegroundColor Green
Stop-Transcript | Out-Null
