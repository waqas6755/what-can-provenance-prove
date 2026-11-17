# Paper 3 GPU kit - one-command runner for Windows (RTX 3070 Laptop, 8 GB).
# Run from PowerShell in this folder:
#     powershell -ExecutionPolicy Bypass -File .\run_gpu.ps1
# It creates a private Python environment in .venv (nothing is installed system-wide),
# installs PyTorch with CUDA plus three libraries, runs a 1-minute self-test, then the
# experiments (about 2-3 hours), and writes results_gpu.zip next to this script.

$ErrorActionPreference = "Continue"   # native tools (pip) write progress to stderr; exit codes are checked explicitly
Set-Location $PSScriptRoot
Start-Transcript -Path "$PSScriptRoot\run_gpu_transcript.txt" -Append | Out-Null

function Find-Python {
    foreach ($v in @("3.12", "3.11", "3.13", "3.10")) {
        try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { return ,@("py", "-$v") } } catch {}
    }
    try { & python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)" 2>$null; if ($LASTEXITCODE -eq 0) { return ,@("python") } } catch {}
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.10-3.13 not found. Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then run this script again." -ForegroundColor Red
    Stop-Transcript | Out-Null; exit 1
}
Write-Host "Using Python: $py"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if ($py.Length -gt 1) { & $py[0] $py[1] -m venv .venv } else { & $py[0] -m venv .venv }
}
$vpy = "$PSScriptRoot\.venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip

# PyTorch with CUDA: try recent CUDA builds until one gives a working GPU
$ok = $false
foreach ($cu in @("cu128", "cu126", "cu124", "cu121")) {
    Write-Host "Installing PyTorch ($cu)..."
    & $vpy -m pip install torch --index-url "https://download.pytorch.org/whl/$cu"
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
if ($LASTEXITCODE -ne 0) { Write-Host "Library install failed; see run_gpu_transcript.txt" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }
& $vpy -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| CUDA', torch.version.cuda)"

Write-Host "Self-test (tiny models, about 1-2 minutes)..."
& $vpy p3gpu.py g2 g3 --smoke
if ($LASTEXITCODE -ne 0) { Write-Host "Self-test failed; send me results_smoke\log.txt" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }

Write-Host "Running the experiments (about 2-3 hours). Keep the laptop plugged in and stop it from sleeping."
& $vpy p3gpu.py all

if (Test-Path "results_gpu.zip") { Remove-Item "results_gpu.zip" }
Compress-Archive -Path "results_gpu\*", "run_gpu_transcript.txt" -DestinationPath "results_gpu.zip"
Write-Host "Done. Results are in results_gpu.zip" -ForegroundColor Green
Stop-Transcript | Out-Null
