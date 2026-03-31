# Thesis Trading AI — Windows PowerShell setup: venv + dependencies + PyTorch (CUDA)
# Run from thesis_trading_ai: .\setup.ps1
# If execution is blocked: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

$ErrorActionPreference = "Stop"
$py = $null
foreach ($cmd in @("python", "py -3.13", "py -3.12", "py -3.11", "py -3.10")) {
    try { $null = Get-Command $cmd.Split()[0] -ErrorAction Stop; $py = $cmd; break } catch {}
}
if (-not $py) { Write-Error "Python 3.10-3.13 not found. Install from https://www.python.org/downloads/" }

Write-Host "Using: $py"
Write-Host "Creating virtual environment in .venv ..."
& $py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
Write-Host "Installing dependencies (excluding PyTorch) ..."
pip install -r requirements-base.txt
Write-Host "Installing PyTorch with CUDA 12.6 ..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
Write-Host ""
Write-Host "Setup complete. Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Verify GPU:  python -c `"import torch; print('CUDA:', torch.cuda.is_available())`""
