@echo off
REM Thesis Trading AI — Windows setup: venv + dependencies + PyTorch (CUDA)
REM Run from thesis_trading_ai folder: setup.bat
REM Prerequisites: Python 3.10–3.12 installed, NVIDIA driver installed (see SETUP.md)

set PYTHON=python
where python >nul 2>nul || set PYTHON=py -3.13
where %PYTHON% >nul 2>nul || set PYTHON=py -3.12
where %PYTHON% >nul 2>nul || set PYTHON=py -3.11
where %PYTHON% >nul 2>nul || (echo Python not found. Install Python 3.10–3.13 from https://www.python.org/downloads/ && exit /b 1)

echo Creating virtual environment in .venv ...
%PYTHON% -m venv .venv
if errorlevel 1 (echo Failed to create venv. && exit /b 1)

call .venv\Scripts\activate.bat
echo Upgrading pip ...
python -m pip install --upgrade pip

echo Installing dependencies (excluding PyTorch) ...
pip install -r requirements-base.txt
if errorlevel 1 (echo pip install failed. && exit /b 1)

REM PyTorch with CUDA 12.6 (match your driver: cu118, cu126, or cu128 — see SETUP.md)
echo Installing PyTorch with CUDA 12.6 ...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
  echo PyTorch CUDA install failed. Try CPU-only: pip install torch torchvision torchaudio
  exit /b 1
)

echo.
echo Setup complete. Activate with:  .venv\Scripts\activate
echo Verify GPU:  python -c "import torch; print('CUDA:', torch.cuda.is_available())"
exit /b 0
