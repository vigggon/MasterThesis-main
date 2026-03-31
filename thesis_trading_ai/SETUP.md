# Setup — Thesis Trading AI (Windows)

## What you do manually

### 1. Install Python 3.10, 3.11, 3.12, or 3.13

- Download: https://www.python.org/downloads/
- During install, check **“Add Python to PATH”**.
- Confirm in a new terminal: `python --version` or `py -3.13 --version`.

### 2. Install / update NVIDIA GPU driver (for CUDA)

PyTorch CUDA wheels bundle their own CUDA runtime; you only need a **compatible NVIDIA driver**.

- **Driver download:** https://www.nvidia.com/Download/index.aspx  
  - Choose: Product type = GeForce / Quadro / etc., your GPU, Windows.
- **Rough compatibility:**
  - **PyTorch CUDA 12.6** (default in our setup): use a **recent driver** (e.g. 535+ or 545+).  
  - **PyTorch CUDA 11.8**: older driver is OK (e.g. 525+).
- After installing, open a **new** terminal and run:
  ```text
  nvidia-smi
  ```
  You should see driver version and GPU; if not, the driver is not installed correctly.

### 3. (Optional) Use a different PyTorch CUDA build

If the default (CUDA 12.6) fails or `torch.cuda.is_available()` is False:

- **Older GPU / driver:** in `setup.bat` (or `setup.ps1`) change `cu126` to `cu118` and run the PyTorch install line again inside the venv:
  ```text
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **Newer driver:** you can try `cu128`:
  ```text
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```

---

## What the setup script does automatically

From the **thesis_trading_ai** folder (where `setup.bat` and `requirements-base.txt` are):

1. Create a virtual environment in **`.venv`**.
2. Activate it and upgrade **pip**.
3. Install all dependencies from **requirements-base.txt** (no PyTorch).
4. Install **PyTorch** (and torchvision, torchaudio) with **CUDA 12.6** from the official PyTorch index.

### Run the automated setup

**Option A — Command Prompt (recommended)**

```cmd
cd path\to\thesis_trading_ai
setup.bat
```

**Option B — PowerShell**

```powershell
cd path\to\thesis_trading_ai
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser   # once, if scripts are blocked
.\setup.ps1
```

### After setup

1. Activate the environment (every new terminal where you work on the project):
   - **Cmd:** `path\to\thesis_trading_ai\.venv\Scripts\activate.bat`
   - **PowerShell:** `path\to\thesis_trading_ai\.venv\Scripts\Activate.ps1`
2. Verify GPU:
   ```text
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```
   Or:
   ```text
   python scripts\verify_gpu.py
   ```
3. Run the pipeline from **src** (e.g. after downloading data):  
   `python data_download.py`, then the rest as in the main README.

---

## Version summary

| Component        | Version / note |
|-----------------|----------------|
| Python          | 3.10 – 3.13    |
| PyTorch (CUDA)  | From PyTorch index: **cu126** (or cu118 / cu128) |
| NVIDIA driver   | Recent (e.g. 535+) for CUDA 12.6; 525+ for CUDA 11.8 |
| CUDA Toolkit    | Not required; PyTorch wheels include their own CUDA runtime |

If something fails, run `nvidia-smi` and check the driver version, then try the **cu118** or **cu128** variant as above.
