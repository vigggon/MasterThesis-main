"""Quick check: PyTorch installed and CUDA available. Run after setup from project root: python scripts/verify_gpu.py"""
import sys
try:
    import torch
except ImportError:
    print("PyTorch not installed. Run setup.bat or setup.ps1 first.")
    sys.exit(1)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version (runtime):", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. Training will use CPU (slower). Check SETUP.md for driver/CUDA.")
