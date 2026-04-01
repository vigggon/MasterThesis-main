import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Import the RESULTS_ROOT from our config loader
from utils.config_loader import RESULTS_ROOT

def get_logger(name: str, log_file: str = "thesis.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    # Formatters
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_dir = RESULTS_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / log_file
    
    fh = RotatingFileHandler(file_path, maxBytes=5*1024*1024, backupCount=2)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
