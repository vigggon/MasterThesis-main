"""
Walk-Forward Analysis Runner.
Simulates rolling retraining to verify model generalization across years.
Folds:
1. Train 2019-2023 -> Backtest 2024
2. Train 2019-2024 -> Backtest 2025
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sys
import subprocess
import pandas as pd
from pathlib import Path
from utils import PROCESSED_DIR, EXPERIMENTS_ROOT, RESULTS_ROOT

PYTHON_EXE = sys.executable
TRAIN_SCRIPT = "src/train.py"
BACKTEST_SCRIPT = "src/backtester.py"
DATA_BUILDER = "src/dataset_builder.py"
UTILS_FILE = "src/utils.py"

def replace_in_file(path, target, replacement):
    with open(path, 'r') as f:
        content = f.read()
    content = content.replace(target, replacement)
    with open(path, 'w') as f:
        f.write(content)

def run_command(cmd):
    print(f">> Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def run_fold(fold_name, train_end_year, backtest_year):
    print(f"\n=== Running Fold: {fold_name} (Train -{train_end_year}, Test {backtest_year}) ===")
    
    # 1. Modify src/utils.py to set date ranges (Hacky but effective)
    # Original: BACKTEST_START = "2024-01-01", BACKTEST_END = "2026-12-31" set by BACKTEST_DAYS usually.
    # We will use explicit BACKTEST_START/END and set BACKTEST_DAYS = None to force it.
    
    # Read utils
    with open(UTILS_FILE, 'r') as f:
        original_utils = f.read()
    
    try:
        # Construct Fold Dates
        train_start = "2019-01-01"
        train_end = f"{train_end_year}-12-31"
        test_start = f"{backtest_year}-01-01"
        test_end = f"{backtest_year}-12-31"
        
        # Modify Utils content in memory and write
        # Regex or string replace might be fragile. 
        # Safer: Set BACKTEST_DAYS = None, and overwrite constants.
        
        new_utils = original_utils.replace('BACKTEST_DAYS = 180', 'BACKTEST_DAYS = None')
        new_utils = new_utils.replace('TRAIN_START = "2019-01-01"', f'TRAIN_START = "{train_start}"')
        new_utils = new_utils.replace('TRAIN_END = "2030-12-31"', f'TRAIN_END = "{train_end}"')
        new_utils = new_utils.replace('BACKTEST_START = "2024-01-01"', f'BACKTEST_START = "{test_start}"')
        new_utils = new_utils.replace('BACKTEST_END = "2026-12-31"', f'BACKTEST_END = "{test_end}"')
        
        with open(UTILS_FILE, 'w') as f:
            f.write(new_utils)
            
        # 2. Re-build Dataset
        run_command(f'"{PYTHON_EXE}" {DATA_BUILDER}')
        
        # 3. Train Model (Transformer)
        # Use --min-atr 0.0 (Generalist)
        run_command(f'"{PYTHON_EXE}" {TRAIN_SCRIPT} transformer --min-atr 0.0 --epochs 50') 
        
        # 4. Backtest
        # Use --min-atr 15.0 (Specialized Inference)
        # Save results to a separate fold directory?
        # currently backtester saves to results/backtest/transformer_metrics.csv
        run_command(f'"{PYTHON_EXE}" {BACKTEST_SCRIPT} transformer --min-atr 15.0')
        
        # 5. Capture Metrics
        metrics_file = RESULTS_ROOT / "backtest/transformer_metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            sharpe = df["sharpe_ratio"].iloc[0]
            profit = df["total_return"].iloc[0]
            print(f"RESULT {fold_name}: Profit={profit:.2f}, Sharpe={sharpe:.3f}")
            
            # Save aside
            df.to_csv(RESULTS_ROOT / f"walk_forward_{fold_name}.csv", index=False)
        else:
            print(f"RESULT {fold_name}: No metrics found.")

    finally:
        # Restore Utils
        with open(UTILS_FILE, 'w') as f:
            f.write(original_utils)

def main():
    # Fold 1: Train 2019-2023, Test 2024
    # Note: 2024 data exists fully.
    run_fold("Fold_2024", 2023, 2024)
    
    # Fold 2: Train 2019-2024, Test 2025
    # Note: 2025 data is partial (Aug-Dec)? Or full? 
    # Current dataset (Aug 2019 - Jan 2026).
    # So 2024 is FULL year. 2025 is FULL year.
    run_fold("Fold_2025", 2024, 2025)

if __name__ == "__main__":
    main()
