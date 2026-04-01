"""
Generate "Primary Data" (detailed Trade Tables) for the thesis.
As requested by the advisor, this creates CSVs of every trade performed
in the backtest, including indicator values at the time of entry.
"""
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtesting.backtester import load_backtest_data_and_predictions, generate_trade_table, _get_threshold
from utils import RESULTS_ROOT, MIN_ATR, DAILY_STOP_R, DAILY_TP_R

def run_generate_primary_data():
    output_dir = RESULTS_ROOT / "primary_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = ["transformer", "lstm"]
    
    print("Generating Primary Data (Backtest Trade Tables)...")
    
    for model_name in models:
        print(f"  Processing {model_name}...")
        
        # Load backtest data and predictions
        # returns, times, pred, y, probs, atr_values, X, feature_cols
        _, times, _, _, probs, atr_values, X_features, feature_names = load_backtest_data_and_predictions(
            model_name, 
            spread_points=1.0, 
            min_atr=MIN_ATR,
            daily_max_loss=DAILY_STOP_R,
            daily_take_profit=DAILY_TP_R
        )
        
        th = _get_threshold(model_name)
        
        # Load raw dataframe for exit resolution
        from utils import PROCESSED_DIR
        raw_path = PROCESSED_DIR.parent / "features" / "open_features.csv"
        df_raw = pd.read_csv(raw_path)
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
        
        # Generate table
        df_trades = generate_trade_table(
            probs=probs,
            th=th,
            X_features=X_features.numpy() if hasattr(X_features, "numpy") else X_features,
            feature_names=feature_names,
            spread_points=1.0,
            atr=atr_values,
            times=times,
            daily_max_loss=DAILY_STOP_R,
            daily_take_profit=DAILY_TP_R,
            min_atr=MIN_ATR,
            df_raw=df_raw
        )
        
        # Save
        out_path = output_dir / f"backtest_trades_{model_name}.csv"
        df_trades.to_csv(out_path, index=False)
        print(f"    Saved {len(df_trades)} trades to {out_path}")

if __name__ == "__main__":
    run_generate_primary_data()
