import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Adjust to project path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import build_features
from src.utils import NY_TZ, RESULTS_ROOT

def merge_live_trades_with_features(live_pnl_csv: str, raw_mt5_data_csv: str, output_csv: str = "live_trades_with_features.csv"):
    """
    Retroactively reconstructs the feature state for every trade taken during live forward testing.
    Needs the live PnL log and a fresh raw MT5 data export covering the live testing period.
    """
    print(f"Loading Live Trades from: {live_pnl_csv}")
    trades_df = pd.read_csv(live_pnl_csv)
    
    if len(trades_df) == 0:
        print("No live trades found in the provided PnL CSV.")
        return
        
    print(f"Loading Raw MT5 Data from: {raw_mt5_data_csv}")
    # Load raw MT5 export spanning the live period
    df_raw = pd.read_csv(raw_mt5_data_csv, sep="\t" if "\t" in open(raw_mt5_data_csv).readline() else ",")
    
    # Clean if M5 raw export format
    if "DATE" in df_raw.columns and "TIME" in df_raw.columns:
         df_raw.columns = [col.replace("<", "").replace(">", "") for col in df_raw.columns]
         df_raw["DATE"] = df_raw["DATE"].str.replace(".", "-")
         df_raw["datetime"] = pd.to_datetime(df_raw["DATE"] + " " + df_raw["TIME"])
         df_raw = df_raw.rename(columns={"OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close", "TICKVOL": "volume"})
         df_raw = df_raw[["datetime", "open", "high", "low", "close", "volume"]]
    else:
         df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
         
    # Ensure TZ aware
    if df_raw["datetime"].dt.tz is None:
         df_raw["datetime"] = df_raw["datetime"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)
         
    # Convert trade resolution datetimes
    trades_df["resolved_at"] = pd.to_datetime(trades_df["resolved_at"])
    
    # Rebuild features across the continuous dataset
    print("Rebuilding continuous feature matrix...")
    # NOTE: To exactly match the live runner, we need to filter to session hours first
    from src.run_live_forward import _filter_open_session
    t = df_raw["datetime"].dt.tz_convert(NY_TZ).dt.time
    # Use standard open bounds
    start = datetime.strptime("09:30", "%H:%M").time()
    end = datetime.strptime("11:30", "%H:%M").time()
    mask = (t >= start) & (t <= end)
    
    session_df = df_raw.loc[mask].copy().reset_index(drop=True)
    
    # Optional: If the live runner resamples to 10M, do so here
    from src.feature_engineering import resample_session_data
    session_df = resample_session_data(session_df, timeframe_min=10)
    
    # Calculate features
    feats_df = build_features(session_df)
    
    print("Matching trades to feature snapshots...")
    enhanced_trades = []
    
    # We must match based on entry time, NOT resolution time
    # This requires looking up the exact 10M candle matching the entry price
    for idx, row in trades_df.iterrows():
        # Find the row in feats_df where 'close' matches 'entry_close' 
        # occurring BEFORE 'resolved_at'
        trade_res_time = row["resolved_at"]
        entry_price = row["entry_close"]
        
        # Look backwards from resolution time to find entry
        candidates = feats_df[(feats_df["datetime"] <= trade_res_time)]
        # Match close price to within fractional epsilon
        matched_entry = candidates[np.isclose(candidates["close"], entry_price, atol=1e-3)]
        
        if len(matched_entry) > 0:
            # Take the closest chronological match preceding resolution
            match = matched_entry.iloc[-1]
            trade_dict = row.to_dict()
            trade_dict["entry_time"] = match["datetime"]
            
            # Extract features
            exclude = ["datetime", "open", "high", "low", "close", "volume"]
            for col in match.index:
                if col not in exclude:
                    trade_dict[col] = match[col]
                    
            enhanced_trades.append(trade_dict)
        else:
            print(f"Warning: Could not isolate exact entry bar for trade at {trade_res_time}")
            enhanced_trades.append(row.to_dict())
            
    out_df = pd.DataFrame(enhanced_trades)
    
    # Standardize output for Reviewer
    # [datetime, close, direction, pnl, indicator_1, indicator_2...]
    
    out_path = Path("results/forward_test") / output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Successfully generated reviewer table: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retroactively merge MT5 historical data with live PnL logs.")
    parser.add_argument("--pnl", type=str, required=True, help="Path to live session PnL CSV (e.g. results/forward_test/session_pnl_transformer.csv)")
    parser.add_argument("--data", type=str, required=True, help="Path to freshly exported MT5 CSV covering the live period")
    parser.add_argument("--out", type=str, default="live_trades_with_features.csv", help="Output filename")
    
    args = parser.parse_args()
    merge_live_trades_with_features(args.pnl, args.data, args.out)
