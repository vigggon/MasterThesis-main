"""
Ingest manually exported MT5 CSV data.
Expected format: date, time, open, high, low, close, tick_volume/vol...
Or standard MT5 export format.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from pathlib import Path
from utils import SESSION_DIR, OPEN_START, OPEN_END, PROCESSED_DIR

def ingest_csv(csv_path: str = "data/raw/mt5_147days.csv"):
    path = Path(csv_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Reading {path}...")
    # Try generic read, assuming header exists
    df = pd.read_csv(path)
    
    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    
    # MT5 export usually has <DATE> <TIME> or similar.
    # Check common formats
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        # Try parsing first column as datetime if singular
        try:
            df["datetime"] = pd.to_datetime(df.iloc[:, 0]) 
        except:
            print("Could not parse datetime column. Expected 'date' and 'time' or 'datetime'.")
            return

    # Filter columns
    req_cols = ["open", "high", "low", "close", "tick_volume"]
    if "vol" in df.columns:
        df = df.rename(columns={"vol": "volume"})
    elif "tick_volume" in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
        
    if "volume" not in df.columns:
         df["volume"] = 0
         
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    
    # Ensure UTC?
    # User must know if their data is UTC or not. 
    # Assumption: MT5 Export is Server Time. 
    # Use generic 'as is' or try to align?
    # For now, we assume inputs are 'correct' or consistent with history.
    
    # Filter for Open Session (09:30 - 11:30 ET)
    
    # Ensure main DF is UTC-aware
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC")

    # Create NY copy for filtering
    df_ny = df.copy()
    df_ny["datetime"] = df_ny["datetime"].dt.tz_convert("America/New_York")
    
    # Filter by time
    # 09:30 is 9*60+30 = 570 min
    # 11:30 is 11*60+30 = 690 min
    minutes = df_ny["datetime"].dt.hour * 60 + df_ny["datetime"].dt.minute
    open_mask = (minutes >= 570) & (minutes <= 690)
    
    df_filtered = df[open_mask].copy()
    print(f"Filtered for session 09:30-11:30 ET: {len(df_filtered)} / {len(df)} rows")
    
    if len(df_filtered) == 0:
        print("No rows in session! Checking timestamps...")
        print(df_ny["datetime"].head())
        return

    # Append
    target_path = SESSION_DIR / "nasdaq_open_session.csv"
    current_df = pd.read_csv(target_path)
    current_df["datetime"] = pd.to_datetime(current_df["datetime"], utc=True)
    
    # Ensure new data matches target format (UTC)
    # It is already UTC in `df_filtered` (just the mask was calculated on NY)
    # But clean up columns
    df_filtered = df_filtered[["datetime", "open", "high", "low", "close", "volume"]]
    
    # Remove duplicates
    combined = pd.concat([current_df, df_filtered]).drop_duplicates(subset="datetime").sort_values("datetime")
    
    combined.to_csv(target_path, index=False)
    print(f"Merged! New count: {len(combined)}")
    print("Please now run: python src/feature_engineering.py && python src/label_generator.py && python src/dataset_builder.py")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/mt5_147days.csv"
    ingest_csv(path)
