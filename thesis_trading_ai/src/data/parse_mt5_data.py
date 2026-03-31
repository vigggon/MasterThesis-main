import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from pathlib import Path

RAW_DIR = Path(r"c:/Users/viggo/Desktop/MasterThesis/thesis_trading_ai/data/raw")

def run():
    new_file = RAW_DIR / "USTEC_M5_NEW.csv"
    old_file = RAW_DIR / "nasdaq_m5.csv"
    
    print(f"Reading MT5 export: {new_file}")
    
    # Read the MT5 format (tab separated)
    df_new = pd.read_csv(new_file, sep="\t")
    
    # Clean column names (remove < >)
    df_new.columns = [col.replace("<", "").replace(">", "") for col in df_new.columns]
    
    # Combine DATE and TIME
    # Note: DATE is format YYYY.MM.DD -> YYYY-MM-DD
    df_new["DATE"] = df_new["DATE"].str.replace(".", "-")
    df_new["datetime"] = pd.to_datetime(df_new["DATE"] + " " + df_new["TIME"])
    
    # Rename columns to match old format
    df_new = df_new.rename(columns={
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "TICKVOL": "volume"
    })
    
    # Select final columns
    df_new = df_new[["datetime", "open", "high", "low", "close", "volume"]]
    
    print(f"Reading old data: {old_file}")
    df_old = pd.read_csv(old_file)
    df_old["datetime"] = pd.to_datetime(df_old["datetime"])
    
    # Remove timezone from old data for clean merge if it has any, though it looks naive in raw
    if df_old["datetime"].dt.tz is not None:
         df_old["datetime"] = df_old["datetime"].dt.tz_localize(None)
         
    # Combine, sort, and drop duplicates
    print(f"Old data points: {len(df_old)}")
    print(f"New data points: {len(df_new)}")
    
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined = df_combined.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    
    print(f"Combined data points: {len(df_combined)}")
    print(f"Saving to {old_file}...")
    
    # Save back to nasdaq_m5.csv
    df_combined.to_csv(old_file, index=False)
    print("Done!")

if __name__ == "__main__":
    run()
