"""
Auto-fetch Nasdaq data from ANY matching symbol that returns data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def auto_fetch():
    if not mt5.initialize():
        print("MT5 init failed")
        return

    # Get all symbols
    symbols = mt5.symbols_get()
    if not symbols:
        print("No symbols found")
        mt5.shutdown()
        return
        
    # Filter for Nasdaq candidates
    candidates = []
    terms = ["NAS", "NDX", "US100", "TEC"]
    for s in symbols:
        name = s.name.upper()
        if any(t in name for t in terms):
            candidates.append(s.name)
            
    print(f"Candidates found: {candidates}")
    
    working_symbol = None
    
    for sym in candidates:
        # Try to select
        if not mt5.symbol_select(sym, True):
            # If select fails, try fetching anyway (might be already selected)
            pass
            
        print(f"Testing {sym}...")
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 10)
        
        if rates is not None and len(rates) > 0:
            print(f"SUCCESS: Found data for {sym} ({len(rates)} bars)")
            working_symbol = sym
            break
        else:
            err = mt5.last_error()
            print(f"  No data (Error: {err})")
            
    if not working_symbol:
        print("Could not fetch data from any candidate.")
        mt5.shutdown()
        return

    # Fetch full history
    print(f"Fetching 147 days for {working_symbol}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=147)
    
    # Try range
    rates = mt5.copy_rates_range(working_symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print("Range fetch returned empty. Trying from_pos fallback (approx 42k bars).")
        count = 147 * 24 * 12
        rates = mt5.copy_rates_from_pos(working_symbol, mt5.TIMEFRAME_M5, 0, count)
        
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("Final fetch failed.")
        return

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    
    out_path = Path("data/raw/mt5_147days.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    auto_fetch()
