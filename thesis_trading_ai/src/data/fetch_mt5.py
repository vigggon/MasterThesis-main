"""
Fetch data from MT5 and append to session dataset.
Symbol: US100 or NDX100.
Range: Last 147 days.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
from pathlib import Path
from utils import SESSION_DIR, OPEN_START, OPEN_END

def fetch_and_merge():
    if not mt5.initialize():
        print("MT5 init failed")
        return

    # Find symbol
    symbol = "US100"
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        symbol = "NDX100"
        if not mt5.symbol_select(symbol, True):
            print("Could not select US100 or NDX100")
            print("Error:", mt5.last_error())
            mt5.shutdown()
            return

    print(f"Using symbol: {symbol}")
    
    # Try fetching last 10 bars first to check availability
    test_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
    if test_rates is None or len(test_rates) == 0:
        print("Test fetch (last 10 bars) failed. History might not be ready.")
        print("Error:", mt5.last_error())
        mt5.shutdown()
        return
        
    print(f"Test fetch success: {len(test_rates)} bars.")
    
    # 147 Days ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=147)
    
    print(f"Fetching from {start_date} to {end_date}")
    
    # Fetch M5
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print("Range fetch failed (maybe history gap?). Using ALL available history via from_pos?")
        # Fallback: calculate number of bars in 147 days
        # 147 days * 24 hours * 12 bars/hour = 42336 bars
        # crypto/forex implies 24/7, stocks 24/5?
        # NDX100 is likely 23h or 24h.
        count = 147 * 24 * 12
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
        print(f"Fallback fetch (last {count} bars) result: {len(rates) if rates is not None else 'None'}")
    
    if rates is None or len(rates) == 0:
        print("No data received final.")
        print("Error:", mt5.last_error())
        mt5.shutdown()
        return
        
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    
    # MT5 is usually UTC+2 or UTC+3. 
    # CRITICAL: We need to align with existing dataset which uses UTC.
    # We will assume MT5 Time is relevant to market hours (often broker server time).
    # BUT, to filter 9:30-11:30 ET correctly, we need to know the timezone.
    # Let's infer: Open usually happens at 16:30 Broker Time (UTC+2+Summer=UTC+3? no).
    # If Broker is UTC+2 (Winter) / UTC+3 (Summer), US Open (09:30 ET) = 16:30.
    # Let's just convert to UTC first if possible, or use the pattern.
    # Standard approach: Assume the existing `nasdaq_open_session.csv` has UTC timestamps.
    # We will localize the raw MT5 timestamps to UTC. 
    # NOTE: MT5 `time` is usually Timestamp in seconds, often UTC-aligned OR Server time.
    # For simplest integration, I'll filter by TIME on the row, assuming the Broker Server Time 
    # ALIGNS with the 09:30-11:30 window pattern if I convert to appropriate TZ.
    
    # BETTER APPROACH: Convert to 'America/New_York' and filter 09:30-11:30.
    # IMPORTANT: MT5 timestamps (unit='s') are strictly UTC in the `datetime` object IF built correctly, 
    # but usually `copy_rates_range` gives Server Time.
    # Let's try to assume data is mostly aligned.
    
    # Let's inspect the Raw DF first in a separate check? 
    # No, let's just save the raw data for inspection.
    raw_path = Path("data/raw/mt5_new_raw.csv")
    raw_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")
    
    # Now, try to filter for open session.
    # We need to know what 9:30 ET corresponds to in this data.
    # Let's assume we can map to NY time.
    # If we don't know Server Offset, we can't do this perfectly.
    # HACK: Download, then we will inspect one day to see where volume spikes (Open).
    
    return

if __name__ == "__main__":
    fetch_and_merge()
