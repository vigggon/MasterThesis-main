"""
STEP 1 — Prepare M5 data for pipeline.
Output: data/raw/nasdaq_m5.csv

With Kaggle (NQ_5Years_8_11_2024.csv) + MT5 export (NAS100_*.csv) in data/raw/:
  python data_download.py   → merges Kaggle to 2024-08-09 with MT5 from 2024-08-29 (small gap).
  python data_download.py --no-merge   → use only Kaggle CSV.

Single CSV:  --from-csv path/to/file.csv  --start 2019-08-12  --end 2024-08-09
MT5 live:    --symbol NAS100  /  --list-symbols  /  --diagnose --symbol NAS100
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from utils import RAW_DIR

# Default external CSV (Kaggle etc.): if this file exists in data/raw/, it is used when you run with no args
DEFAULT_EXTERNAL_CSV = "NQ_5Years_8_11_2024.csv"
# MT5 export CSV: NAS100_* from MetaTrader 5 (from 2024-08-29; fills gap after Kaggle end 2024-08-09)
MT5_EXPORT_GLOB = "NAS100_*.csv"
KAGGLE_END = "2024-08-09"   # Last date in Kaggle data; MT5 should start after a small gap
MT5_START = "2024-08-29"   # First date of MT5 export (small gap 2024-08-10 to 2024-08-28)


def ensure_raw_dir():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def prepare_from_external_csv(
    csv_path: str | Path,
    start: str = "2019-08-12",
    end: str = "2024-08-09",
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Use M5 data from an external CSV (e.g. another broker, data vendor) for the full
    2019-08-12 to 2024-08-09. CSV must have columns: datetime, open, high, low, close, volume.
    Optionally clip to [start, end] and save to data/raw/nasdaq_m5.csv.
    """
    out_path = out_path or RAW_DIR / "nasdaq_m5.csv"
    ensure_raw_dir()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalize column names (e.g. Time -> datetime, Open -> open)
    col_map = {}
    for c in df.columns:
        c_lower = c.strip().lower()
        if c_lower in ("time", "date", "datetime"):
            col_map[c] = "datetime"
        elif c_lower in ("open", "high", "low", "close", "volume"):
            col_map[c] = c_lower
    df = df.rename(columns=col_map)
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV must have datetime (or Time/Date), open, high, low, close, volume. Missing: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[["datetime", "open", "high", "low", "close", "volume"]].sort_values("datetime").reset_index(drop=True)
    date_from = pd.Timestamp(start)
    date_to = pd.Timestamp(end)
    df = df[(df["datetime"] >= date_from) & (df["datetime"] <= date_to)]
    if len(df) == 0:
        raise ValueError(f"No rows in range {start} to {end}. Check CSV dates.")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} bars from {df['datetime'].min()} to {df['datetime'].max()} -> {out_path}")
    return df


def load_mt5_export_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load MT5 export CSV: <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>.
    Date format 2024.08.29, time 13:20:00. Returns df with datetime, open, high, low, close, volume.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"MT5 CSV not found: {path}")
    df = pd.read_csv(path, sep="\t")
    # Normalize column names: <DATE> -> date, etc.
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]
    # Build datetime (date may be 2024.08.29 -> replace . with - for ISO)
    date_str = df["date"].astype(str).str.replace(".", "-", regex=False)
    df["datetime"] = pd.to_datetime(date_str + " " + df["time"].astype(str))
    # Volume: use tick_volume if present else vol
    if "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
    elif "vol" in df.columns:
        df["volume"] = df["vol"]
    else:
        df["volume"] = 0
    df = df[["datetime", "open", "high", "low", "close", "volume"]].sort_values("datetime").reset_index(drop=True)
    return df


def merge_kaggle_mt5(
    kaggle_path: str | Path | None = None,
    mt5_path: str | Path | None = None,
    kaggle_end: str = KAGGLE_END,
    mt5_start: str = MT5_START,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Merge Kaggle data (up to kaggle_end) with MT5 export (from mt5_start). Small gap between them is OK.
    Output: nasdaq_m5.csv with Kaggle then MT5, sorted by datetime, no duplicate timestamps.
    """
    out_path = out_path or RAW_DIR / "nasdaq_m5.csv"
    ensure_raw_dir()
    kaggle_path = Path(kaggle_path or RAW_DIR / DEFAULT_EXTERNAL_CSV)
    if not kaggle_path.exists():
        raise FileNotFoundError(f"Kaggle CSV not found: {kaggle_path}. Place NQ_5Years_8_11_2024.csv in data/raw/.")
    # Find MT5 file if not given
    if mt5_path is None:
        mt5_files = list(RAW_DIR.glob(MT5_EXPORT_GLOB))
        if not mt5_files:
            raise FileNotFoundError(f"No MT5 export found: {RAW_DIR / MT5_EXPORT_GLOB}. Export NAS100 from MT5 to data/raw/.")
        mt5_path = sorted(mt5_files)[-1]
    else:
        mt5_path = Path(mt5_path)
    # Load Kaggle: single Time column, clip to <= kaggle_end
    df_k = pd.read_csv(kaggle_path)
    col_map = {}
    for c in df_k.columns:
        c_lower = c.strip().lower()
        if c_lower in ("time", "date", "datetime"):
            col_map[c] = "datetime"
        elif c_lower in ("open", "high", "low", "close", "volume"):
            col_map[c] = c_lower
    df_k = df_k.rename(columns=col_map)
    df_k["datetime"] = pd.to_datetime(df_k["datetime"])
    df_k = df_k[["datetime", "open", "high", "low", "close", "volume"]]
    df_k = df_k[df_k["datetime"] <= pd.Timestamp(kaggle_end)].sort_values("datetime").reset_index(drop=True)
    # Load MT5 export, clip to >= mt5_start
    df_m = load_mt5_export_csv(mt5_path)
    df_m = df_m[df_m["datetime"] >= pd.Timestamp(mt5_start)].reset_index(drop=True)
    # Merge and dedupe
    combined = pd.concat([df_k, df_m], ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    combined.to_csv(out_path, index=False)
    print(f"Merged Kaggle (to {kaggle_end}) + MT5 (from {mt5_start}): {len(combined)} bars, {combined['datetime'].min()} to {combined['datetime'].max()} -> {out_path}")
    return combined


def download_nasdaq_m5(
    symbol: str = "NAS100",
    start: str = "2019-08-12",
    end: str = "2024-08-09",
    timeframe_min: int = 5,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Connect to MT5 and download M5 OHLCV. Fallback symbol US100 if NAS100 unavailable.
    """
    if mt5 is None:
        raise ImportError("MetaTrader5 is required. Install with: pip install MetaTrader5")

    out_path = out_path or RAW_DIR / "nasdaq_m5.csv"
    ensure_raw_dir()

    if not mt5.initialize():
        raise RuntimeError("MT5 initialize failed. Ensure MetaTrader5 terminal is running.")

    try:
        date_from = datetime.strptime(start, "%Y-%m-%d")
        date_to = datetime.strptime(end, "%Y-%m-%d")
        tf = mt5.TIMEFRAME_M5 if timeframe_min == 5 else getattr(mt5, f"TIMEFRAME_M{timeframe_min}", mt5.TIMEFRAME_M5)

        # Try given symbol first, then common variants; enable symbol for history
        candidates = [symbol, "NAS100", "NDX100", "US100", "US100.cash", "NAS100.i", "NDX", "USTEC"]
        candidates = [c for c in candidates if c]
        seen = set()
        rates = None
        used_sym = None
        used_recent_only = False
        now = datetime.now()
        recent_to = now
        recent_from = now - timedelta(days=730)  # last 2 years

        for sym in candidates:
            if sym in seen:
                continue
            seen.add(sym)
            mt5.symbol_select(sym, True)  # ensure symbol is enabled for history
            if mt5.symbol_info(sym) is None:
                continue  # symbol not available on this broker
            # 1) Full requested range (naive datetime; broker may use server time)
            rates = mt5.copy_rates_range(sym, tf, date_from, date_to)
            if rates is not None and len(rates) > 0:
                used_sym = sym
                break
            # 1b) Full range in UTC (some brokers expect UTC)
            date_from_utc = datetime(2019, 8, 12, tzinfo=timezone.utc)
            date_to_utc = datetime(2024, 8, 9, 23, 59, 59, tzinfo=timezone.utc)
            rates = mt5.copy_rates_range(sym, tf, date_from_utc, date_to_utc)
            if rates is not None and len(rates) > 0:
                used_sym = sym
                break
            # 2) Recent range only (many brokers only provide 1–2 years of M5 for indices)
            rates = mt5.copy_rates_range(sym, tf, recent_from, recent_to)
            if rates is not None and len(rates) > 0:
                used_sym = sym
                used_recent_only = True
                break
            # 3) Last N bars (broker may reject large count; try smaller if 100k fails)
            for count in (100000, 50000, 10000, 5000):
                rates = mt5.copy_rates_from_pos(sym, tf, 1, count)
                if rates is not None and len(rates) > 0:
                    used_sym = sym
                    used_recent_only = True
                    break
            if used_sym is not None:
                break
            # 4) From position 0 (some brokers only return data this way)
            for count in (100000, 50000, 10000, 5000):
                rates = mt5.copy_rates_from_pos(sym, tf, 0, count)
                if rates is not None and len(rates) > 0:
                    used_sym = sym
                    used_recent_only = True
                    break
            if used_sym is not None:
                break
        if rates is None or len(rates) == 0:
            raise ValueError(
                "No data for NAS100/US100. Your broker may use a different symbol name.\n"
                "Run:  python data_download.py --diagnose --symbol NAS100   to see what MT5 returns.\n"
                "Or:   python data_download.py --list-symbols   then  --symbol \"ExactSymbolName\""
            )
        symbol = used_sym or symbol

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"time": "datetime", "tick_volume": "volume"})
        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("datetime").reset_index(drop=True)
        # Clip to requested range (in case we used recent or from_pos)
        df = df[(df["datetime"] >= pd.Timestamp(date_from)) & (df["datetime"] <= pd.Timestamp(date_to))]
        if len(df) == 0:
            raise ValueError(f"Symbol {symbol}: no bars in range {start} to {end}. Try a shorter range or check broker history.")
        if used_recent_only:
            print(f"Note: Broker returned only limited history for {symbol}. Saved {len(df)} bars.")
        df.to_csv(out_path, index=False)
        return df
    finally:
        mt5.shutdown()


def diagnose_symbol(symbol: str, timeframe_min: int = 5):
    """Print what MT5 returns for this symbol (range, recent range, from_pos). Helps debug 'No data'."""
    if mt5 is None:
        raise ImportError("MetaTrader5 is required.")
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize failed. Ensure MetaTrader5 terminal is running.")
    try:
        tf = mt5.TIMEFRAME_M5 if timeframe_min == 5 else getattr(mt5, f"TIMEFRAME_M{timeframe_min}", mt5.TIMEFRAME_M5)
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"{symbol}: symbol_info = None (symbol not available or wrong name)")
            return
        print(f"{symbol}: path={getattr(info, 'path', '')}, visible={getattr(info, 'visible', False)}")
        date_from = datetime(2019, 8, 12)
        date_to = datetime(2024, 8, 9)
        r1 = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        print(f"  copy_rates_range(2019-08-12..2024-08-09): {len(r1) if r1 is not None else 'None'}")
        now = datetime.now()
        recent_from = now - timedelta(days=730)
        r2 = mt5.copy_rates_range(symbol, tf, recent_from, now)
        print(f"  copy_rates_range(last 2 years):          {len(r2) if r2 is not None else 'None'}")
        r3 = mt5.copy_rates_from_pos(symbol, tf, 1, 5000)
        print(f"  copy_rates_from_pos(1, 5000):            {len(r3) if r3 is not None else 'None'}")
        r4 = mt5.copy_rates_from_pos(symbol, tf, 0, 5000)
        print(f"  copy_rates_from_pos(0, 5000):            {len(r4) if r4 is not None else 'None'}")
    finally:
        mt5.shutdown()


def list_symbols(filter_substring: str = ""):
    """Print symbols from MT5 Market Watch. Use filter_substring e.g. '100' or 'NAS' to narrow down."""
    if mt5 is None:
        raise ImportError("MetaTrader5 is required.")
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize failed. Ensure MetaTrader5 terminal is running.")
    try:
        symbols = mt5.symbols_get()
        if symbols is None:
            print("No symbols returned. Check MT5 connection and Market Watch.")
            return
        names = sorted([s.name for s in symbols])
        if filter_substring:
            names = [n for n in names if filter_substring.upper() in n.upper()]
        print(f"Symbols (filter: {filter_substring or 'all'}):")
        for n in names:
            print(f"  {n}")
        if not names:
            print("  (none)")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NAS100/US100 M5 data from MetaTrader5 or use external CSV. With Kaggle + MT5 export in data/raw/, merges them (Kaggle to 2024-08-09, MT5 from 2024-08-29).")
    parser.add_argument("--symbol", type=str, default="NAS100", help="MT5 symbol (default: NAS100)")
    parser.add_argument("--from-csv", type=str, default="", help="Use single external CSV; columns: datetime, open, high, low, close, volume")
    parser.add_argument("--start", type=str, default="2019-08-12", help="Start date (for --from-csv or clip)")
    parser.add_argument("--end", type=str, default="2024-08-09", help="End date (for --from-csv or clip)")
    parser.add_argument("--merge-mt5", action="store_true", help="Merge Kaggle (to 2024-08-09) + MT5 export NAS100_*.csv (from 2024-08-29) -> nasdaq_m5.csv")
    parser.add_argument("--no-merge", action="store_true", help="Skip merge; use only Kaggle CSV even if MT5 export exists")
    parser.add_argument("--list-symbols", action="store_true", help="List available MT5 symbols and exit")
    parser.add_argument("--filter", type=str, default="", help="Filter symbols by substring when using --list-symbols (e.g. 100, NAS)")
    parser.add_argument("--diagnose", action="store_true", help="Print what MT5 returns for --symbol (no download)")
    args = parser.parse_args()
    if args.list_symbols:
        list_symbols(filter_substring=args.filter)
    elif args.diagnose:
        diagnose_symbol(args.symbol)
    elif args.from_csv:
        prepare_from_external_csv(args.from_csv, start=args.start, end=args.end)
    elif args.merge_mt5:
        merge_kaggle_mt5(kaggle_end=KAGGLE_END, mt5_start=MT5_START)
    else:
        default_kaggle = RAW_DIR / DEFAULT_EXTERNAL_CSV
        mt5_files = list(RAW_DIR.glob(MT5_EXPORT_GLOB))
        if default_kaggle.exists() and mt5_files and not args.no_merge:
            print("Merging Kaggle + MT5 export (Kaggle to 2024-08-09, MT5 from 2024-08-29)")
            merge_kaggle_mt5(kaggle_end=KAGGLE_END, mt5_start=MT5_START)
        elif default_kaggle.exists():
            print(f"Using {DEFAULT_EXTERNAL_CSV} from data/raw/")
            prepare_from_external_csv(default_kaggle, start=args.start, end=args.end)
        else:
            download_nasdaq_m5(symbol=args.symbol, start=args.start, end=args.end)
