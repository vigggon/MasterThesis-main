"""
STEP 3 — Filter to Nasdaq open session (New York 09:30–11:30).
Output: data/session_filtered/nasdaq_open_session.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from pathlib import Path

from utils import RAW_DIR, SESSION_DIR, NY_TZ, OPEN_START, OPEN_END


def load_raw() -> pd.DataFrame:
    path = RAW_DIR / "nasdaq_m5.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run data_download.py first. Expected: {path}")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)
    return df


def filter_open_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:30–11:30 New York time."""
    t = df["datetime"].dt.tz_convert(NY_TZ).dt.time
    start = pd.Timestamp(OPEN_START).time()
    end = pd.Timestamp(OPEN_END).time()
    mask = (t >= start) & (t <= end)
    return df.loc[mask].copy().reset_index(drop=True)


def run_session_filter() -> pd.DataFrame:
    df = load_raw()
    out = filter_open_session(df)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SESSION_DIR / "nasdaq_open_session.csv"
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    run_session_filter()
