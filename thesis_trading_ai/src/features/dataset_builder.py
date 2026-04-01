"""
STEP 6 — Dataset construction: WINDOW=24, time-based split.
Train: TRAIN_START through TRAIN_END (or day before backtest if BACKTEST_DAYS set).
Backtest: last BACKTEST_DAYS of data (e.g. 90 = latest 3 months) or BACKTEST_START–BACKTEST_END. No leakage.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from pathlib import Path

from utils import (
    LABELED_DIR,
    PROCESSED_DIR,
    WINDOW,
    TRAIN_START,
    TRAIN_END,
    BACKTEST_DAYS,
    BACKTEST_START,
    BACKTEST_END,
)


def load_labeled() -> pd.DataFrame:
    path = LABELED_DIR / "open_labeled.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run label_generator.py first. Expected: {path}")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


# Feature columns (exclude datetime, OHLCV, label)
OHLCV = ["open", "high", "low", "close", "volume"]


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"datetime", "label"} | set(OHLCV)
    return [c for c in df.columns if c not in exclude]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (N, window, n_features), y: (N,), times: (N,) datetime index."""
    X, y, times = [], [], []
    
    # Pre-calculate windows to avoid slow iloc
    # Check for time continuity:
    # We require the time difference between the first and last row of the window to be roughly Window * 5 mins (allowing for small gaps/weekends? No, sessions are contiguous in filtered data normally).
    # Actually, filtered data (open session only) has huge gaps between days (11:30 -> 09:30 next day).
    # BUT, the model treats them as a sequence. The previous logic ignored 'day' boundaries, stitching close of day T to open of day T+1.
    # We want to maintain that behavior, BUT if there is a HUGE gap (e.g. 1 year), we should break.
    # Let's say max gap allowed between SESSIONS is e.g. 10 days. 
    # But wait, `dataset_builder` builds sequences of length WINDOW.
    # If standard operation stitches days, then row i and row i-1 are just consecutive bars in OUR dataset.
    # We only care if the actual time difference between row i and row i-1 is 'too large' implying a missing period we don't want to bridge.
    # The normal gap between 11:30 and 09:30 is ~22 hours.
    # The gap between Friday 11:30 and Monday 09:30 is ~2.5 days.
    # A gap of 1 year is > 300 days.
    # So we can check if adjacent rows have a gap > 7 days (e.g. holidays?).
    
    # Better: If gap > 10 days, reset sequence?
    # Actually, `build_sequences` iterates i from window to len.
    # If we find a gap > 10 days anywhere inside [i-window, i], we skip?
    # Optimization: Pre-compute gaps.
    
    # Vectorized gap check
    time_diffs = df["datetime"].diff()
    # Mask of 'bad gaps' (> 10 days)
    # 10 days = 10 * 24 * 3600 seconds
    bad_gaps = time_diffs > pd.Timedelta(days=14) # generous for holidays
    
    # We can't easily vectorize the window check without rolling, but iterative is fine for this dataset size (~10k rows).
    
    # But wait, `time_diffs` aligns with index.
    # If `bad_gaps[k]` is True, then there is a gap between k-1 and k.
    # Any window covering k (i.e. i > k and i-window < k) is invalid.
    
    bad_indices = bad_gaps[bad_gaps].index.tolist()
    
    for i in range(window, len(df)):
        # Check if this window overlaps a bad gap
        # Fast check: are there any bad_indices in range (i-window+1, i+1)?
        # Since we iterate, we can track the 'last bad gap'.
        # If (i - window) < last_bad_gap <= i, skip.
        
        # Let's simple check:
        # subset times
        t_block = df.iloc[i - window : i]["datetime"]
        # Check total span? No, stitching days is fine.
        # Check max diff inside block?
        # We only care if there is a jump > 14 days inside the block.
        # Since we just computed `bad_gaps`, we can use it.
        
        # Optimization: use numpy searchsorted or just iterate
        # Assuming bad_indices is sorted.
        
        has_gap = False
        # Range of rows involved: [i-window, i-1] (inclusive, 0-indexed relative to df)
        # We need to check gaps at indices: i-window+1 to i. 
        # (Gap at k means diff between k-1 and k).
        
        # Let's use a simpler heuristic for now: check start and end time?
        # No, start/end doesn't reveal internal gaps.
        
        pass 
        # Refactoring to basic loop for clarity and safety with new data
        
    dates = df["datetime"].values
    last_gap_idx = -1
    
    # Pre-find all Gap Indices (where diff > 10 days)
    # Using numpy for speed
    dt64 = dates.astype("datetime64[ns]")
    diffs = dt64[1:] - dt64[:-1]
    # 14 days in ns
    threshold = np.timedelta64(14, 'D')
    gap_indices = np.where(diffs > threshold)[0] + 1 # +1 because diff is shifted
    
    # gap_indices contains 'k' where row k is >14d after row k-1.
    
    current_gap_ptr = 0
    
    for i in range(window, len(df)):
        start_idx = i - window
        end_idx = i # exclusive
        
        # Check if any gap index is in [start_idx + 1, end_idx]
        # We advance current_gap_ptr to skip gaps before the window
        while current_gap_ptr < len(gap_indices) and gap_indices[current_gap_ptr] <= start_idx:
            current_gap_ptr += 1
            
        if current_gap_ptr < len(gap_indices) and gap_indices[current_gap_ptr] < end_idx:
            # Found a gap inside the window
            continue
            
        lab = df.iloc[i]["label"]
        if pd.isna(lab):
            continue
            
        block = df.iloc[start_idx:end_idx]
        X.append(block[feature_cols].values.astype(np.float32))
        y.append(int(lab))
        times.append(df.iloc[i]["datetime"])
        
    return np.array(X), np.array(y), np.array(times)


def run_dataset_builder() -> dict:
    df = load_labeled()
    feature_cols = get_feature_columns(df)
    df = df.sort_values("datetime").reset_index(drop=True)

    train_start = pd.Timestamp(TRAIN_START, tz="UTC")

    if BACKTEST_DAYS is not None:
        # Backtest = last N days of data (e.g. latest 3 months from MetaTrader)
        backtest_end = df["datetime"].max()
        backtest_start = backtest_end - pd.Timedelta(days=BACKTEST_DAYS)
        # Train ends the day before backtest so no overlap
        train_end_cap = backtest_start - pd.Timedelta(days=1)
        train_end = min(pd.Timestamp(TRAIN_END, tz="UTC"), train_end_cap)
    else:
        backtest_start = pd.Timestamp(BACKTEST_START, tz="UTC")
        backtest_end = pd.Timestamp(BACKTEST_END, tz="UTC")
        train_end = pd.Timestamp(TRAIN_END, tz="UTC")

    df_train = df[(df["datetime"] >= train_start) & (df["datetime"] <= train_end)]
    df_backtest = df[(df["datetime"] >= backtest_start) & (df["datetime"] <= backtest_end)]

    X_train, y_train, t_train = build_sequences(df_train, feature_cols, WINDOW)
    X_backtest, y_backtest, t_backtest = build_sequences(df_backtest, feature_cols, WINDOW)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        PROCESSED_DIR / "train.npz",
        X=X_train, y=y_train, times=t_train,
        feature_cols=feature_cols,
    )
    np.savez_compressed(
        PROCESSED_DIR / "backtest.npz",
        X=X_backtest, y=y_backtest, times=t_backtest,
        feature_cols=feature_cols,
    )
    pd.Series(feature_cols).to_csv(PROCESSED_DIR / "feature_cols.csv", index=False)

    return {
        "train": (X_train.shape, y_train.shape),
        "backtest": (X_backtest.shape, y_backtest.shape),
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    run_dataset_builder()
