"""
STEP 5 — Trade labeling: 3-class (hold, long, short).
SL=1×ATR, TP=3.0×ATR (break-even win rate 25.0%, 3:1 R:R to overcome transaction costs), max 12 candles. Long: TP above entry, SL below. Short: TP below entry, SL above.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from pathlib import Path

from utils import FEATURES_DIR, LABELED_DIR, ATR_PERIOD, SL_ATR_MULT, TP_ATR_MULT, MAX_HOLD_CANDLES


def load_features() -> pd.DataFrame:
    path = FEATURES_DIR / "open_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run feature_engineering.py first. Expected: {path}")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _simulate_long(row: pd.Series, future: pd.DataFrame, atr_col: str, sl_mult: float, tp_mult: float, max_hold: int) -> tuple:
    """Returns (win: bool, bar_index: int where decided)."""
    atr_val = row[atr_col]
    if pd.isna(atr_val) or atr_val <= 0:
        return False, max_hold
    entry = row["close"]
    sl_long = entry - atr_val * sl_mult
    tp_long = entry + atr_val * tp_mult
    for step, (_, r) in enumerate(future.iterrows()):
        if step >= max_hold:
            break
        low, high, close = r["low"], r["high"], r["close"]
        if high >= tp_long and low <= sl_long:
            return (close >= entry, step)
        if high >= tp_long:
            return (True, step)
        if low <= sl_long:
            return (False, step)
    return False, max_hold


def _simulate_short(row: pd.Series, future: pd.DataFrame, atr_col: str, sl_mult: float, tp_mult: float, max_hold: int) -> tuple:
    """Returns (win: bool, bar_index: int where decided). Short TP below entry, SL above."""
    atr_val = row[atr_col]
    if pd.isna(atr_val) or atr_val <= 0:
        return False, max_hold
    entry = row["close"]
    sl_short = entry + atr_val * sl_mult
    tp_short = entry - atr_val * tp_mult
    for step, (_, r) in enumerate(future.iterrows()):
        if step >= max_hold:
            break
        low, high, close = r["low"], r["high"], r["close"]
        if low <= tp_short and high >= sl_short:
            return (close <= entry, step)
        if low <= tp_short:
            return (True, step)
        if high >= sl_short:
            return (False, step)
    return False, max_hold


def label_candle_3class(
    row: pd.Series,
    future: pd.DataFrame,
    atr_col: str = "atr_14",
    sl_mult: float = SL_ATR_MULT,
    tp_mult: float = TP_ATR_MULT,
    max_hold: int = MAX_HOLD_CANDLES,
) -> int:
    """
    3-class: 0=hold, 1=long (TP before SL within max_hold), 2=short (short TP before short SL).
    If both long and short would win, assign the one that wins first (earlier bar); tie → long (1).
    """
    long_win, long_bar = _simulate_long(row, future, atr_col, sl_mult, tp_mult, max_hold)
    short_win, short_bar = _simulate_short(row, future, atr_col, sl_mult, tp_mult, max_hold)
    if long_win and not short_win:
        return 1
    if short_win and not long_win:
        return 2
    if long_win and short_win:
        return 1 if long_bar <= short_bar else 2
    return 0


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    labels = []
    for idx in range(len(df)):
        future = df.iloc[idx + 1 : idx + 1 + MAX_HOLD_CANDLES]
        if len(future) < 2:
            labels.append(0)
            continue
        lab = label_candle_3class(df.iloc[idx], future)
        labels.append(lab)
    df["label"] = labels
    return df


def run_label_generator() -> pd.DataFrame:
    df = load_features()
    out = generate_labels(df)
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(LABELED_DIR / "open_labeled.csv", index=False)
    return out


if __name__ == "__main__":
    run_label_generator()
