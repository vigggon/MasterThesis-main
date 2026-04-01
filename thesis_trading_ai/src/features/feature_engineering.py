"""
STEP 4 — Feature engineering for open session.
Resamples to 10m (per user request).
All features are scale-invariant (ratios, %, 0–1) so the model is not dominated by raw price/ATR.
Adds: session position, volume ratio, short-term trend, Bollinger position, normalized volatility.
Output: data/features/open_features.csv
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from pathlib import Path

from utils import SESSION_DIR, FEATURES_DIR, ATR_PERIOD


def load_session() -> pd.DataFrame:
    path = SESSION_DIR / "nasdaq_open_session.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run session_filter.py first. Expected: {path}")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


def resample_session_data(df: pd.DataFrame, timeframe_min: int = 10) -> pd.DataFrame:
    """Resample OHLCV data to a new timeframe (e.g., 10m)."""
    # Ensure datetime index
    df = df.set_index("datetime").sort_index()
    
    # Resample rules
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    
    # Resample (Pandas aligns to epoch by default; 09:30 aligns with 10T)
    resampled = df.resample(f"{timeframe_min}min").agg(agg_dict)
    
    # Drop rows with NaN (empty bins)
    resampled = resampled.dropna()
    
    # Reset index
    resampled = resampled.reset_index()
    
    print(f"Resampled {len(df)} source rows -> {len(resampled)} {timeframe_min}m rows.")
    return resampled


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def _safe_div(num: pd.Series, denom: pd.Series, fill: float = 0.0) -> pd.Series:
    return (num / denom.replace(0, np.nan)).fillna(fill)


def _clip(x: pd.Series, low: float, high: float) -> pd.Series:
    return x.clip(low, high)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine/cosine encoding of minute of day (0-1440)."""
    # Open starting at 9:30 AM = 570 minutes
    # Session is 2 hours long (120 minutes)
    dt = pd.to_datetime(df["datetime"])
    minutes = dt.dt.hour * 60 + dt.dt.minute
    # Cycle is 24 hours = 1440 minutes
    df["time_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    df["time_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    return df


def add_lagged_volatility(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Add lagged ATR and volatility measures to capture regiment changes."""
    # Assuming 'atr_14_pct' and 'rolling_std_20_pct' are already in features
    # We want to know if volatility is rising or falling
    for col in ["atr_14_pct", "rolling_std_20_pct"]:
        if col in features.columns:
            # Ratio of current vol to vol 30 mins ago (6 bars)
            features[f"{col}_ratio_6"] = _safe_div(features[col], features[col].shift(6), fill=1.0)
            features[f"{col}_ratio_6"] = _clip(features[f"{col}_ratio_6"], 0.5, 2.0)
    return features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    features = pd.DataFrame(index=df.index)

    # —— Returns (already scale-invariant) ——
    log_ret = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    features["log_return_1"] = log_ret
    features["log_return_3"] = np.log(c / c.shift(3)).replace([np.inf, -np.inf], np.nan)
    features["log_return_5"] = np.log(c / c.shift(5)).replace([np.inf, -np.inf], np.nan)
    features["log_return_6"] = np.log(c / c.shift(6)).replace([np.inf, -np.inf], np.nan)

    # —— Volatility as % of price (scale-invariant) ——
    roll_std_6 = c.rolling(6).std()
    roll_std_10 = c.rolling(10).std()
    roll_std_20 = c.rolling(20).std()
    features["rolling_std_6_pct"] = _safe_div(roll_std_6, c)
    features["rolling_std_10_pct"] = _safe_div(roll_std_10, c)
    features["rolling_std_20_pct"] = _safe_div(roll_std_20, c)
    features["range_6_pct"] = (h.rolling(6).max() - l.rolling(6).min()) / c.replace(0, np.nan)
    features["high_low_pct"] = (h - l) / l.replace(0, np.nan)
    features["body_pct"] = (c - o).abs() / (h - l).replace(0, np.nan)

    # ATR: keep raw for label_generator; add % for model
    atr_14 = atr(h, l, c, ATR_PERIOD)
    features["atr_14"] = atr_14
    features["atr_14_pct"] = _safe_div(atr_14, c)

    dt = pd.to_datetime(df["datetime"])
    day = dt.dt.date

    # —— Opening range (first 6 bars) ——
    or6_high = df.groupby(day)["high"].transform(lambda s: s.iloc[:6].max() if len(s) >= 6 else s.max())
    or6_low = df.groupby(day)["low"].transform(lambda s: s.iloc[:6].min() if len(s) >= 6 else s.min())
    features["distance_from_or6_high"] = (c - or6_high) / or6_high.replace(0, np.nan)
    features["distance_from_or6_low"] = (c - or6_low) / or6_low.replace(0, np.nan)

    # —— Session context: open, high/low, VWAP ——
    session_open = df.groupby(day)["open"].transform("first")
    session_high = df.groupby(day)["high"].cummax()
    session_low = df.groupby(day)["low"].cummin()
    features["distance_from_session_open"] = (c - session_open) / session_open.replace(0, np.nan)
    features["or_high_breakout_pct"] = (h - session_high.shift(1)) / session_high.shift(1).replace(0, np.nan)
    features["or_low_breakout_pct"] = (l - session_low.shift(1)) / session_low.shift(1).replace(0, np.nan)
    vwap = (df.assign(tpv=c * v).groupby(day)["tpv"].cumsum() /
            df.assign(tpv=v).groupby(day)["tpv"].cumsum())
    features["vwap_deviation"] = (c - vwap) / vwap.replace(0, np.nan)
    
    # —— [NEW] Gap / Overnight Context ——
    # Previous session close (last close of previous day)
    prev_session_close = df.groupby(day)["close"].transform("last").shift(1)
    # Overnight gap: how much price gapped from previous close to current session open
    features["overnight_gap_pct"] = _clip(
        (session_open - prev_session_close) / prev_session_close.replace(0, np.nan),
        -0.05, 0.05
    )
    # Gap direction (1 = gap up, -1 = gap down, 0 = flat)
    features["gap_direction"] = np.sign(features["overnight_gap_pct"])
    # Distance from previous session high/low (context)
    prev_session_high = df.groupby(day)["high"].transform("max").shift(1)
    prev_session_low = df.groupby(day)["low"].transform("min").shift(1)
    features["dist_from_prev_high_pct"] = _clip(
        (c - prev_session_high) / prev_session_high.replace(0, np.nan),
        -0.10, 0.10
    )
    features["dist_from_prev_low_pct"] = _clip(
        (c - prev_session_low) / prev_session_low.replace(0, np.nan),
        -0.10, 0.10
    )
    
    # —— [NEW] Volatility Regime ——
    # Compare current ATR to its 20-period moving average (is volatility expanding or contracting?)
    atr_ma_20 = atr_14.rolling(20, min_periods=1).mean()
    features["atr_regime"] = _clip(_safe_div(atr_14, atr_ma_20, fill=1.0), 0.5, 2.0)
    # Intraday range expansion (current bar range vs average)
    bar_range = h - l
    avg_bar_range = bar_range.rolling(20, min_periods=1).mean()
    features["range_expansion"] = _clip(_safe_div(bar_range, avg_bar_range, fill=1.0), 0.3, 3.0)

    # —— Session position: which bar in the open (0–1) ——
    bar_index = df.groupby(day).cumcount()
    bar_count = df.groupby(day).transform("size")
    features["session_bar_fraction"] = _safe_div(bar_index, (bar_count - 1).replace(0, np.nan), fill=0.0)
    features["session_bar_fraction"] = _clip(features["session_bar_fraction"], 0.0, 1.0)

    # —— Volume: ratio vs recent average (capped) ——
    vol_ma = v.rolling(10, min_periods=1).mean()
    features["volume_ratio_10"] = _clip(_safe_div(v, vol_ma, fill=1.0), 0.1, 3.0)
    
    # [NEW] Relative Volume (50-period context)
    vol_ma_50 = v.rolling(50, min_periods=1).mean()
    features["rel_vol_50"] = _clip(_safe_div(v, vol_ma_50, fill=1.0), 0.1, 5.0)

    # —— Short-term trend: 6-bar return per bar (scale-invariant) ——
    features["slope_6_pct"] = (c - c.shift(6)) / (6 * c.replace(0, np.nan))

    # —— Distance from recent high/low (room to run) ——
    rev_6_high = h.rolling(6).max()
    rev_6_low = l.rolling(6).min()
    features["dist_from_6h_pct"] = (c - rev_6_high) / rev_6_high.replace(0, np.nan)
    features["dist_from_6l_pct"] = (c - rev_6_low) / rev_6_low.replace(0, np.nan)
    
    # [NEW] Longer context returns (1 hour = 12 bars, 2 hours = 24 bars)
    features["log_return_12"] = np.log(c / c.shift(12)).replace([np.inf, -np.inf], np.nan)
    features["log_return_24"] = np.log(c / c.shift(24)).replace([np.inf, -np.inf], np.nan)

    # —— Indicators: RSI (0–100), EMA distance as %, Bollinger ——
    features["rsi_14"] = rsi(c, 14)
    ema_9 = ema(c, 9)
    ema_21 = ema(c, 21)
    features["distance_from_ema9_pct"] = (c - ema_9) / c.replace(0, np.nan)
    features["distance_from_ema21_pct"] = (c - ema_21) / c.replace(0, np.nan)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    features["bollinger_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    # Position within band (0 = at lower, 1 = at upper)
    features["bb_position"] = _clip((c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan), 0.0, 1.0)

    # —— Time features ——
    time_feats = add_time_features(df[["datetime"]])
    features["time_sin"] = time_feats["time_sin"]
    features["time_cos"] = time_feats["time_cos"]
    
    # [NEW] Day of Week (0=Monday, 6=Sunday)
    dow = dt.dt.dayofweek
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # —— Lagged Volatility ——
    features = add_lagged_volatility(df, features)

    features = features.bfill().fillna(0)
    # Clip extreme returns/ratios so a few bars don't dominate
    for col in ["log_return_1", "log_return_3", "log_return_5", "log_return_6", "slope_6_pct",
                "distance_from_ema9_pct", "distance_from_ema21_pct", "vwap_deviation",
                "distance_from_or6_high", "distance_from_or6_low", "distance_from_session_open",
                "or_high_breakout_pct", "or_low_breakout_pct", "dist_from_6h_pct", "dist_from_6l_pct",
                "atr_14_pct_ratio_6", "rolling_std_20_pct_ratio_6",
                "log_return_12", "log_return_24"]: # Added new cols to clip
        if col in features.columns:
            features[col] = _clip(features[col], -0.05, 0.05)
    return pd.concat([df[["datetime", "open", "high", "low", "close", "volume"]], features], axis=1)


def run_feature_engineering() -> pd.DataFrame:
    df = load_session()
    
    # [PIVOT] Resample to 10m per user request
    df = resample_session_data(df, timeframe_min=10)
    
    out = build_features(df)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "open_features.csv"
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    run_feature_engineering()
