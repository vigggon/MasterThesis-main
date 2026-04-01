"""
Analyze losing days: correlate Daily PnL with average daily features (ATR, RSI, Volume, etc.).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

from utils import PROCESSED_DIR, RESULTS_ROOT
from backtesting.backtester import load_backtest_data_and_predictions

def run_loss_analysis(model_name: str = "lstm", spread_points: float = 1.0):
    # Load data
    returns, times, _, _, _, _, _, _ = load_backtest_data_and_predictions(
        model_name, spread_points=spread_points
    )
    
    # Load features explicitly to get columns like RSI
    data = np.load(PROCESSED_DIR / "backtest.npz", allow_pickle=True)
    X = data["X"]
    feature_cols = list(data["feature_cols"])
    
    # Create valid feature map
    # We want to aggregate features by day and see if "High ATR" or "Low RSI" days are the losers.
    
    # 1. Organize by day
    days = pd.Series(pd.to_datetime(times)).dt.date.values
    
    daily_stats = defaultdict(lambda: {
        "pnl": 0.0, 
        "count": 0, 
        "atr_mean": [], 
        "rsi_mean": [],
        "vol_ratio_mean": [],
        "regime_mean": []
    })
    
    # Indices for features
    try:
        idx_atr = feature_cols.index("atr_14")
        idx_rsi = feature_cols.index("rsi_14")
        idx_vol = feature_cols.index("volume_ratio_10") # or similar
        idx_regime = feature_cols.index("atr_regime")
    except ValueError:
        print("Required features not found. Check feature_cols.")
        return

    for i, d in enumerate(days):
        # Stats per bar
        atr_val = X[i, -1, idx_atr]
        rsi_val = X[i, -1, idx_rsi]
        vol_val = X[i, -1, idx_vol]
        
        daily_stats[d]["count"] += 1
        daily_stats[d]["pnl"] += returns[i]
        daily_stats[d]["atr_mean"].append(atr_val)
        daily_stats[d]["rsi_mean"].append(rsi_val)
        daily_stats[d]["vol_ratio_mean"].append(vol_val)
        daily_stats[d]["regime_mean"].append(X[i, -1, idx_regime])
        
    # Aggregate
    df_rows = []
    for d, stats in daily_stats.items():
        if stats["count"] == 0: continue
        row = {
            "date": d,
            "pnl": stats["pnl"],
            "trades": stats["count"], # Bars, actually. Need trades from pred? 
            # The returns[i] is 0 if no trade. So pnl is correct.
            # But let's check active trades count.
            "atr": np.mean(stats["atr_mean"]),
            "rsi": np.mean(stats["rsi_mean"]),
            "vol": np.mean(stats["vol_ratio_mean"]),
            "regime": np.mean(stats["regime_mean"])
        }
        df_rows.append(row)
        
    df = pd.DataFrame(df_rows)
    df["is_loser"] = df["pnl"] < 0
    df["is_bad_loser"] = df["pnl"] < -5.0 # Arbitrary threshold for "Bad Day"
    
    print("--- Correlation with PnL ---")
    print(df[["pnl", "atr", "rsi", "vol", "regime"]].corr()["pnl"])
    
    print("\n--- Low Volatility Analysis ---")
    bins = [0.0, 0.8, 1.0, 1.2, 1.5, 99.0]
    labels = ["<0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", ">1.5"]
    df["regime_bin"] = pd.cut(df["regime"], bins=bins, labels=labels)
    print(df.groupby("regime_bin")["pnl"].agg(["mean", "count", "sum"]))
    
    print("\n--- Bad Loser Days (<-5R) stats vs Normal ---")
    bad_days = df[df["is_bad_loser"]]
    good_days = df[~df["is_bad_loser"]]
    
    print(f"Bad Days: {len(bad_days)}")
    print(f"Good Days: {len(good_days)}")
    print("\nMean ATR:")
    print(f"  Bad: {bad_days['atr'].mean():.4f}")
    print(f"  Good: {good_days['atr'].mean():.4f}")
    
    print("\nMean RSI:")
    print(f"  Bad: {bad_days['rsi'].mean():.1f}")
    print(f"  Good: {good_days['rsi'].mean():.1f}")
    
    print("\nMean Volume Ratio:")
    print(f"  Bad: {bad_days['vol'].mean():.2f}")
    print(f"  Good: {good_days['vol'].mean():.2f}")

    out_dir = RESULTS_ROOT / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "daily_loss_analysis.csv", index=False)

if __name__ == "__main__":
    run_loss_analysis()
