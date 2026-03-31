"""
Stability analysis: Daily PnL reliability, worst day, and consistency.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

from utils import RESULTS_ROOT
from evaluation.backtester import load_backtest_data_and_predictions

def run_stability_analysis(model_name: str = "lstm", spread_points: float = 1.0, commission_points: float = 0.0) -> dict:
    # returns: array of R-multiples for every bar
    r_bar, times, _, _, _, _, _, _ = load_backtest_data_and_predictions(
        model_name, spread_points=spread_points, commission_points=commission_points
    )
    # Organize by day
    days = pd.Series(pd.to_datetime(times)).dt.date.values
    daily_pnl = defaultdict(float)
    
    for i, d in enumerate(days):
        ret = r_bar[i]
        if ret != 0:
            daily_pnl[d] += ret
                
    # Metrics
    unique_days = sorted(list(set(days)))
    pnl_vector = [daily_pnl[d] for d in unique_days]
    
    mean_daily_pnl = np.mean(pnl_vector)
    std_daily_pnl = np.std(pnl_vector)
    worst_day_pnl = np.min(pnl_vector) if pnl_vector else 0.0
    best_day_pnl = np.max(pnl_vector) if pnl_vector else 0.0
    prob_loss_day = np.mean([1 if x < 0 else 0 for x in pnl_vector])
    
    out = {
        "mean_daily_pnl": float(mean_daily_pnl),
        "std_daily_pnl": float(std_daily_pnl),
        "worst_day_pnl": float(worst_day_pnl),
        "best_day_pnl": float(best_day_pnl),
        "prob_losing_day": float(prob_loss_day),
        "total_days": len(unique_days)
    }
    
    out_dir = RESULTS_ROOT / "stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pnl_vector, index=unique_days, columns=["pnl"]).to_csv(out_dir / f"{model_name}_daily_pnl.csv")
    pd.DataFrame([out]).to_csv(out_dir / f"{model_name}_stability_metrics.csv", index=False)
    
    print(f"[{model_name}] Stability (spread={spread_points})")
    print(f"  Mean Daily PnL: {mean_daily_pnl:.2f} R")
    print(f"  Std Daily PnL:  {std_daily_pnl:.2f} R")
    print(f"  Worst Day:      {worst_day_pnl:.2f} R")
    print(f"  Prob Losing Day:{prob_loss_day:.1%}")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="lstm")
    parser.add_argument("--spread", type=float, default=1.0)
    args = parser.parse_args()
    run_stability_analysis(args.model, spread_points=args.spread)
