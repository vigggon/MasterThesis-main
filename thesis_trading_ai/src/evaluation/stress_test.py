"""
Stress Testing Suite for Thesis Trading AI.
Tests:
1. Monte Carlo Permutation (1000x)
2. Yearly / Regime Breakdown
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from backtesting.backtester import load_backtest_data_and_predictions, equity_curve, max_drawdown, sharpe
from utils import RESULTS_ROOT, MIN_ATR, DAILY_STOP_R, DAILY_TP_R, BEST_TRANSFORMER_DIR, BEST_LSTM_DIR

def run_monte_carlo(returns, n_sims=1000):
    """
    Shuffle trade order n_sims times.
    Compare actual Max DD and Sharpe to random distribution.
    """
    print(f"\n--- Test 4: Monte Carlo Permutation ({n_sims} runs) ---")
    
    # Filter for actual trades (non-zero returns)
    trades = returns[returns != 0]
    n_trades = len(trades)
    
    if n_trades < 10:
        print("Not enough trades for Monte Carlo.")
        return

    actual_sharpe = sharpe(returns)
    actual_dd = max_drawdown(equity_curve(returns))
    actual_profit = returns.sum()
    
    sim_sharpes = []
    sim_dds = []
    sim_profits = []
    
    np.random.seed(42)
    
    for _ in range(n_sims):
        shuffled = np.random.permutation(trades)
        # Reconstruct full return series (zeros matter for Sharpe denominator if time-based, 
        # but for per-trade metrics we usually shuffle just trades. 
        # Here we place trades back into a time series? No, standard MC shuffles the sequence of PnL.)
        
        # Simple shuffle of the PnL vector
        sim_equity = np.cumsum(shuffled)
        
        # Metrics
        # Note: Sharpe on just trades vs Sharpe on time series. 
        # If we just shuffle trades, we lose time density info.
        # But for "Lucky Sequencing", we care about Equity Curve shape.
        
        s_dd = max_drawdown(sim_equity) # Unit based
        s_profit = shuffled.sum() # Should be identical to actual
        
        sim_dds.append(s_dd)
        sim_profits.append(s_profit)
        
    sim_dds = np.array(sim_dds)
    
    # Z-Score for Drawdown
    # actual_dd is usually negative (e.g. -1.28). sim_dds are negative.
    # If actual_dd is lower (more negative) than mean, it's worse.
    # If actual_dd is higher (less negative), it's better.
    
    mean_dd = sim_dds.mean()
    std_dd = sim_dds.std()
    z_score_dd = (actual_dd - mean_dd) / std_dd if std_dd > 0 else 0
    
    print(f"Actual Profit: {actual_profit:.2f} R")
    print(f"Actual Max DD: {actual_dd:.2f} R")
    print(f"MC Mean Max DD: {mean_dd:.2f} R (Std: {std_dd:.2f})")
    print(f"Distance from Mean: {z_score_dd:.2f} sigmas")
    
    pct_better = (sim_dds > actual_dd).mean() # Percentage of random runs that had SMALLER drawdown (less negative)
    pct_worse = (sim_dds < actual_dd).mean()
    
    print(f"Probability of Random Run having WORSE Drawdown: {pct_worse:.1%}")
    print(f"Probability of Random Run having BETTER Drawdown: {pct_better:.1%}")
    
    if z_score_dd > 0:
        print("Verdict: Actual curve is MORE STABLE than random chance (Good).")
    else:
        print("Verdict: Actual curve is RISKIER than random chance (Bad).")

def run_yearly_breakdown(returns, times):
    """
    Split returns by year.
    """
    print(f"\n--- Test 2: Yearly Breakdown ---")
    df = pd.DataFrame({"time": times, "ret": returns})
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    
    years = df["year"].unique()
    for y in sorted(years):
        sub = df[df["year"] == y]
        r = sub["ret"].values
        prof = r.sum()
        dd = max_drawdown(equity_curve(r))
        trds = np.count_nonzero(r)
        
        print(f"Year {y}: Profit {prof:>6.2f} R | Trades {trds:>4} | Max DD {dd:>6.2f} R")

def main():
    model = "transformer"
    print(f"Loading backtest data for {model}...")
    
    # Load data
    returns, times, pred, y, probs, atr_values, _, _ = load_backtest_data_and_predictions(
        model, 
        spread_points=1.0, 
        min_atr=MIN_ATR, 
        daily_max_loss=DAILY_STOP_R, 
        daily_take_profit=DAILY_TP_R
    )
    
    run_yearly_breakdown(returns, times)
    run_monte_carlo(returns)

if __name__ == "__main__":
    main()
