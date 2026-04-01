"""
Final Thesis Asset Generator.
Produces high-quality plots for the thesis document:
1. Equity Curves (Profit in R)
2. Monte Carlo Simulation Distributions
3. Comparison Chart
"""
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtesting.backtester import load_backtest_data_and_predictions, equity_curve, max_drawdown
from utils import RESULTS_ROOT, MIN_ATR, DAILY_STOP_R, DAILY_TP_R

def generate_assets():
    plot_dir = RESULTS_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    models = ["transformer", "lstm"]
    plt.style.use('bmh') # Clean professional look
    
    # 1. Individual Equity Curves
    for model_name in models:
        print(f"Plotting Equity Curve for {model_name}...")
        returns, times, _, _, _, _, _, _ = load_backtest_data_and_predictions(
            model_name, spread_points=1.0, min_atr=MIN_ATR,
            daily_max_loss=DAILY_STOP_R, daily_take_profit=DAILY_TP_R
        )
        curve = equity_curve(returns)
        
        plt.figure(figsize=(10, 6))
        plt.plot(pd.to_datetime(times), curve, label=f"{model_name.capitalize()} Equity", linewidth=2)
        plt.title(f"Backtest Equity Curve: {model_name.capitalize()}")
        plt.xlabel("Date")
        plt.ylabel("Profit (R-Multiples)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"equity_curve_{model_name}.pdf")
        plt.close()

    # 2. Performance Comparison
    print("Plotting Comparison Equity Curves...")
    plt.figure(figsize=(12, 7))
    for model_name in models:
        returns, times, _, _, _, _, _, _ = load_backtest_data_and_predictions(
            model_name, spread_points=1.0, min_atr=MIN_ATR,
            daily_max_loss=DAILY_STOP_R, daily_take_profit=DAILY_TP_R
        )
        curve = equity_curve(returns)
        plt.plot(pd.to_datetime(times), curve, label=model_name.upper())

    plt.title("Transformer vs LSTM Performance Comparison")
    plt.xlabel("Date")
    plt.ylabel("Profit (R-Multiples)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "performance_comparison.pdf")
    plt.close()

    # 3. Monte Carlo Distributions
    for model_name in models:
        print(f"Running Monte Carlo for {model_name}...")
        returns, _, _, _, _, _, _, _ = load_backtest_data_and_predictions(
            model_name, spread_points=1.0, min_atr=MIN_ATR,
            daily_max_loss=DAILY_STOP_R, daily_take_profit=DAILY_TP_R
        )
        trades = returns[returns != 0]
        n_sims = 1000
        sim_dds = []
        
        for _ in range(n_sims):
            shuffled = np.random.permutation(trades)
            sim_equity = np.cumsum(shuffled)
            sim_dds.append(max_drawdown(sim_equity))
            
        plt.figure(figsize=(10, 6))
        plt.hist(sim_dds, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        actual_dd = max_drawdown(np.cumsum(trades))
        plt.axvline(actual_dd, color='red', linestyle='dashed', linewidth=2, label=f'Actual Max DD ({actual_dd:.2f}R)')
        plt.title(f"Monte Carlo Drawdown Distribution: {model_name.capitalize()}")
        plt.xlabel("Max Drawdown (R)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"monte_carlo_{model_name}.pdf")
        plt.close()

    # 4. Daily Return Distributions (Stability)
    for model_name in models:
        print(f"Plotting Daily PnL Distribution for {model_name}...")
        returns, times, _, _, _, _, _, _ = load_backtest_data_and_predictions(
            model_name, spread_points=1.0, min_atr=MIN_ATR,
            daily_max_loss=DAILY_STOP_R, daily_take_profit=DAILY_TP_R
        )
        # Aggregated by day
        days = pd.Series(pd.to_datetime(times)).dt.date
        daily_pnl = pd.Series(returns).groupby(days).sum()
        
        plt.figure(figsize=(10, 6))
        plt.hist(daily_pnl, bins=25, color='seagreen', edgecolor='black', alpha=0.7)
        plt.axvline(daily_pnl.mean(), color='blue', linestyle='dashed', linewidth=2, label=f'Mean Daily PnL ({daily_pnl.mean():.2f}R)')
        plt.title(f"Daily PnL Distribution: {model_name.capitalize()}")
        plt.xlabel("Daily Profit (R)")
        plt.ylabel("Frequency (Days)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"daily_pnl_dist_{model_name}.pdf")
        plt.close()

    # 5. Stability Metrics Comparison
    print("Plotting Stability Comparison...")
    comparison_data = []
    for model_name in models:
        returns, times, _, _, _, _, _, _ = load_backtest_data_and_predictions(
            model_name, spread_points=1.0, min_atr=MIN_ATR,
            daily_max_loss=DAILY_STOP_R, daily_take_profit=DAILY_TP_R
        )
        days = pd.Series(pd.to_datetime(times)).dt.date
        daily_pnl = pd.Series(returns).groupby(days).sum()
        comparison_data.append({
            "Model": model_name.upper(),
            "Mean PnL": daily_pnl.mean(),
            "Volatility (Std)": daily_pnl.std(),
            "Worst Day": daily_pnl.min()
        })
    
    df_comp = pd.DataFrame(comparison_data).set_index("Model")
    ax = df_comp.plot(kind="bar", figsize=(12, 7), rot=0, color=['#4c72b0', '#55a868', '#c44e52'])
    plt.title("Comparison of Stability Metrics (Daily R)")
    plt.ylabel("R-Multiples")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "stability_metrics_comparison.pdf")
    plt.close()

    print(f"All assets generated in {plot_dir}")

if __name__ == "__main__":
    generate_assets()
