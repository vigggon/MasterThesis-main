import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/backtest")

def calc_sharpe(returns, risk_free=0.0):
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns - risk_free) / np.std(returns, ddof=1) * np.sqrt(252)

def calc_profit_factor(returns):
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    if gross_loss == 0:
        return np.inf
    return gross_profit / gross_loss

def bootstrap_ci(returns, statistic_fn, n_bootstrap=1000, alpha=0.05):
    bootstrapped_stats = []
    n = len(returns)
    np.random.seed(42)
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        stat = statistic_fn(sample)
        bootstrapped_stats.append(stat)
    
    bootstrapped_stats = np.array(bootstrapped_stats)
    bootstrapped_stats = bootstrapped_stats[np.isfinite(bootstrapped_stats)]
    lower = np.percentile(bootstrapped_stats, (alpha / 2) * 100)
    upper = np.percentile(bootstrapped_stats, (1 - alpha / 2) * 100)
    return lower, upper

def main():
    print("Bootstrap 95% CI for Backtest Results (n=1000 iter)")
    print("-" * 60)
    models = ["transformer", "bilstm"]
    
    for model in models:
        eq_path = RESULTS_DIR / f"{model}_equity.csv"
        trades_path = RESULTS_DIR / f"{model}_trades.csv"
        
        df_eq = pd.read_csv(eq_path)
        df_eq['date'] = pd.to_datetime(df_eq['timestamp']).dt.date
        
        # We need the equity before trading started to get the first return
        # But pct_change() is fine, it just misses the first day.
        # Actually backtester uses cumulative percentage return of initial balance. 
        # But let's just use simple pct_change.
        daily_equity = df_eq.groupby('date')['equity'].last()
        daily_r = daily_equity.pct_change().dropna().values
        
        sharpe = calc_sharpe(daily_r)
        sharpe_lower, sharpe_upper = bootstrap_ci(daily_r, calc_sharpe)
        
        df_trades = pd.read_csv(trades_path)
        df_trades['net_r'] = df_trades['pnl'] / df_trades['atr_14']
        trade_r = df_trades['net_r'].values
        
        pf = calc_profit_factor(trade_r)
        pf_lower, pf_upper = bootstrap_ci(trade_r, calc_profit_factor)
        
        print(f"{model.upper()}")
        print(f"  Sharpe Ratio:  {sharpe:.3f} [95% CI: {sharpe_lower:.3f} - {sharpe_upper:.3f}]")
        print(f"  Profit Factor: {pf:.3f} [95% CI: {pf_lower:.3f} - {pf_upper:.3f}]")
        print()

if __name__ == "__main__":
    main()
