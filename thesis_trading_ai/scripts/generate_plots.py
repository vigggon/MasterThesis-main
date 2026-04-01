import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.config_loader import RESULTS_ROOT

def calc_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return dd

def plot_drawdown_comparison(df_eq_bt, df_eq_lv, model_name, plot_dir):
    plt.figure(figsize=(10, 6))
    if df_eq_bt is not None and not df_eq_bt.empty:
        idx = pd.to_datetime(df_eq_bt.get('timestamp', df_eq_bt.index), format='mixed', utc=True)
        dd = df_eq_bt['drawdown'] * 100
        plt.plot(idx, dd, label=f'Backtest Drawdown', alpha=0.6)
    if df_eq_lv is not None and not df_eq_lv.empty:
        idx = pd.to_datetime(df_eq_lv.get('timestamp', df_eq_lv.index), format='mixed', utc=True)
        dd = df_eq_lv['drawdown'] * 100
        plt.plot(idx, dd, label=f'Live Drawdown', linewidth=2)
    plt.title(f"{model_name}: Drawdown Comparison (Regime Stability)")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"drawdown_comparison_{model_name.lower()}.pdf")
    plt.close()

def plot_trade_returns_histogram(df_trades, model_name, plot_dir, prefix):
    if df_trades is None or df_trades.empty: return
    plt.figure(figsize=(10, 6))
    col = 'return_pct' if 'return_pct' in df_trades else 'pnl_r'
    plt.hist(df_trades[col], bins=40, color='teal', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"{model_name} {prefix} Trade Returns Distribution")
    plt.xlabel("Return (%)" if 'return_pct' in df_trades else "Return (R)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(plot_dir / f"returns_hist_{model_name.lower()}_{prefix.lower()}.pdf")
    plt.close()

def plot_volatility_performance_overlay(df_trades, model_name, plot_dir):
    if df_trades is None or df_trades.empty: return
    # Use normalized ATR if available
    vol_col = 'atr_14_pct' if 'atr_14_pct' in df_trades else ('atr_14' if 'atr_14' in df_trades else None)
    if not vol_col: return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_trades[vol_col], df_trades['return_pct'], alpha=0.5, c='darkblue')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{model_name}: Volatility vs. Performance (Regime Sensitivity)")
    plt.xlabel("Volatility (ATR %)")
    plt.ylabel("Trade Return (%)")
    
    # Add trendline
    z = np.polyfit(df_trades[vol_col], df_trades['return_pct'], 1)
    p = np.poly1d(z)
    plt.plot(df_trades[vol_col], p(df_trades[vol_col]), "r--", alpha=0.8, label='Trend')
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"volatility_overlay_{model_name.lower()}.pdf")
    plt.close()

def plot_equity_and_drawdown(df_equity_tx, df_equity_lstm, plot_dir, prefix):
    plt.style.use('bmh')
    
    # Equity
    plt.figure(figsize=(10, 6))
    if df_equity_tx is not None:
        idx = pd.to_datetime(df_equity_tx['timestamp'], format='mixed', utc=True) if 'timestamp' in df_equity_tx else pd.to_datetime(df_equity_tx['datetime'], format='mixed', utc=True)
        eq = df_equity_tx['equity'] if 'equity' in df_equity_tx else df_equity_tx['equity_units']
        plt.plot(idx, eq, label='Transformer Equity')
    if df_equity_lstm is not None:
        idx = pd.to_datetime(df_equity_lstm['timestamp'], format='mixed', utc=True) if 'timestamp' in df_equity_lstm else pd.to_datetime(df_equity_lstm['datetime'], format='mixed', utc=True)
        eq = df_equity_lstm['equity'] if 'equity' in df_equity_lstm else df_equity_lstm['equity_units']
        plt.plot(idx, eq, label='BiLSTM Equity')
        
    plt.title(f"{prefix} Equity Curve (Transformer vs BiLSTM)")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{prefix.lower()}_equity_curve.pdf")
    plt.close()

    # Drawdown
    plt.figure(figsize=(10, 6))
    if df_equity_tx is not None:
        idx = pd.to_datetime(df_equity_tx['timestamp'], format='mixed', utc=True) if 'timestamp' in df_equity_tx else pd.to_datetime(df_equity_tx['datetime'], format='mixed', utc=True)
        eq = df_equity_tx['equity'] if 'equity' in df_equity_tx else df_equity_tx['equity_units']
        plt.plot(idx, calc_drawdown(eq.values) * 100, label='Transformer Drawdown')
    if df_equity_lstm is not None:
        idx = pd.to_datetime(df_equity_lstm['timestamp'], format='mixed', utc=True) if 'timestamp' in df_equity_lstm else pd.to_datetime(df_equity_lstm['datetime'], format='mixed', utc=True)
        eq = df_equity_lstm['equity'] if 'equity' in df_equity_lstm else df_equity_lstm['equity_units']
        plt.plot(idx, calc_drawdown(eq.values) * 100, label='BiLSTM Drawdown')
        
    plt.title(f"{prefix} Drawdown Curve (%)")
    plt.xlabel("Time")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{prefix.lower()}_drawdown_curve.pdf")
    plt.close()

def plot_returns_distribution(df_trades_tx, df_trades_lstm, plot_dir, prefix):
    plt.figure(figsize=(10, 6))
    if df_trades_tx is not None and not df_trades_tx.empty:
        col = 'return_pct' if 'return_pct' in df_trades_tx else 'pnl_r'
        plt.hist(df_trades_tx[col], bins=30, alpha=0.5, label='Transformer')
    if df_trades_lstm is not None and not df_trades_lstm.empty:
        col = 'return_pct' if 'return_pct' in df_trades_lstm else 'pnl_r'
        plt.hist(df_trades_lstm[col], bins=30, alpha=0.5, label='BiLSTM')
        
    plt.title(f"{prefix} Trade Returns Distribution")
    plt.xlabel("Return (%)" if prefix == "Live" else "Return R-Multiple")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{prefix.lower()}_returns_dist.pdf")
    plt.close()

def plot_rolling_sharpe(df_trades_tx, df_trades_lstm, plot_dir, prefix, window=10):
    plt.figure(figsize=(10, 6))
    
    def calc_rolling_sharpe(df):
        if df is None or len(df) < window: return None
        col = 'return_pct' if 'return_pct' in df else 'pnl_r'
        rets = df[col]
        rolling_mean = rets.rolling(window=window).mean()
        rolling_std = rets.rolling(window=window).std()
        # Annualized
        return (rolling_mean / rolling_std) * np.sqrt(252 * 6)

    if df_trades_tx is not None:
        rs = calc_rolling_sharpe(df_trades_tx)
        if rs is not None:
            plt.plot(rs, label='Transformer Rolling Sharpe')
            
    if df_trades_lstm is not None:
        rs = calc_rolling_sharpe(df_trades_lstm)
        if rs is not None:
            plt.plot(rs, label='BiLSTM Rolling Sharpe')
            
    plt.title(f"{prefix} Rolling Sharpe Ratio (Window={window} trades)")
    plt.xlabel("Trade Number")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{prefix.lower()}_rolling_sharpe.pdf")
    plt.close()

def do_monte_carlo(trades_df, title_prefix, plot_dir):
    if trades_df is None or trades_df.empty: return
    col = 'return_pct' if 'return_pct' in trades_df else 'pnl_r'
    
    trades = trades_df[col].values
    n_sims = 1000
    sim_sharpes = []
    
    actual_sharpe = np.mean(trades) / np.std(trades) * np.sqrt(252 * 6) if np.std(trades) != 0 else 0
    
    for _ in range(n_sims):
        shuffled = np.random.choice(trades, size=len(trades), replace=True)
        s = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252 * 6) if np.std(shuffled) != 0 else 0
        sim_sharpes.append(s)
        
    plt.figure(figsize=(10, 6))
    plt.hist(sim_sharpes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(actual_sharpe, color='red', linestyle='dashed', linewidth=2, label=f'Actual Sharpe ({actual_sharpe:.2f})')
    plt.title(f"Monte Carlo Sharpe Distribution ({n_sims} runs) - {title_prefix}")
    plt.xlabel("Simulated Sharpe Ratio")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"backtest_monte_carlo_sharpe_{title_prefix.lower()}.pdf")
    plt.close()

    # CVaR 95 function removed: relying correctly on backtester's R-multiple stats

def generate_live_stats(df_trades):
    if df_trades is None or df_trades.empty:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    col = 'return_pct' if 'return_pct' in df_trades else 'pnl_r'
    rets = df_trades[col].values
    
    n_trades = len(rets)
    wins = rets[rets > 0]
    losses_disp = rets[rets < 0]
    
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses_disp) if len(losses_disp) > 0 else 0
    median_return = np.median(rets)
    std_return = np.std(rets)
    max_win = np.max(rets)
    max_loss = np.min(rets)
    
    gains = wins.sum()
    los_abs = np.abs(losses_disp.sum())
    profit_factor = gains / los_abs if los_abs != 0 else float('inf')
    
    # Descriptive Sharpe
    mean_ret = np.mean(rets)
    sharpe = mean_ret / std_return * np.sqrt(252 * 6) if std_return != 0 else 0
    
    return n_trades, win_rate, avg_win, avg_loss, median_return, std_return, max_win, max_loss, profit_factor, sharpe

def load_data(path):
    if path.exists():
        return pd.read_csv(path)
    return None

def process_live_files():
    # If the user has live session PnL logs, convert them to standardized outputs
    live_in_dir = RESULTS_ROOT / "forward_test"
    live_out_dir = RESULTS_ROOT / "live"
    live_out_dir.mkdir(parents=True, exist_ok=True)
    
    for model in ["transformer", "lstm"]:
        session_file = live_in_dir / f"session_pnl_{model}.csv"
        if session_file.exists():
            df = pd.read_csv(session_file)
            if not df.empty:
                # Standardized column mapping
                df_out = pd.DataFrame()
                df_out['timestamp_open'] = df['open_time']
                df_out['timestamp_close'] = df['close_time']
                df_out['position'] = df['direction']
                df_out['entry_price'] = df['entry_price']
                df_out['exit_price'] = df['exit_price']
                df_out['pnl'] = df.get('pnl_points', 0.0)
                
                # Calculate return_pct if missing or recalibrate
                if 'entry_price' in df and 'exit_price' in df:
                    df_out['return_pct'] = ((df['exit_price'] - df['entry_price']) / df['entry_price'] * 100)
                    # Adjust for short position
                    mask_short = df['direction'].str.lower().str.contains('short')
                    df_out.loc[mask_short, 'return_pct'] *= -1
                else:
                    df_out['return_pct'] = 0.0
                
                out_name = "bilstm" if model == "lstm" else model
                df_out.to_csv(live_out_dir / f"{out_name}_trades.csv", index=False)
                
                # Setup equity to correctly map to 1.x account percentage!
                df_eq = pd.DataFrame()
                df_eq['timestamp'] = df['close_time']
                
                # Assume 0.5% risk per trade to translate R-units into account percentage
                df_eq['equity'] = 1.0 + np.cumsum(df_out['return_pct']) * 0.005
                peak_pct = np.maximum.accumulate(df_eq['equity'].values)
                df_eq['drawdown'] = (df_eq['equity'].values - peak_pct) / np.where(peak_pct > 0, peak_pct, 1)
                
                df_eq.to_csv(live_out_dir / f"{out_name}_equity.csv", index=False)

def main():
    plot_dir = RESULTS_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    backtest_dir = RESULTS_ROOT / "backtest"
    live_dir = RESULTS_ROOT / "live"
    
    # Optional preprocessing of existing raw live strings
    process_live_files()
    
    # Data loading
    bt_tx_trades = load_data(backtest_dir / "transformer_trades.csv")
    bt_tx_equity = load_data(backtest_dir / "transformer_equity.csv")
    bt_lstm_trades = load_data(backtest_dir / "bilstm_trades.csv")
    bt_lstm_equity = load_data(backtest_dir / "bilstm_equity.csv")
    
    lv_tx_trades = load_data(live_dir / "transformer_trades.csv")
    lv_tx_equity = load_data(live_dir / "transformer_equity.csv")
    lv_lstm_trades = load_data(live_dir / "bilstm_trades.csv")
    lv_lstm_equity = load_data(live_dir / "bilstm_equity.csv")
    
    # ------------------
    # BACKTEST (Statistical)
    # ------------------
    print("Generating Backtest Plots & Stats...")
    plot_equity_and_drawdown(bt_tx_equity, bt_lstm_equity, plot_dir, "Backtest")
    plot_returns_distribution(bt_tx_trades, bt_lstm_trades, plot_dir, "Backtest")
    plot_rolling_sharpe(bt_tx_trades, bt_lstm_trades, plot_dir, "Backtest")
    
    do_monte_carlo(bt_tx_trades, "Transformer", plot_dir)
    do_monte_carlo(bt_lstm_trades, "LSTM", plot_dir)
    
    bt_stats = []
    bt_tx_met = load_data(backtest_dir / "transformer_metrics.csv")
    bt_bilstm_met = load_data(backtest_dir / "bilstm_metrics.csv")
    
    if bt_tx_met is not None and not bt_tx_met.empty:
        r = bt_tx_met.iloc[0]
        bt_stats.append({"model": "transformer", "sharpe": r.sharpe_ratio, "profit_factor": r.profit_factor, "max_drawdown": r.max_drawdown_pct * 100.0, "cvar95": r.cvar95})
    
    if bt_bilstm_met is not None and not bt_bilstm_met.empty:
        r = bt_bilstm_met.iloc[0]
        bt_stats.append({"model": "bilstm", "sharpe": r.sharpe_ratio, "profit_factor": r.profit_factor, "max_drawdown": r.max_drawdown_pct * 100.0, "cvar95": r.cvar95})
        
    pd.DataFrame(bt_stats).to_csv(RESULTS_ROOT / "backtest_metrics.csv", index=False)
    
    # ------------------
    # LIVE TEST (Descriptive)
    # ------------------
    print("Generating Live Test Plots & Stats...")
    plot_equity_and_drawdown(lv_tx_equity, lv_lstm_equity, plot_dir, "Live")
    plot_returns_distribution(lv_tx_trades, lv_lstm_trades, plot_dir, "Live")
    # NO ROLLING SHARPE OR MONTE CARLO FOR LIVE.
    
    # Trade sequence cumulative PnL
    plt.figure(figsize=(10, 6))
    if lv_tx_trades is not None and not lv_tx_trades.empty:
        plt.plot(np.cumsum(lv_tx_trades['return_pct'].values), marker='o', label='Transformer Cum PnL')
    if lv_lstm_trades is not None and not lv_lstm_trades.empty:
        plt.plot(np.cumsum(lv_lstm_trades['return_pct'].values), marker='x', label='BiLSTM Cum PnL')
    plt.title("Live Test: Trade Sequence Cumulative PnL")
    plt.xlabel("Trade Sequence Number")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "live_trade_sequence_pnl.pdf")
    plt.close()
    
    lv_stats = []
    if lv_tx_trades is not None and not lv_tx_trades.empty:
        nt, wr, aw, al, md, sr, mw, ml, pf, sh = generate_live_stats(lv_tx_trades)
        lv_stats.append({"model": "transformer", "n_trades": nt, "win_rate": wr, "profit_factor": pf, "sharpe": sh})
    if lv_lstm_trades is not None and not lv_lstm_trades.empty:
        nt, wr, aw, al, md, sr, mw, ml, pf, sh = generate_live_stats(lv_lstm_trades)
        lv_stats.append({"model": "bilstm", "n_trades": nt, "win_rate": wr, "profit_factor": pf, "sharpe": sh})
        
    pd.DataFrame(lv_stats).to_csv(RESULTS_ROOT / "live_metrics.csv", index=False)

    # ------------------
    # EXECUTION FIDELITY (MT5)
    # ------------------
    print("Generating Execution Fidelity Analytics...")
    fidelity_dir = RESULTS_ROOT / "live" / "mt5_standardized"
    
    for model_name, lv_trades in [("transformer", lv_tx_trades), ("bilstm", lv_lstm_trades)]:
        mt5_path = fidelity_dir / f"{model_name}_mt5_trades.csv"
        if mt5_path.exists() and lv_trades is not None and not lv_trades.empty:
            df_mt5 = pd.read_csv(mt5_path)
            
            # Calculate account return based on starting balance (10,000)
            initial_balance = 10000.0
            n = min(len(lv_trades), len(df_mt5))
            sig_pnl = np.cumsum(lv_trades['return_pct'].iloc[:n].values)
            mt5_pnl = np.cumsum(df_mt5['pnl_net'].iloc[:n].values) / initial_balance * 100
            
            plt.figure(figsize=(10, 6))
            plt.plot(sig_pnl, marker='o', label='Theoretical Signal PnL', alpha=0.7)
            plt.plot(mt5_pnl, marker='x', label='Actual MT5 Realized PnL', linewidth=2)
            plt.fill_between(range(n), sig_pnl, mt5_pnl, color='gray', alpha=0.2, label='Execution Gap')
            
            plt.title(f"{model_name.upper()}: Execution Fidelity Comparison")
            plt.xlabel("Trade Sequence Number")
            plt.ylabel("Cumulative Return (%)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"execution_fidelity_{model_name.lower()}.pdf")
            plt.close()
            
            # Log the divergence
            gap = sig_pnl[-1] - mt5_pnl[-1]
            print(f"[{model_name}] Total Execution Gap: {gap:.4f}%")
            
            # --- NEW: MT5 Actuals Dedicated Plot ---
            plt.figure(figsize=(10, 6))
            idx_mt5 = pd.to_datetime(df_mt5['timestamp_open'].iloc[:n], format='mixed', utc=True)
            plt.plot(idx_mt5, mt5_pnl, marker='o', color='green' if mt5_pnl[-1] >= 0 else 'maroon', linewidth=2, label=f'Actual MT5 Realized ({mt5_pnl[-1]:.2f}%)')
            plt.axhline(0, color='red', linestyle='dashed')
            plt.title(f"{model_name.upper()}: Actual MT5 Live Forward Performance")
            plt.xlabel("Trade Execution Time")
            plt.ylabel("Cumulative Net Return (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / f"mt5_actual_{model_name.lower()}.pdf")
            plt.close()

    # ------------------
    # COMPARISON & HIGH IMPACT
    # ------------------
    print("Generating Comparisons & High Impact Plots...")
    for model, bt_eq, lv_eq, bt_tr, lv_tr in [
        ("Transformer", bt_tx_equity, lv_tx_equity, bt_tx_trades, lv_tx_trades), 
        ("BiLSTM", bt_lstm_equity, lv_lstm_equity, bt_lstm_trades, lv_lstm_trades)
    ]:
        # Equity Comparison
        if bt_eq is not None and lv_eq is not None and not bt_eq.empty and not lv_eq.empty:
            plt.figure(figsize=(12, 6))
            idx_b = pd.to_datetime(bt_eq['timestamp'], format='mixed', utc=True) if 'timestamp' in bt_eq else pd.to_datetime(bt_eq['datetime'], format='mixed', utc=True)
            eq_b = bt_eq['equity'] if 'equity' in bt_eq else bt_eq['equity_units']
            plt.plot(idx_b, eq_b, label=f'{model} Backtest', alpha=0.7)
            idx_l = pd.to_datetime(lv_eq['timestamp'], format='mixed', utc=True)
            pad = eq_b.iloc[-1] if not eq_b.empty else 0
            plt.plot(idx_l, lv_eq['equity'] + pad, label=f'{model} Live (Offset)', linewidth=2)
            plt.title(f"{model}: Backtest vs Live Comparison")
            plt.xlabel("Date")
            plt.ylabel("Abstract Equity Baseline")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"comparison_{model.lower()}.pdf")
            plt.close()

        # Drawdown Comparison
        plot_drawdown_comparison(bt_eq, lv_eq, model, plot_dir)
        
        # Histograms
        plot_trade_returns_histogram(bt_tr, model, plot_dir, "Backtest")
        plot_trade_returns_histogram(lv_tr, model, plot_dir, "Live")
        
        # Volatility Overlay (Using Backtest for larger sample)
        plot_volatility_performance_overlay(bt_tr, model, plot_dir)

    print("All tasks finished successfully. Results are in /plots and /results folders.")

if __name__ == "__main__":
    main()
