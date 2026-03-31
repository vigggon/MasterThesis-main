"""
Visualizations: Backtest (2024-01-01 to 2024-08-09) only. Forward test = live only (no file).
Equity curves, daily return distributions, stability bar chart.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import RESULTS_ROOT


def load_equity(model_name: str) -> pd.DataFrame:
    path = RESULTS_ROOT / "backtest" / f"{model_name}_equity.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["model"] = model_name
    return df


def load_metrics(model_name: str) -> dict:
    path = RESULTS_ROOT / "backtest" / f"{model_name}_metrics.csv"
    if not path.exists():
        return {}
    return pd.read_csv(path).iloc[0].to_dict()


def plot_equity_comparison():
    """Equity curves: LSTM vs Transformer (backtest only)."""
    out_dir = RESULTS_ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for name in ["lstm", "transformer"]:
        df = load_equity(name)
        if df.empty:
            continue
        df = df.sort_values("datetime")
        # Use equity_pct (percentage) instead of 'equity'
        ax.plot(df["datetime"], df["equity_pct"], label=name.capitalize())
    ax.set_ylabel("Equity (100% = starting capital)")
    ax.set_xlabel("Date")
    ax.set_title("Backtest Equity Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "equity_backtest.png", dpi=150)
    plt.close(fig)


def plot_daily_return_distributions():
    """Daily return distributions: Backtest per model."""
    out_dir = RESULTS_ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, model in zip(axes, ["lstm", "transformer"]):
        path = RESULTS_ROOT / "backtest" / f"{model}_equity.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        daily = df.groupby("date")["return"].sum()
        ax.hist(daily, bins=20, alpha=0.6, label="Backtest", density=True, color='steelblue', edgecolor='black')
        ax.set_title(model.capitalize(), fontsize=12, fontweight='bold')
        ax.set_xlabel("Daily return (units)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Daily Return Distributions", fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "daily_return_distributions.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_performance_degradation():
    """Bar chart: Backtest metrics (Sharpe, max drawdown, etc.) per model."""
    out_dir = RESULTS_ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in ["lstm", "transformer"]:
        m = load_metrics(model)
        if m:
            m["model"] = model
            rows.append(m)
    if not rows:
        return
    df = pd.DataFrame(rows)
    metrics = ["total_return", "sharpe_ratio", "max_drawdown", "profit_factor"]
    metrics = [m for m in metrics if m in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, m in zip(axes[: len(metrics)], metrics):
        df.set_index("model")[m].plot(kind="bar", ax=ax)
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=0)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_backtest.png", dpi=150)
    plt.close(fig)


def plot_stability_bars():
    """Stability metrics bar chart (backtest only). Lower is better for std metrics."""
    out_dir = RESULTS_ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in ["lstm", "transformer"]:
        path = RESULTS_ROOT / "stability" / f"{model}_backtest_stability.csv"
        if not path.exists():
            path = RESULTS_ROOT / "backtest" / f"{model}_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        d = df.iloc[0].to_dict()
        d["model"] = model
        rows.append(d)
    if not rows:
        return
    df = pd.DataFrame(rows)
    
    # Plot each stability metric
    for col in ["daily_accuracy_std", "confidence_std"]:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(df["model"], df[col], color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel(col.replace('_', ' ').title())
        ax.set_title(f"{col.replace('_', ' ').title()} (Lower = More Stable)")
        ax.tick_params(axis="x", rotation=0)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(out_dir / f"stability_{col}.png", dpi=150)
        plt.close(fig)
    
    # Skip worst_day_accuracy if all zeros or missing
    if "worst_day_accuracy" in df.columns and df["worst_day_accuracy"].sum() > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(df["model"], df["worst_day_accuracy"], color=['#1f77b4', '#ff7f0e'])
        ax.set_ylabel("Worst Day Accuracy")
        ax.set_title("Worst Day Accuracy (Higher = More Robust)")
        ax.tick_params(axis="x", rotation=0)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(out_dir / f"stability_worst_day_accuracy.png", dpi=150)
        plt.close(fig)
    
    # Combined stability comparison
    if "daily_accuracy_std" in df.columns and "confidence_std" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df["daily_accuracy_std"], width, label='Daily Accuracy Std', color='#1f77b4')
        bars2 = ax.bar(x + width/2, df["confidence_std"], width, label='Confidence Std', color='#ff7f0e')
        
        ax.set_ylabel('Standard Deviation (Lower = More Stable)')
        ax.set_title('Stability Comparison: LSTM vs Transformer')
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"].str.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        fig.tight_layout()
        fig.savefig(out_dir / "stability_comparison.png", dpi=150)
        plt.close(fig)


def run_all_visualizations():
    plot_equity_comparison()
    plot_daily_return_distributions()
    plot_performance_degradation()
    plot_stability_bars()


if __name__ == "__main__":
    run_all_visualizations()
