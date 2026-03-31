"""
Hyperparameter sweep: train + backtest all Transformer and LSTM configurations.
Outputs a thesis-ready comparison table to results/sweep_results.csv.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import time
import json
import numpy as np
import torch
from pathlib import Path

from utils import (
    EXPERIMENTS_ROOT, RESULTS_ROOT, PROCESSED_DIR, NUM_CLASSES,
    TRADE_THRESHOLD, TP_ATR_MULT, SL_ATR_MULT, MIN_ATR,
    RISK_PCT_PER_TRADE, MAX_DD_CAP, DAILY_STOP_R, MAX_HOLD_CANDLES,
)
from training.train import run_train
from models import get_lstm, get_transformer

# ── Hyperparameter grids ──────────────────────────────────────────────
TRANSFORMER_GRID = [
    {"num_layers": nl, "nhead": nh, "d_model": dm}
    for nl in [1, 2, 3]
    for nh in [4, 8]
    for dm in [64, 128]
]

LSTM_GRID = [
    {"num_layers": nl, "hidden_size": hs}
    for nl in [1, 2, 3]
    for hs in [32, 64, 128]
]

SPREAD_POINTS = 2.0  # Realistic spread + slippage


def _config_name(model: str, cfg: dict) -> str:
    """Unique name for the experiment directory."""
    if model == "transformer":
        return f"transformer_L{cfg['num_layers']}_H{cfg['nhead']}_D{cfg['d_model']}"
    else:
        return f"lstm_L{cfg['num_layers']}_U{cfg['hidden_size']}"


def _backtest_config(exp_dir: Path, model_name: str, cfg: dict) -> dict | None:
    """Run backtester on a single config. Returns metrics dict or None."""
    ckpt_path = exp_dir / "best.pt"
    if not ckpt_path.exists():
        return None

    import pandas as pd

    # Load backtest data
    data = np.load(PROCESSED_DIR / "backtest.npz", allow_pickle=True)
    X = torch.from_numpy(data["X"]).float()
    y = data["y"]
    times = data["times"]
    feature_cols = list(data["feature_cols"])

    # ATR
    try:
        atr_idx = feature_cols.index("atr_14")
        atr_values = X[:, -1, atr_idx].numpy()
    except ValueError:
        atr_values = np.ones(len(X))

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cols = ckpt.get("feature_cols", None)

    if ckpt_cols is not None:
        col_to_idx = {c: i for i, c in enumerate(feature_cols)}
        indices = [col_to_idx[c] for c in ckpt_cols if c in col_to_idx]
        X_model = X[:, :, indices]
        n_features = len(ckpt_cols)
    else:
        X_model = X
        n_features = X.shape[2]

    # Build model
    if model_name == "lstm":
        model = get_lstm(
            n_features,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=0.0,
            num_classes=NUM_CLASSES,
        )
    else:
        model = get_transformer(
            n_features,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dropout=0.0,
            num_classes=NUM_CLASSES,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Threshold
    th_path = exp_dir / "trade_threshold.json"
    if th_path.exists():
        with open(th_path) as f:
            th = float(json.load(f)["threshold"])
    else:
        th = TRADE_THRESHOLD

    # Predict
    with torch.no_grad():
        logits = model(X_model)
        probs = torch.softmax(logits, dim=1).numpy()

    # Import backtester functions
    from evaluation.backtester import _returns_from_probs, equity_curve, max_drawdown, sharpe, profit_factor

    # Load raw data for walk-forward
    raw_path = PROCESSED_DIR.parent / "features" / "open_features.csv"
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    else:
        df_raw = None

    returns = _returns_from_probs(
        probs, th,
        spread_points=SPREAD_POINTS,
        atr=atr_values,
        times=times,
        daily_max_loss=DAILY_STOP_R,
        min_atr=MIN_ATR,
        df_raw=df_raw,
    )

    n_trades = int(np.count_nonzero(returns))
    if n_trades == 0:
        return {
            "total_r": 0.0, "trades": 0, "win_rate": 0.0,
            "sharpe": 0.0, "profit_factor": 0.0,
            "max_dd_r": 0.0, "realistic_return_low": 0.0, "realistic_return_high": 0.0,
        }

    eq = equity_curve(returns)
    total_r = float(returns.sum())

    return {
        "total_r": round(total_r, 2),
        "trades": n_trades,
        "win_rate": round(float((returns[returns != 0] > 0).mean()) * 100, 1),
        "sharpe": round(float(sharpe(returns)), 3),
        "profit_factor": round(float(profit_factor(returns)), 2),
        "max_dd_r": round(float(max_drawdown(eq)), 2),
        "realistic_return_low": round(total_r * 0.5, 1),   # 0.5% risk (Std Professional)
        "realistic_return_high": round(total_r * 0.75, 1), # 0.75% risk (Aggressive)
    }


def run_sweep():
    """Run the full hyperparameter sweep."""
    results = []
    out_csv = RESULTS_ROOT / "sweep_results.csv"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # All possible keys for the CSV header
    header_keys = ["model", "num_layers", "nhead", "d_model", "hidden_size", 
                   "total_r", "trades", "win_rate", "sharpe", "profit_factor", 
                   "max_dd_r", "realistic_return_low", "realistic_return_high"]

    # If CSV exists, read existing results to avoid duplicate work and for header consistency
    existing_names = set()
    if out_csv.exists():
        with open(out_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
                cfg = {k: v for k, v in row.items() if k in ["num_layers", "nhead", "d_model", "hidden_size"]}
                # conversion to int where applicable
                for k in cfg: 
                    try: cfg[k] = int(cfg[k])
                    except: pass
                existing_names.add(_config_name(row["model"], cfg))

    total = len(TRANSFORMER_GRID) + len(LSTM_GRID)
    print(f"=== Hyperparameter Sweep: {total} configurations ===\n")

    full_grid = [("transformer", c) for c in TRANSFORMER_GRID] + [("lstm", c) for c in LSTM_GRID]

    for idx, (m_type, cfg) in enumerate(full_grid, 1):
        name = _config_name(m_type, cfg)
        if name in existing_names:
            print(f"[{idx}/{total}] Skipping {name} (already in CSV)")
            continue

        exp_dir = EXPERIMENTS_ROOT / name
        
        # Check if already trained but not in CSV
        if (exp_dir / "trade_threshold.json").exists():
            print(f"[{idx}/{total}] {name} already trained. Running backtest ...")
        else:
            print(f"\n[{idx}/{total}] Training {name} ...")
            t0 = time.time()
            try:
                train_args = {
                    "model_name": m_type,
                    "exp_dir": str(exp_dir),
                    "tune_threshold": True,
                    "min_atr": MIN_ATR,
                    "transaction_cost": 0.1,
                }
                if m_type == "transformer":
                    train_args.update({"d_model": cfg["d_model"], "nhead": cfg["nhead"], "tx_layers": cfg["num_layers"]})
                else:
                    train_args.update({"hidden_size": cfg["hidden_size"], "num_layers": cfg["num_layers"]})
                
                run_train(**train_args)
            except Exception as e:
                print(f"  TRAIN FAILED: {e}")
                continue
            elapsed = time.time() - t0
            print(f"  Trained in {elapsed:.0f}s. Backtesting ...")

        metrics = _backtest_config(exp_dir, m_type, cfg)
        if metrics is None:
            print("  BACKTEST FAILED: no checkpoint")
            continue

        row = {"model": m_type, **cfg, **metrics}
        results.append(row)
        
        # Write incrementally
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header_keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(results)

        print(f"  Result: {metrics['total_r']}R | {metrics['trades']} trades | {metrics['win_rate']}% WR | Sharpe {metrics['sharpe']}")

    # ── Final Report ──────────────────────────────────────────────────
    print(f"\n=== Sweep complete. Results saved to {out_csv} ===")
    print(f"\n{'Model':<15} {'Config':<25} {'Total R':>8} {'Trades':>7} {'Win%':>6} {'Sharpe':>7} {'PF':>5} {'Return%':>10}")
    print("-" * 95)
    for r in results:
        if "error" in r:
            continue
        if r["model"] == "transformer":
            cfg_str = f"L{r['num_layers']}_H{r['nhead']}_D{r['d_model']}"
        else:
            cfg_str = f"L{r['num_layers']}_U{r['hidden_size']}"
            
        tr = float(r["total_r"])
        ret_l = float(r["realistic_return_low"])
        ret_h = float(r["realistic_return_high"])
        print(f"{r['model']:<15} {cfg_str:<25} {tr:>8.1f} {r['trades']:>7} {r['win_rate']:>5.1f}% {r['sharpe']:>7.3f} {r['profit_factor']:>5.2f} {ret_l:>4.1f}-{ret_h:.1f}%")


if __name__ == "__main__":
    run_sweep()
