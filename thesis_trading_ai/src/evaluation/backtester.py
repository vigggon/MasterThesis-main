"""
Backtest trading simulation: 2024-01-01 to 2024-08-09 open sessions (unseen), fixed trained model.
Metrics: equity curve, Sharpe, max drawdown, profit factor, etc.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path

from utils import (
    PROCESSED_DIR, EXPERIMENTS_ROOT, RESULTS_ROOT, TRADE_THRESHOLD, NUM_CLASSES, 
    BEST_TRANSFORMER_DIR, BEST_LSTM_DIR,
    LSTM_HIDDEN, LSTM_LAYERS, TX_DMODEL, TX_NHEAD, TX_LAYERS,
    RISK_PCT_PER_TRADE, MAX_DD_CAP, TP_ATR_MULT, SL_ATR_MULT, 
    DAILY_STOP_R, DAILY_TP_R, MIN_ATR, ATR_SCALE_THRESHOLD, MAX_HOLD_CANDLES
)


from models import get_lstm, get_transformer

# Caches for performance optimization
_TIME_TO_IDX_CACHE = {}
_CACHE_ID = None


def _pred_3class_np(probs: np.ndarray, th: float) -> np.ndarray:
    """3-class: 0=hold, 1=long, 2=short."""
    p_long = probs[:, 1]
    p_short = probs[:, 2]
    trade = (p_long >= th) | (p_short >= th)
    pred = np.where(~trade, 0, np.where(p_long >= p_short, 1, 2))
    return pred.astype(np.int64)


def _returns_from_probs(
    probs: np.ndarray, 
    th: float, 
    spread_points: float = 0.0, 
    commission_points: float = 0.0,
    atr: np.ndarray | None = None,
    atr_regime: np.ndarray | None = None,
    times: np.ndarray | None = None,
    daily_max_loss: float | None = None,
    daily_take_profit: float | None = None,
    min_atr: float = 0.0,
    df_raw: pd.DataFrame | None = None
) -> np.ndarray:
    """
    Compute per-bar returns by actually simulating the trade walk-forward holding logic.
    Enforces MAX 1 concurrent position: a new trade cannot open until the previous one closes.
    """
    pred = _pred_3class_np(probs, th)
    returns = np.zeros(len(pred))
    
    combined_cost_points = spread_points + commission_points
    
    current_date = None
    daily_pnl = 0.0
    
    # Track when current position closes (bar index in df_raw)
    position_closes_at_raw_idx = -1
    
    if daily_max_loss is not None and times is not None:
         dates = pd.to_datetime(times).date
    else:
         dates = None

    # Time-to-index lookup happens via _returns_from_probs._time_to_idx_cache
    # which is initialized/updated inside the loop if df_raw is provided.

    for i in range(len(pred)):
        if dates is not None:
            d = dates[i]
            if d != current_date:
                current_date = d
                daily_pnl = 0.0
                # Reset position tracking at day boundary
                position_closes_at_raw_idx = -1
            
            if daily_pnl < (daily_max_loss or -999.0):
                returns[i] = 0.0
                continue
                
            if daily_take_profit is not None and daily_pnl > daily_take_profit:
                returns[i] = 0.0
                continue

        if atr is not None and atr[i] < min_atr:
            returns[i] = 0.0
            continue
                
        scale = 1.0
        if atr_regime is not None:
            regime = atr_regime[i]
            if regime > 2.0:
                scale = 0.25
            elif regime > 1.6:
                scale = 0.5
            elif regime > 1.3:
                scale = 0.75
                
        # Simulate walking forward
        direction = pred[i]
        if direction not in (1, 2) or df_raw is None:
            returns[i] = 0.0
            continue

        bar_time = times[i]
        
        # Optimized lookup
        target_time = pd.to_datetime(bar_time)
        if target_time.tzinfo is None:
            target_time = target_time.tz_localize("UTC")
        else:
            target_time = target_time.tz_convert("UTC")
            
        # We assume df_raw has unique datetimes. If not already indexed, we can use a dictionary.
        # For simplicity and speed within this function, we'll assume df_raw is large
        # so we'll pre-calculate the mapping if not passed.
        global _CACHE_ID, _TIME_TO_IDX_CACHE
        if _CACHE_ID != id(df_raw):
            raw_dt = pd.to_datetime(df_raw["datetime"])
            if raw_dt.dt.tz is None: raw_dt = raw_dt.dt.tz_localize("UTC")
            else: raw_dt = raw_dt.dt.tz_convert("UTC")
            _TIME_TO_IDX_CACHE = {t: idx for idx, t in enumerate(raw_dt)}
            _CACHE_ID = id(df_raw)

        idx_in_raw = _TIME_TO_IDX_CACHE.get(target_time)
        if idx_in_raw is None:
            returns[i] = 0.0
            continue
        
        # POSITION LIMIT: Skip if we already have an open position
        if idx_in_raw <= position_closes_at_raw_idx:
            returns[i] = 0.0
            continue
        
        entry = df_raw.loc[idx_in_raw, "close"]
        current_atr = atr[i] if atr[i] > 1e-4 else 1.0
        
        # Build future trajectory - STRICTLY same date only
        entry_date = df_raw.loc[idx_in_raw, "datetime"].date()
        future_candidates = df_raw.iloc[idx_in_raw + 1 : idx_in_raw + 1 + MAX_HOLD_CANDLES]
        future = future_candidates[future_candidates["datetime"].dt.date == entry_date]
        
        fh = future["high"].tolist()
        fl = future["low"].tolist()
        fc = future["close"].tolist()
        
        raw_pnl_pt = 0.0
        close_step = len(future) - 1  # default: closes at last future bar
        
        if direction == 1:
            sl = entry - current_atr * SL_ATR_MULT
            tp = entry + current_atr * TP_ATR_MULT
            resolved = False
            for step in range(len(future)):
                h, l, c = fh[step], fl[step], fc[step]
                if h >= tp and l <= sl:
                    raw_pnl_pt = sl - entry
                    close_step = step
                    resolved = True
                    break
                if h >= tp:
                    raw_pnl_pt = tp - entry
                    close_step = step
                    resolved = True
                    break
                if l <= sl:
                    raw_pnl_pt = sl - entry
                    close_step = step
                    resolved = True
                    break
            if not resolved:
                raw_pnl_pt = fc[-1] - entry if len(fc) > 0 else 0.0
                
        elif direction == 2:
            sl = entry + current_atr * SL_ATR_MULT
            tp = entry - current_atr * TP_ATR_MULT
            resolved = False
            for step in range(len(future)):
                h, l, c = fh[step], fl[step], fc[step]
                if l <= tp and h >= sl:
                    raw_pnl_pt = entry - sl
                    close_step = step
                    resolved = True
                    break
                if l <= tp:
                    raw_pnl_pt = entry - tp
                    close_step = step
                    resolved = True
                    break
                if h >= sl:
                    raw_pnl_pt = entry - sl
                    close_step = step
                    resolved = True
                    break
            if not resolved:
                raw_pnl_pt = entry - fc[-1] if len(fc) > 0 else 0.0

        # Record when this position closes so the next trade can't overlap
        if len(future) > 0:
            position_closes_at_raw_idx = future.index[min(close_step, len(future) - 1)]
        else:
            position_closes_at_raw_idx = idx_in_raw

        # Convert raw points to R-multiples
        raw_r = raw_pnl_pt / current_atr
        
        # Deduct transaction cost
        cost_r = combined_cost_points / current_atr
        net_r = raw_r - cost_r

        returns[i] = net_r * scale
        
        if dates is not None:
            daily_pnl += returns[i]
            
    return returns


def generate_trade_table(
    probs: np.ndarray, 
    th: float, 
    X_features: np.ndarray,
    feature_names: list,
    spread_points: float = 0.0, 
    commission_points: float = 0.0,
    atr: np.ndarray | None = None,
    atr_regime: np.ndarray | None = None,
    times: np.ndarray | None = None,
    daily_max_loss: float | None = None,
    daily_take_profit: float | None = None,
    min_atr: float = 0.0,
    df_raw: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Simulates trades and returns a detailed trade-by-trade DataFrame (Primary Data).
    Includes indicator values at time of entry as requested by advisor.
    """
    pred = _pred_3class_np(probs, th)
    trades = []
    
    combined_cost_points = spread_points + commission_points
    current_date = None
    daily_pnl = 0.0
    position_closes_at_raw_idx = -1
    
    if daily_max_loss is not None and times is not None:
         dates = pd.to_datetime(times).date
    else:
         dates = None

    global _CACHE_ID, _TIME_TO_IDX_CACHE
    if df_raw is not None:
        if _CACHE_ID != id(df_raw):
            raw_dt = pd.to_datetime(df_raw["datetime"])
            if raw_dt.dt.tz is None: raw_dt = raw_dt.dt.tz_localize("UTC")
            else: raw_dt = raw_dt.dt.tz_convert("UTC")
            _TIME_TO_IDX_CACHE = {t: idx for idx, t in enumerate(raw_dt)}
            _CACHE_ID = id(df_raw)

    for i in range(len(pred)):
        if dates is not None:
            d = dates[i]
            if d != current_date:
                current_date = d
                daily_pnl = 0.0
                position_closes_at_raw_idx = -1
            if daily_pnl < (daily_max_loss or -999.0): continue
            if daily_take_profit is not None and daily_pnl > daily_take_profit: continue

        if atr is not None and atr[i] < min_atr: continue
                
        scale = 1.0
        if atr_regime is not None:
            regime = atr_regime[i]
            if regime > 2.0: scale = 0.25
            elif regime > 1.6: scale = 0.5
            elif regime > 1.3: scale = 0.75
                
        direction = pred[i]
        if direction not in (1, 2) or df_raw is None: continue

        bar_time = times[i]
        target_time = pd.to_datetime(bar_time)
        if target_time.tzinfo is None: target_time = target_time.tz_localize("UTC")
        else: target_time = target_time.tz_convert("UTC")
            
        idx_in_raw = _TIME_TO_IDX_CACHE.get(target_time)
        if idx_in_raw is None: continue
        if idx_in_raw <= position_closes_at_raw_idx: continue
        
        entry_price = df_raw.loc[idx_in_raw, "close"]
        current_atr = atr[i] if atr[i] > 1e-4 else 1.0
        
        entry_date = df_raw.loc[idx_in_raw, "datetime"].date()
        future_candidates = df_raw.iloc[idx_in_raw + 1 : idx_in_raw + 1 + MAX_HOLD_CANDLES]
        future = future_candidates[future_candidates["datetime"].dt.date == entry_date]
        
        fh = future["high"].tolist()
        fl = future["low"].tolist()
        fc = future["close"].tolist()
        
        raw_pnl_pt = 0.0
        close_step = len(future) - 1
        
        if direction == 1:
            sl, tp = entry_price - current_atr * SL_ATR_MULT, entry_price + current_atr * TP_ATR_MULT
            resolved = False
            for step in range(len(future)):
                h, l = fh[step], fl[step]
                if h >= tp and l <= sl: raw_pnl_pt, close_step, resolved = (sl - entry_price), step, True; break
                if h >= tp: raw_pnl_pt, close_step, resolved = (tp - entry_price), step, True; break
                if l <= sl: raw_pnl_pt, close_step, resolved = (sl - entry_price), step, True; break
            if not resolved: raw_pnl_pt = fc[-1] - entry_price if len(fc) > 0 else 0.0
        else:
            sl, tp = entry_price + current_atr * SL_ATR_MULT, entry_price - current_atr * TP_ATR_MULT
            resolved = False
            for step in range(len(future)):
                h, l = fh[step], fl[step]
                if l <= tp and h >= sl: raw_pnl_pt, close_step, resolved = (entry_price - sl), step, True; break
                if l <= tp: raw_pnl_pt, close_step, resolved = (entry_price - tp), step, True; break
                if h >= sl: raw_pnl_pt, close_step, resolved = (entry_price - sl), step, True; break
            if not resolved: raw_pnl_pt = entry_price - fc[-1] if len(fc) > 0 else 0.0

        if len(future) > 0:
            exit_bar_idx = future.index[min(close_step, len(future) - 1)]
            close_time = df_raw.loc[exit_bar_idx, "datetime"]
            position_closes_at_raw_idx = exit_bar_idx
            exit_price = fc[close_step]
        else:
            close_time = bar_time
            position_closes_at_raw_idx = idx_in_raw
            exit_price = entry_price

        raw_r = raw_pnl_pt / current_atr
        net_r = (raw_r - (combined_cost_points / current_atr)) * scale
        daily_pnl += net_r

        # Create trade record
        trade_data = {
            "open_time": bar_time,
            "close_time": close_time,
            "direction": "long" if direction == 1 else "short",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_points": round(raw_pnl_pt, 2),
            "pnl_r": round(net_r, 2),
            "atr": round(current_atr, 2)
        }
        
        # Add all indicator values at open time
        for idx, col_name in enumerate(feature_names):
            # X_features is (N, window, n_feats). We want value at decision step (last step of window).
            trade_data[col_name] = float(X_features[i, -1, idx])
            
        trades.append(trade_data)
            
    return pd.DataFrame(trades)

def _get_threshold(model_name: str) -> float:
    """Use threshold from trade_threshold.json if present (tuned in train), else utils default."""
    exp_dir = BEST_TRANSFORMER_DIR if model_name == "transformer" else BEST_LSTM_DIR
    path = exp_dir / "trade_threshold.json"
    if path.exists():
        with open(path) as f:
            return float(json.load(f)["threshold"])
    return TRADE_THRESHOLD

def load_backtest_data_and_predictions(model_name: str, spread_points: float = 0.0, commission_points: float = 0.0, dynamic_risk: bool = False, daily_max_loss: float | None = None, daily_take_profit: float | None = None, min_atr: float = 0.0):
    data = np.load(PROCESSED_DIR / "backtest.npz", allow_pickle=True)
    X = torch.from_numpy(data["X"]).float()
    y = data["y"]
    times = data["times"]
    feature_cols = list(data["feature_cols"])

    # Extract ATR for dynamic cost calc
    try:
        atr_idx = feature_cols.index("atr_14")
        # X is (N, window, features). We need ATR at the decision step (last step of window? or first?)
        # dataset_builder builds sequences. y[i] is label for X[i].
        # X[i] contains [t-window+1 ... t]. decision is made at t.
        # So we want the LAST time step's ATR.
        atr_values = X[:, -1, atr_idx].numpy()
    except ValueError:
        print("Warning: 'atr_14' not found in features. Dynamic cost calc will be skipped/inaccurate.")
        atr_values = None

    try:
        regime_idx = feature_cols.index("atr_regime")
        atr_regime = X[:, -1, regime_idx].numpy()
    except ValueError:
        atr_regime = None

    exp_dir = BEST_TRANSFORMER_DIR if model_name == "transformer" else BEST_LSTM_DIR
    ckpt = torch.load(exp_dir / "best.pt", map_location="cpu", weights_only=False)
    
    # Align features: checkpoint expectation vs loaded data
    ckpt_cols = ckpt.get("feature_cols", None)
    if ckpt_cols is not None:
        col_to_idx = {c: i for i, c in enumerate(feature_cols)}
        indices = [col_to_idx[c] for c in ckpt_cols if c in col_to_idx]
        X_model = X[:, :, indices]
        n_features = len(ckpt_cols)
    else:
        X_model = X
        n_features = X.shape[2]

    if model_name == "lstm":
        model = get_lstm(n_features, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=0.0, num_classes=NUM_CLASSES)
    else:
        model = get_transformer(n_features, d_model=TX_DMODEL, nhead=TX_NHEAD, num_layers=TX_LAYERS, dropout=0.0, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    th = _get_threshold(model_name)
    with torch.no_grad():
        logits = model(X_model)
        probs = torch.softmax(logits, dim=1).numpy()
    pred = _pred_3class_np(probs, th)
    # Only pass atr_regime if dynamic_risk is True
    regime_arg = atr_regime if dynamic_risk else None
    
    # Load raw dataframe for walk-forward simulation
    raw_path = PROCESSED_DIR.parent / "features" / "open_features.csv"
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    else:
        print(f"Warning: {raw_path} not found. Some backtest features may fail.")
        df_raw = None

    returns = _returns_from_probs(probs, th, spread_points=spread_points, commission_points=commission_points, atr=atr_values, atr_regime=regime_arg, times=times, daily_max_loss=daily_max_loss, daily_take_profit=daily_take_profit, min_atr=min_atr, df_raw=df_raw)
    return returns, times, pred, y, probs, atr_values, X, feature_cols


def equity_curve(returns: np.ndarray) -> np.ndarray:
    return np.cumsum(returns)


def equity_curve_pct(
    returns: np.ndarray,
    risk_pct: float = 0.01,
    max_dd_cap: float | None = None,
) -> np.ndarray:
    """
    Equity curve with percentage risk per trade. If max_dd_cap (e.g. 0.25), stop trading
    when equity drops more than that from peak (subsequent returns ignored).
    """
    equity = np.ones(len(returns) + 1)
    stopped = False
    for i, r in enumerate(returns):
        if stopped:
            equity[i + 1] = equity[i]
            continue
        if r != 0:
            equity[i + 1] = equity[i] * (1.0 + risk_pct * r)
        else:
            equity[i + 1] = equity[i]
        if max_dd_cap is not None:
            peak = equity[: i + 2].max()
            if (peak - equity[i + 1]) / peak >= max_dd_cap:
                stopped = True
    return equity[1:]


def max_drawdown_pct(equity_pct: np.ndarray) -> float:
    """Max drawdown as fraction (e.g. -0.25 = -25%)."""
    peak = np.maximum.accumulate(equity_pct)
    dd = (equity_pct - peak) / np.where(peak > 0, peak, 1)
    return float(np.min(dd))


def sharpe(returns: np.ndarray, risk_free: float = 0.0, ann_factor: float = np.sqrt(252 * 78)) -> float:
    # 78 5-min bars (approx 12 per hour) -> Now 10-min bars (6 per hour * 2 hrs = 12 bars) + pre/post?
    # Session is 2 hours (09:30-11:30). 12 bars per session.
    # Ann factor for 10-min bars: sqrt(252 * 12)
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free) / returns.std() * np.sqrt(ann_factor)


def max_drawdown(equity: np.ndarray) -> float:
    """Max drawdown from unit-based equity (can exceed -100% if equity goes negative)."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(np.min(dd))


def profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return gains / losses


# Default risk per trade for "realistic" equity (1% of equity per trade)
# RISK_PCT_PER_TRADE = 0.01  To utils
# Optional: stop trading when drawdown exceeds this (e.g. 0.25 = 25%)
# MAX_DD_CAP = 0.25  To utils


def run_backtester(
    model_name: str = "lstm",
    risk_pct: float = RISK_PCT_PER_TRADE,
    risk_per_trade_dollars: float | None = None,
    max_dd_cap: float | None = MAX_DD_CAP,
    spread_points: float = 0.0,
    commission_points: float = 0.0,
    dynamic_risk: bool = False,
    daily_max_loss: float | None = None,
    daily_take_profit: float | None = None,
    min_atr: float = MIN_ATR,
) -> dict:
    returns, times, pred, y, probs, atr_values, _, _ = load_backtest_data_and_predictions(
        model_name, spread_points=spread_points, commission_points=commission_points, 
        dynamic_risk=dynamic_risk, daily_max_loss=daily_max_loss, 
        daily_take_profit=daily_take_profit, min_atr=min_atr
    )
    equity_units = equity_curve(returns)
    equity_pct = equity_curve_pct(returns, risk_pct=risk_pct)
    equity_pct_capped = equity_curve_pct(returns, risk_pct=risk_pct, max_dd_cap=max_dd_cap) if max_dd_cap is not None else equity_pct
    max_dd_units = max_drawdown(equity_units)
    max_dd_pct = max_drawdown_pct(equity_pct)
    max_dd_pct_capped = max_drawdown_pct(equity_pct_capped) if max_dd_cap is not None else max_dd_pct
    n_trades = int(np.count_nonzero(returns))
    
    # Calculate realistic return based on fixed position sizing
    # Assume 1 unit ≈ $10-15 profit on typical account size
    total_return_units = float(returns.sum())
    realistic_return_low = (total_return_units * 10) / 10000  # $10/unit on $10k account
    realistic_return_high = (total_return_units * 15) / 10000  # $15/unit on $10k account
    
    metrics = {
        "total_return": total_return_units,
        "sharpe_ratio": float(sharpe(returns)),
        "max_drawdown_units": float(max_dd_units),
        "max_drawdown_pct": float(max_dd_pct),
        "profit_factor": float(profit_factor(returns)),
        "n_trades": n_trades,
        "win_rate": float((returns[returns != 0] > 0).mean()) if (returns != 0).any() else 0.0,
        "risk_pct_per_trade": risk_pct,
        "final_equity_pct": float(equity_pct[-1]) if len(equity_pct) else 1.0,
        "realistic_return_pct_low": realistic_return_low * 100,
        "realistic_return_pct_high": realistic_return_high * 100,
    }
    
    out_dir = RESULTS_ROOT / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-trade details using specializes table generator
    raw_feats_path = PROCESSED_DIR.parent / "features" / "open_features.csv"
    if raw_feats_path.exists():
        df_raw = pd.read_csv(raw_feats_path)
    else:
        df_raw = None

    df_trades = generate_trade_table(
        probs, _get_threshold(model_name), 
        spread_points=spread_points, commission_points=commission_points,
        atr=atr_values, times=times, 
        daily_max_loss=daily_max_loss, daily_take_profit=daily_take_profit,
        min_atr=min_atr, df_raw=df_raw
    )
    df_trades.to_csv(out_dir / f"{model_name}_trades.csv", index=False)

    df_equity = pd.DataFrame({
        "datetime": times,
        "return": returns,
        "equity_units": equity_units,
        "equity_pct": equity_pct,
    })
    if max_dd_cap is not None:
        df_equity["equity_pct_capped"] = equity_pct_capped
    df_equity.to_csv(out_dir / f"{model_name}_equity.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / f"{model_name}_metrics.csv", index=False)

    # Print key metrics (actual backtest range from data)
    t_min, t_max = pd.Timestamp(times[0]), pd.Timestamp(times[-1])
    cost_info = ""
    if spread_points > 0 or commission_points > 0:
        cost_info = f" (spread_pts={spread_points:.2f}, comm_pts={commission_points:.2f})"
    print(f"[{model_name}] Backtest ({t_min.strftime('%Y-%m-%d')} to {t_max.strftime('%Y-%m-%d')}){cost_info}")
    if daily_max_loss is not None:
        print(f"  Daily Stop Loss: {daily_max_loss} R")
    if daily_take_profit is not None:
        print(f"  Daily Take Profit: {daily_take_profit} R")
    if min_atr > 0.0:
        print(f"  Min ATR Filter: {min_atr}")
    if dynamic_risk:
        print(f"  Dynamic Risk Scaling: ENABLED (>1.3=0.75x, >1.6=0.5x, >2.0=0.25x)")
    print(f"  Total profit (return units): {metrics['total_return']:.2f}")
    if risk_per_trade_dollars is not None:
        profit_cash = metrics["total_return"] * risk_per_trade_dollars
        print(f"  Total profit (${risk_per_trade_dollars:.0f}/trade): ${profit_cash:,.0f}")
    print(f"  Trades: {metrics['n_trades']}, Win rate: {metrics['win_rate']:.1%}")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}, PF: {metrics['profit_factor']:.2f}")
    print(f"  Max DD (units): {metrics['max_drawdown_units']:.2f}")
    print(f"  Max DD ({risk_pct:.1%} risk/trade): {metrics['max_drawdown_pct']:.1%}")
    print(f"")
    print(f"  REALISTIC RETURN (fixed position sizing):")
    print(f"    Estimated: {metrics['realistic_return_pct_low']:.1f}-{metrics['realistic_return_pct_high']:.1f}% over {(t_max - t_min).days // 30} months")
    print(f"    (Assumes 1 unit = $10-15 on $10k account, no compounding)")
    print(f"")
    # print(f"  THEORETICAL ONLY (1% risk compounding - unrealistic for intraday):")
    # print(f"    Final equity: {metrics['final_equity_pct']:.2%}")
    # if max_dd_cap is not None and "final_equity_pct_capped" in metrics:
    #     print(f"    Final equity (capped at {max_dd_cap:.0%} DD): {metrics['final_equity_pct_capped']:.2%}")
    # print(f"    Note: Compounding assumes infinite scaling (impractical for intraday strategies)")
    return metrics


def run_sweep_threshold(model_name: str, spread_points: float = 0.0, commission_points: float = 0.0):
    """Try thresholds 0.35–0.6 and print total return, n_trades, win_rate for each (helps find profitable threshold)."""
    _, times, _, y, probs, atr_values, _, _ = load_backtest_data_and_predictions(model_name, spread_points=spread_points, commission_points=commission_points)
    thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    cost_info = f" (spread_pts={spread_points:.2f}, comm_pts={commission_points:.2f})" if (spread_points > 0 or commission_points > 0) else ""
    print(f"[{model_name}] Threshold sweep (TP={TP_ATR_MULT}, SL={SL_ATR_MULT}){cost_info}")
    print("  th     total_return  n_trades  win_rate")
    
    # Load raw dataframe for walk-forward simulation
    from utils import PROCESSED_DIR
    raw_path = PROCESSED_DIR.parent / "features" / "open_features.csv"
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
    else:
        df_raw = None

    for th in thresholds:
        returns = _returns_from_probs(probs, th, spread_points=spread_points, commission_points=commission_points, atr=atr_values, df_raw=df_raw, times=times)
        n_trades = int((returns != 0).sum())
        wr = (returns[returns != 0] > 0).mean() if n_trades else 0.0
        print(f"  {th:.2f}   {returns.sum():+12.2f}  {n_trades:>7}  {wr:.1%}")
    print("  (Update trade_threshold.json or utils.TRADE_THRESHOLD to use a different threshold.)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest LSTM/Transformer on backtest split (see utils.BACKTEST_DAYS / BACKTEST_START–END).")
    parser.add_argument("model", nargs="?", default="lstm", help="lstm or transformer")
    parser.add_argument("--risk-pct", type=float, default=RISK_PCT_PER_TRADE, help="Risk per trade as fraction of equity (default: 0.01)")
    parser.add_argument("--dollars-per-trade", type=float, default=None, help="Interpret return units as $ (e.g. 100)")
    parser.add_argument("--spread", type=float, default=2.0, help="Transaction cost: spread+slippage in POINTS per trade (default: 2.0 realistic)")
    parser.add_argument("--commission", type=float, default=0.0, help="Transaction cost: commission in POINTS per trade (default: 0.0)")
    parser.add_argument("--dynamic-risk", action="store_true", help="Enable volatility-based risk scaling (reduce size in high vol)")
    parser.add_argument("--no-dd-cap", action="store_true", help="Disable drawdown cap (no stop-trading simulation)")
    parser.add_argument("--daily-stop", type=float, default=DAILY_STOP_R, help=f"Daily Stop Loss in R (default: {DAILY_STOP_R}). If None, disabled.")
    parser.add_argument("--daily-tp", type=float, default=DAILY_TP_R, help=f"Daily Take Profit in R (default: {DAILY_TP_R}). If None, disabled.")
    parser.add_argument("--min-atr", type=float, default=MIN_ATR, help=f"Minimum ATR absolute value to trade (default: {MIN_ATR}).")
    parser.add_argument("--both", action="store_true", help="Run backtest for both lstm and transformer")
    parser.add_argument("--sweep-threshold", action="store_true", help="Sweep threshold 0.35–0.6 and print backtest PnL per threshold")
    args = parser.parse_args()
    if args.sweep_threshold:
        run_sweep_threshold(args.model, spread_points=args.spread, commission_points=args.commission)
    elif args.both:
        for name in ("lstm", "transformer"):
            run_backtester(
                model_name=name,
                risk_pct=args.risk_pct,
                risk_per_trade_dollars=args.dollars_per_trade,
                max_dd_cap=None if args.no_dd_cap else MAX_DD_CAP,
                spread_points=args.spread,
                commission_points=args.commission,
                dynamic_risk=args.dynamic_risk,
                daily_max_loss=args.daily_stop,
                min_atr=args.min_atr,
            )
            print()
    else:
        run_backtester(
            model_name=args.model,
            risk_pct=args.risk_pct,
            risk_per_trade_dollars=args.dollars_per_trade,
            max_dd_cap=None if args.no_dd_cap else MAX_DD_CAP,
            spread_points=args.spread,
            commission_points=args.commission,
            dynamic_risk=args.dynamic_risk,
            daily_max_loss=args.daily_stop,
            daily_take_profit=args.daily_tp,
            min_atr=args.min_atr,
        )
