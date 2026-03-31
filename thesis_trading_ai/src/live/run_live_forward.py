"""
Live forward testing: connect to MT5, poll M5 bars during NY open (09:30–11:30),
build features, run LSTM and Transformer, log signals and simulated session P&L.
Run after full pipeline (train.npz, best.pt checkpoints exist).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from utils import (
    OPEN_START,
    OPEN_END,
    NY_TZ,
    WINDOW,
    RESULTS_ROOT,
    ATR_PERIOD,
    SL_ATR_MULT,
    TP_ATR_MULT,
    MIN_ATR,
    MAX_HOLD_CANDLES,
    ACCOUNT_BALANCE,
    RISK_PCT_PER_TRADE,
    POINT_VALUE_PER_LOT,
)
from pipeline.feature_engineering import build_features
from evaluation.forward_test import load_both_models, predict_both_models_on_new_data

# Symbol candidates (broker-dependent)
SYMBOLS = ("USTEC", "NAS100", "NDX100", "US100")


def _bars_from_mt5(symbol: str, count: int = 1000) -> pd.DataFrame | None:
    """Fetch last `count` M10 bars from MT5. Returns DataFrame or None."""
    if mt5 is None:
        return None
    tf = mt5.TIMEFRAME_M10
    rates = mt5.copy_rates_from_pos(symbol, tf, 1, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Auto-correct Broker Time Offset
    # Many brokers (e.g., IC Markets) send Server Time (UTC+2/3) as unix timestamp.
    # If the data appears to be in the future relative to UTC, we subtract the offset.
    last_time = df["time"].max()
    now_utc = pd.Timestamp.now(tz="UTC").replace(tzinfo=None) # naive UTC for comparison with naive broker time
    diff = last_time - now_utc
    if diff.total_seconds() > 1800:  # If > 30 mins in future
        offset_hours = round(diff.total_seconds() / 3600)
        df["time"] -= pd.Timedelta(hours=offset_hours)
        
        # Only print once to avoid spam
        if not hasattr(_bars_from_mt5, "_offset_warned"):
            print(f"[DEBUG] Detected MT5 time offset: +{offset_hours}h. Correcting data...")
            _bars_from_mt5._offset_warned = True

    df = df.rename(columns={"time": "datetime", "tick_volume": "volume"})
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(NY_TZ)
    return df


def _resolve_symbol() -> str:
    """Try SYMBOLS until one returns data."""
    if mt5 is None:
        raise RuntimeError("MetaTrader5 is required. pip install MetaTrader5")
    
    # helper assumes mt5 is already initialized by caller
    
    for sym in SYMBOLS:
        mt5.symbol_select(sym, True)
        df = _bars_from_mt5(sym, count=200)
        if df is not None and len(df) >= WINDOW:
            return sym
    
    raise RuntimeError("No data for NAS100/NDX100/US100. Check MT5 symbol and connection.")


def _in_session(dt_ny) -> bool:
    """True if dt_ny (NY time) is within OPEN_START–OPEN_END."""
    t = dt_ny.time() if hasattr(dt_ny, "time") else dt_ny
    start = datetime.strptime(OPEN_START, "%H:%M").time()
    end = datetime.strptime(OPEN_END, "%H:%M").time()
    return start <= t <= end


def _filter_open_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:30–11:30 NY time."""
    t = df["datetime"].dt.tz_convert(NY_TZ).dt.time
    start = datetime.strptime(OPEN_START, "%H:%M").time()
    end = datetime.strptime(OPEN_END, "%H:%M").time()
    mask = (t >= start) & (t <= end)
    return df.loc[mask].copy().reset_index(drop=True)


def _suggested_lots(atr_val: float, point_value: float = POINT_VALUE_PER_LOT, symbol: str = "") -> float:
    """Risk RISK_PCT_PER_TRADE of ACCOUNT_BALANCE; compliant with symbol volume step."""
    if atr_val <= 0:
        return 0.0
        
    risk_amount = ACCOUNT_BALANCE * RISK_PCT_PER_TRADE
    sl_points = atr_val * SL_ATR_MULT
    if sl_points <= 0:
        return 0.0
        
    # Lots such that sl_points * point_value * lots = risk_amount
    raw_lots = risk_amount / (sl_points * point_value) if point_value else 0.0
    
    # Normalize via MT5 info if available
    if mt5 is not None and symbol:
        info = mt5.symbol_info(symbol)
        if info:
            step = info.volume_step
            min_lot = info.volume_min
            max_lot = info.volume_max
            
            # Snap to step
            if step > 0:
                raw_lots = round(raw_lots / step) * step
                
            # Clamp
            raw_lots = max(min_lot, min(raw_lots, max_lot))
            
            # Final rounding to eliminate float errors (e.g. 0.100000001 -> 0.1)
            # using the number of decimals in 'step'
            import decimal
            decimals = abs(decimal.Decimal(str(step)).as_tuple().exponent) if step > 0 else 2
            return round(raw_lots, decimals)

    return round(max(0.01, min(raw_lots, 10.0)), 2)


def _resolve_pnl(
    entry: float,
    atr_val: float,
    future_highs: list,
    future_lows: list,
    future_closes: list,
    direction: int,
) -> tuple:
    """Simulated P&L with pessimistic intra-bar resolution.
    Returns (pnl_points, exit_price, close_step)."""
    if atr_val <= 0 or not future_closes:
        return 0.0, entry, 0
    if direction == 1:
        sl, tp = entry - atr_val * SL_ATR_MULT, entry + atr_val * TP_ATR_MULT
        for step in range(min(MAX_HOLD_CANDLES, len(future_closes))):
            low, high, close = future_lows[step], future_highs[step], future_closes[step]
            if high >= tp and low <= sl:
                # PESSIMISTIC: assume SL hit first
                return (sl - entry), sl, step
            if high >= tp:
                return (tp - entry), tp, step
            if low <= sl:
                return (sl - entry), sl, step
        exit_p = future_closes[-1]
        return float(exit_p - entry), exit_p, len(future_closes) - 1
    else:
        sl, tp = entry + atr_val * SL_ATR_MULT, entry - atr_val * TP_ATR_MULT
        for step in range(min(MAX_HOLD_CANDLES, len(future_closes))):
            low, high, close = future_lows[step], future_highs[step], future_closes[step]
            if low <= tp and high >= sl:
                # PESSIMISTIC: assume SL hit first
                return (entry - sl), sl, step
            if low <= tp:
                return (entry - tp), tp, step
            if high >= sl:
                return (entry - sl), sl, step
        exit_p = future_closes[-1]
        return float(entry - exit_p), exit_p, len(future_closes) - 1


def _execute_trade(symbol: str, direction: int, lots: float, sl_points: float, tp_points: float):
    """Send market order to MT5 and return ticket if successful."""
    if lots <= 0.0:
        return None

    action = mt5.TRADE_ACTION_DEAL
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if direction == 1 else mt5.symbol_info_tick(symbol).bid
    
    # SL/TP calculation
    if direction == 1:
        sl = price - sl_points
        tp = price + tp_points
    else:
        sl = price + sl_points
        tp = price - tp_points

    request = {
        "action": action,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "Thesis AI",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode} ({result.comment})")
        return None
    else:
        print(f"Order executed: {result.volume} lots @ {result.price}, Ticket={result.order}")
        return result.order # Return the Ticket (order/position ID)


def _close_position(symbol: str, ticket: int, lots: float):
    """Close an existing MT5 position by ticket."""
    if mt5 is None:
        return None
    
    # Check if position still exists
    positions = mt5.positions_get(ticket=ticket)
    if positions is None or len(positions) == 0:
        # print(f"[DEBUG] Position {ticket} not found (already closed by SL/TP).")
        return None

    pos = positions[0]
    # To close a BUY, we must SELL. To close a SELL, we must BUY.
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": "Thesis AI Time Exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Close failed for ticket {ticket}: {result.retcode} ({result.comment})")
    else:
        print(f"Closed ticket {ticket} due to 2-hour limit.")
    return result


def _append_trade_row(path: Path, row_data: dict, write_header: bool):
    """Append a single trade row to CSV, writing header if needed."""
    import csv
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row_data)


def run_live_forward(
    model_name: str,
    symbol: str | None = None,
    account_balance: float = ACCOUNT_BALANCE,
    risk_pct: float = RISK_PCT_PER_TRADE,
    point_value: float = POINT_VALUE_PER_LOT,
    mt5_path: str | None = None,
    min_atr: float = 0.0,
) -> None:
    # 1. Initialize MT5 (Specific Terminal)
    if mt5 is None:
        raise RuntimeError("MetaTrader5 required.")
    
    init_args = {}
    if mt5_path:
        init_args["path"] = mt5_path
        
    if not mt5.initialize(**init_args):
        raise RuntimeError(f"MT5 initialize failed for path: {mt5_path or 'default'}. Error: {mt5.last_error()}")

    try:
        symbol = symbol or _resolve_symbol() # relies on initialized mt5
        
        # 2. Setup Logging
        model_tag = model_name.lower()
        (RESULTS_ROOT / "forward_test").mkdir(parents=True, exist_ok=True)
        signals_path = RESULTS_ROOT / "forward_test" / f"live_signals_{model_tag}.csv"
        session_pnl_path = RESULTS_ROOT / "forward_test" / f"session_pnl_{model_tag}.csv"

        # 3. Load Models
        # We load both common logic but will only use one
        lstm, transformer, feature_cols, device = load_both_models()
        
        # Helper to get prediction
        def get_signal(X):
            p_lstm, p_tx = predict_both_models_on_new_data(lstm, transformer, X, device)
            if model_name == "lstm":
                return p_lstm
            elif model_name == "transformer":
                return p_tx
            else:
                 # fallback combined (original logic)
                 return p_lstm if p_lstm in (1, 2) else (p_tx if p_tx in (1, 2) else 0)

        print(f"Live forward test started for [{model_name.upper()}].")
        print(f"Symbol={symbol}, MT5={'Default' if not mt5_path else mt5_path}")
        print(f"Min ATR Filter: {min_atr}")
        print("Ctrl+C to stop.\n")

        need_header = not signals_path.exists()
        need_pnl_header = not session_pnl_path.exists() or session_pnl_path.stat().st_size == 0
        last_bar_time = None
        pending: list = []  # Each entry: (entry_time, entry_close, atr_val, direction, fh, fl, fc, ticket, lots, features_dict)
        pnl_columns = None  # Will be set on first trade

        while True:
            now_ny = pd.Timestamp.now(tz=NY_TZ)
            if not _in_session(now_ny):
                if pending:
                    # Resolve pending at session end
                    for p in pending:
                        entry_time, entry_close, atr_val, direction, fh, fl, fc, ticket, lots, feat_dict = p
                        if ticket:
                            _close_position(symbol, ticket, lots)
                        pnl_pts, exit_price, close_step = _resolve_pnl(entry_close, atr_val, fh, fl, fc, direction)
                        pnl_r = pnl_pts / atr_val if atr_val > 0 else 0.0
                        row_data = {
                            "open_time": entry_time,
                            "close_time": now_ny.isoformat(),
                            "entry_price": entry_close,
                            "exit_price": exit_price,
                            "direction": "long" if direction == 1 else "short",
                            "pnl_points": pnl_pts,
                            "pnl_r": pnl_r,
                            "atr": atr_val,
                            "lots": lots,
                            "ticket": ticket or "",
                        }
                        row_data.update(feat_dict)
                        _append_trade_row(session_pnl_path, row_data, need_pnl_header)
                        need_pnl_header = False
                    pending.clear()
                time.sleep(60)
                continue

            # Fetch 10m bars
            # NOTE: _bars_from_mt5 defaults to M5. We need to update that function too or override here?
            # Ideally we update _bars_from_mt5 to use M10.
            # For now, let's assume one function call update fits all.
            # See separate tool call for _bars_from_mt5 update.
            
            df = _bars_from_mt5(symbol, count=1000)
            if df is None or len(df) < WINDOW:
                time.sleep(30)
                continue

            session_df = _filter_open_session(df)
            if len(session_df) < WINDOW:
                time.sleep(30)
                continue

            feats = build_features(session_df)
            if feats is None or len(feats) < WINDOW:
                time.sleep(30)
                continue

            # Last closed bar
            row = feats.iloc[-1]
            bar_time = row["datetime"]
            
            if last_bar_time is not None and bar_time == last_bar_time:
                # print(f"[DEBUG] No new bar. Last: {last_bar_time}")
                time.sleep(10)
                continue
            last_bar_time = bar_time

            # Prepare Input
            X = feats[feature_cols].iloc[-WINDOW:].values.astype(np.float32)
            X = X.reshape(1, WINDOW, -1)
            
            # Predict
            pred = get_signal(X)
            atr_val = float(row.get("atr_14", 0) or 0)
            
            # ATR Filter
            if min_atr > 0 and atr_val < min_atr:
                pred = 0 # No Trade
            
            lots = _suggested_lots(atr_val, point_value, symbol)
            
            # Log Signal
            bar_close = float(row["close"])
            with open(signals_path, "a", newline="") as f:
                w = csv.writer(f)
                if need_header:
                    w.writerow(["datetime", "close", "pred", "lots", "atr"])
                    need_header = False
                w.writerow([bar_time, bar_close, pred, lots, atr_val])

            # Update Pending Logic (Simulation)
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            
            new_pending = []
            for p in pending:
                (entry_time, entry_close, entry_atr, direction, fh, fl, fc, ticket, p_lots, feat_dict) = p
                fh.append(bar_high)
                fl.append(bar_low)
                fc.append(bar_close)
                
                # Check if position is still open
                is_open = True
                if ticket:
                    # MT5 Live checkout
                    positions = mt5.positions_get(ticket=ticket)
                    if positions is None or len(positions) == 0:
                        is_open = False
                else:
                    # Simulated checkout
                    sl_pts = entry_atr * SL_ATR_MULT
                    tp_pts = entry_atr * TP_ATR_MULT
                    if direction == 1:
                        if bar_low <= entry_close - sl_pts or bar_high >= entry_close + tp_pts:
                            is_open = False
                    else:
                        if bar_high >= entry_close + sl_pts or bar_low <= entry_close - tp_pts:
                            is_open = False

                if len(fc) >= MAX_HOLD_CANDLES or not is_open:
                    if ticket and is_open:
                        _close_position(symbol, ticket, p_lots)
                    pnl_pts, exit_price, close_step = _resolve_pnl(entry_close, entry_atr, fh, fl, fc, direction)
                    pnl_r = pnl_pts / entry_atr if entry_atr > 0 else 0.0
                    row_data = {
                        "open_time": entry_time,
                        "close_time": str(bar_time),
                        "entry_price": entry_close,
                        "exit_price": exit_price,
                        "direction": "long" if direction == 1 else "short",
                        "pnl_points": pnl_pts,
                        "pnl_r": pnl_r,
                        "atr": entry_atr,
                        "lots": p_lots,
                        "ticket": ticket or "",
                    }
                    row_data.update(feat_dict)
                    _append_trade_row(session_pnl_path, row_data, need_pnl_header)
                    need_pnl_header = False
                else:
                    new_pending.append(p)
            pending = new_pending

            # MAX 1 CONCURRENT POSITION: only enter if no open trades
            if pred in (1, 2) and len(pending) == 0:
                print(f"[{model_name.upper()}] SIGNAL {pred} @ {bar_time} (ATR={atr_val:.2f})")
                
                # Capture all features at trade entry
                feat_dict = {}
                for col in feature_cols:
                    if col in feats.columns:
                        feat_dict[col] = float(row[col])
                
                # Execute Trade
                sl_pts = atr_val * SL_ATR_MULT
                tp_pts = atr_val * TP_ATR_MULT
                ticket = _execute_trade(symbol, pred, lots, sl_pts, tp_pts)
                
                if ticket is not None:
                    pending.append((bar_time, bar_close, atr_val, pred, [], [], [], ticket, lots, feat_dict))
                else:
                    print(f"[{model_name.upper()}] Trade execution failed (no ticket). Signal abandoned.")
            elif pred in (1, 2) and len(pending) > 0:
                print(f"[{model_name.upper()}] SIGNAL {pred} @ {bar_time} SKIPPED (position open)")
            else:
                print(f"[{model_name.upper()}] No Signal @ {bar_time} (ATR={atr_val:.2f})")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live forward test: MT5 + LSTM/Transformer")
    parser.add_argument("model", choices=["lstm", "transformer"], help="Model to run")
    parser.add_argument("--symbol", type=str, default=None, help="MT5 symbol")
    parser.add_argument("--balance", type=float, default=ACCOUNT_BALANCE, help="Account balance")
    parser.add_argument("--risk", type=float, default=RISK_PCT_PER_TRADE, help="Risk fraction")
    parser.add_argument("--point-value", type=float, default=POINT_VALUE_PER_LOT, help="USD per point")
    parser.add_argument("--mt5-path", type=str, default=None, help="Path to specific terminal64.exe")
    parser.add_argument("--min-atr", type=float, default=MIN_ATR, help="Min ATR Filter")
    
    args = parser.parse_args()
    
    from utils import MIN_ATR # Import default if needed, though we set it in parser
    
    run_live_forward(
        model_name=args.model,
        symbol=args.symbol,
        account_balance=args.balance,
        risk_pct=args.risk,
        point_value=args.point_value,
        mt5_path=args.mt5_path,
        min_atr=args.min_atr,
    )
