"""
Shim entry point for live forward testing.
Calls the actual implementation in src/live/run_live_forward.py
"""
import sys
import os

# Add the parent directory to sys.path so we can find 'live' and 'config'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from live.run_live_forward import run_live_forward
import argparse
from utils import MIN_ATR, ACCOUNT_BALANCE, RISK_PCT_PER_TRADE, POINT_VALUE_PER_LOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live forward test shim")
    parser.add_argument("model", choices=["lstm", "transformer"], help="Model to run")
    parser.add_argument("--symbol", type=str, default=None, help="MT5 symbol")
    parser.add_argument("--balance", type=float, default=ACCOUNT_BALANCE, help="Account balance")
    parser.add_argument("--risk", type=float, default=RISK_PCT_PER_TRADE, help="Risk fraction")
    parser.add_argument("--point-value", type=float, default=POINT_VALUE_PER_LOT, help="USD per point")
    parser.add_argument("--mt5-path", type=str, default=None, help="Path to specific terminal64.exe")
    parser.add_argument("--min-atr", type=float, default=MIN_ATR, help="Min ATR Filter")
    
    args = parser.parse_args()
    
    run_live_forward(
        model_name=args.model,
        symbol=args.symbol,
        account_balance=args.balance,
        risk_pct=args.risk,
        point_value=args.point_value,
        mt5_path=args.mt5_path,
        min_atr=args.min_atr,
    )
