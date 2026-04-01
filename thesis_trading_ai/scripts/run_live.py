import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.config_loader import config
from utils.logger import get_logger
from live.run_live_forward import run_live_forward

def main():
    logger = get_logger("live_runner")
    
    parser = argparse.ArgumentParser(description="Run Live Forward Test")
    parser.add_argument("model", choices=["lstm", "transformer"], help="Model to run")
    parser.add_argument("--symbol", type=str, default=None, help="MT5 symbol (override config)")
    parser.add_argument("--mt5-path", type=str, default=None, help="Path to specific terminal64.exe")
    args = parser.parse_args()

    symbol = args.symbol or config.get("data.symbols")[0]
    
    logger.info(f"Starting live forward test for model: {args.model.upper()}")
    logger.info(f"Symbol: {symbol}")
    
    try:
        run_live_forward(
            model_name=args.model,
            symbol=symbol,
            account_balance=config.get("trading.account_balance", 10000.0),
            risk_pct=config.get("trading.risk_pct_per_trade", 0.005),
            point_value=config.get("trading.point_value_per_lot", 1.0),
            mt5_path=args.mt5_path,
            min_atr=config.get("optimization.min_atr", 15.0),
        )
    except KeyboardInterrupt:
        logger.info("Live run stopped by user.")
    except Exception as e:
        logger.error(f"Error during live run: {e}", exc_info=True)

if __name__ == "__main__":
    main()
