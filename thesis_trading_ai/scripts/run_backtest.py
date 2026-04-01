import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtesting.backtester import run_backtester
from utils.config_loader import config
from utils.logger import get_logger

def main():
    logger = get_logger("backtest_runner")
    parser = argparse.ArgumentParser(description="Run Backtest for Thesis Models")
    parser.add_argument("--model", choices=["lstm", "transformer", "both"], default="both", help="Model to backtest")
    parser.add_argument("--spread", type=float, default=0.2, help="Spread in points (e.g. 0.2 for realistic NAS100)")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission in points")
    args = parser.parse_args()

    models_to_run = ["lstm", "transformer"] if args.model == "both" else [args.model]

    for model_name in models_to_run:
        logger.info(f"Running backtest for {model_name.upper()} (spread={args.spread})...")
        try:
            metrics = run_backtester(
                model_name=model_name,
                risk_pct=config.get("trading.risk_pct_per_trade", 0.005),
                max_dd_cap=config.get("trading.max_dd_cap", 0.25),
                spread_points=args.spread,
                commission_points=args.commission,
                dynamic_risk=False,
                daily_max_loss=config.get("optimization.daily_stop_r", -4.0),
                daily_take_profit=config.get("optimization.daily_tp_r", 12.0),
                min_atr=config.get("optimization.min_atr", 15.0),
            )
            logger.info(f"[{model_name.upper()}] Backtest completed. Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe_ratio']:.2f}")
        except Exception as e:
            logger.error(f"Failed to backtest {model_name}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
