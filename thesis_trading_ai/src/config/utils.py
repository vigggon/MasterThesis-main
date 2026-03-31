"""Shared utilities for thesis trading AI pipeline."""
from pathlib import Path

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
VOLATILITY_DIR = DATA_ROOT / "volatility_analysis"
SESSION_DIR = DATA_ROOT / "session_filtered"
FEATURES_DIR = DATA_ROOT / "features"
LABELED_DIR = DATA_ROOT / "labeled"
PROCESSED_DIR = DATA_ROOT / "processed"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
RESULTS_ROOT = PROJECT_ROOT / "results"

# Winners from Hyperparameter Sweep
BEST_TRANSFORMER_DIR = EXPERIMENTS_ROOT / "transformer_L3_H8_D64"
BEST_LSTM_DIR = EXPERIMENTS_ROOT / "lstm_L1_U128"

# Time splits (no leakage).
# Train: TRAIN_START through TRAIN_END (clipped to day before backtest when BACKTEST_DAYS set).
# Backtest: if BACKTEST_DAYS set, last N days of data; else BACKTEST_START to BACKTEST_END.
# Recommended with MT5 recent data: TRAIN_START 2022-01-01, BACKTEST_DAYS 90, TRAIN_END high so train uses all data up to backtest.
TRAIN_START = "2019-01-01"   # Train on full historical data with recency weighting
TRAIN_END = "2030-12-31"     # Ceiling; when BACKTEST_DAYS set, train actually ends at (last_date - BACKTEST_DAYS - 1)
BACKTEST_DAYS = 180          # Backtest = last 180 days (6 months) for more robust evaluation. None = use fixed range below
BACKTEST_START = "2024-01-01"
BACKTEST_END = "2026-12-31"  # Used only when BACKTEST_DAYS is None
# Forward = live only (no file; run_live_forward.py)

# Session
OPEN_START = "09:30"
OPEN_END = "11:30"
NY_TZ = "America/New_York"

# Model / labeling
WINDOW = 24
ATR_PERIOD = 14
SL_ATR_MULT = 1.0
TP_ATR_MULT = 3.0   # Break-even win rate = 1/(1+3.0) = 25.0% (3:1 R:R - overcomes realistic transaction costs)
MAX_HOLD_CANDLES = 12

# Trade decision: predict trade when P(trade) >= threshold (avoids collapse to 0 trades with argmax)
TRADE_THRESHOLD = 0.4

# 3-class: 0=hold, 1=long, 2=short
NUM_CLASSES = 3

# Model architecture (Winners from Jan 2026 Sweep)
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
TX_DMODEL = 64
TX_NHEAD = 8
TX_LAYERS = 3

# Backtest Risk / Optimization Targets
RISK_PCT_PER_TRADE = 0.005
MAX_DD_CAP = 0.25

# Live Trading Defaults
ACCOUNT_BALANCE = 10000.0
POINT_VALUE_PER_LOT = 1.0 # USD per point per 1 lot (e.g. NAS100)

# Aggressive Optimization Defaults (Verified Jan 2026)
DAILY_STOP_R = -4.0    # Circuit Breaker
DAILY_TP_R = 12.0      # Upside Cap
MIN_ATR = 15.0         # Low Vol Filter
ATR_SCALE_THRESHOLD = 1.3 # Scale down if > 1.3x Avg ATR

