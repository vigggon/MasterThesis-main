import yaml
from pathlib import Path

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
VOLATILITY_DIR = DATA_ROOT / "volatility_analysis"
SESSION_DIR = DATA_ROOT / "session_filtered"
FEATURES_DIR = DATA_ROOT / "features"
LABELED_DIR = DATA_ROOT / "labeled"
PROCESSED_DIR = DATA_ROOT / "processed"

EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
RESULTS_ROOT = PROJECT_ROOT / "results"

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
        with open(CONFIG_PATH, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split(".")
        val = self._config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

config = ConfigLoader()

# Exports for backward compatibility where possible,
# to minimize changes in other files, but config.get() is preferred.

TRAIN_START = config.get("data.train_start")
TRAIN_END = config.get("data.train_end")
BACKTEST_DAYS = config.get("data.backtest_days")
BACKTEST_START = config.get("data.backtest_start")
BACKTEST_END = config.get("data.backtest_end")

OPEN_START = config.get("data.open_start")
OPEN_END = config.get("data.open_end")
NY_TZ = config.get("data.ny_tz")

WINDOW = config.get("features.window", 24)
ATR_PERIOD = config.get("features.atr_period", 14)

SL_ATR_MULT = config.get("trading.sl_atr_mult", 1.0)
TP_ATR_MULT = config.get("trading.tp_atr_mult", 3.0)
MAX_HOLD_CANDLES = config.get("trading.max_hold_candles", 12)
TRADE_THRESHOLD = config.get("trading.trade_threshold", 0.4)
NUM_CLASSES = config.get("trading.num_classes", 3)
RISK_PCT_PER_TRADE = config.get("trading.risk_pct_per_trade", 0.005)
MAX_DD_CAP = config.get("trading.max_dd_cap", 0.25)
ACCOUNT_BALANCE = config.get("trading.account_balance", 10000.0)
POINT_VALUE_PER_LOT = config.get("trading.point_value_per_lot", 1.0)

DAILY_STOP_R = config.get("optimization.daily_stop_r", -4.0)
DAILY_TP_R = config.get("optimization.daily_tp_r", 12.0)
MIN_ATR = config.get("optimization.min_atr", 15.0)
ATR_SCALE_THRESHOLD = config.get("optimization.atr_scale_threshold", 1.3)

LSTM_HIDDEN = config.get("models.lstm.hidden_size", 128)
LSTM_LAYERS = config.get("models.lstm.num_layers", 1)
BEST_LSTM_DIR = EXPERIMENTS_ROOT / config.get("models.lstm.best_dir_name", "lstm_L1_U128")

TX_DMODEL = config.get("models.transformer.d_model", 64)
TX_NHEAD = config.get("models.transformer.nhead", 8)
TX_LAYERS = config.get("models.transformer.num_layers", 3)
BEST_TRANSFORMER_DIR = EXPERIMENTS_ROOT / config.get("models.transformer.best_dir_name", "transformer_L3_H8_D64")
