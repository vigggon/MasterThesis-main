# Project Structure (Refactored)

## High-Level Entrypoints (`/scripts`)

Use these scripts for standard thesis workflows:

- `run_backtest.py` — Runs standardized backtest for LSTM and Transformer.
- `run_live.py` — Entry point for live forward testing (MetaTrader 5).
- `generate_plots.py` — Processes results and generates all thesis-ready plots and metrics.

## Source Code (`/src`)

The core logic is divided into modular packages:

### `src/backtesting`

- `backtester.py` — Core simulation engine. Handles trade resolution, equity calculation, and standardized CSV exports.

### `src/features`

- `feature_engineering.py` — Logic for building 43 technical and contextual features.
- `label_generator.py` — Time-based 2-hour session labeling.
- `dataset_builder.py` — Preparation of `.npz` files for training.

### `src/utils`

- `config_loader.py` — Singleton configuration manager (reads `config/config.yaml`).
- `logger.py` — Standardized logging factory.

### `src/live`

- `run_live_forward.py` — Main execution loop for live MT5 polling and trade execution.

### `src/models`

- Model definitions for LSTM and Transformer architectures.

## Configuration (`/config`)

- `config.yaml` — **Central source of truth** for all parameters (TP/SL, Symbols, Batch Size, etc.).

## Results (`/results`)

Standardized output directory for thesis evaluation:

### `/results/backtest`

- `transformer_trades.csv`, `lstm_trades.csv` — Individual trade logs.
- `transformer_equity.csv`, `lstm_equity.csv` — Equity and drawdown series.
- `backtest_metrics.csv` — Summary statistics (Sharpe, PF, MaxDD, CVaR).

### `/results/live`

- `transformer_trades.csv`, `lstm_trades.csv` — Forward test results mapped to standardized format.
- `transformer_equity.csv`, `lstm_equity.csv` — Descriptive equity curves.
- `live_metrics.csv` — Behavioral summary statistics.

### `/results/plots`

- Equity curves, drawdown charts, return distributions, rolling Sharpe, and Monte Carlo permutation tests.

---

## Technical Legacy

Older scripts in `src/` (e.g., `visualize.py`, `evaluate_ml.py`) are retained for compatibility but have been superseded by the `scripts/` entrypoints for the final thesis pipeline.
