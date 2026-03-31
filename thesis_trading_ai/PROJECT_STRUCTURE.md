# Project Structure

## Core Pipeline Scripts (Run from `src/`)

**Main workflow (in order):**
1. `data_download.py` — Download/merge Kaggle + MT5 data
2. `volatility_validation.py` — Validate opening session volatility
3. `session_filter.py` — Filter 09:30-11:30 NY opening session
4. `feature_engineering.py` — Create 43 features (ATR, RSI, gaps, volatility regime, etc.)
5. `label_generator.py` — Generate 3-class labels (hold/long/short) with TP=2.0×ATR, SL=1.0×ATR
6. `dataset_builder.py` — Build train/backtest .npz files with sequences
7. `train.py` — Train LSTM/Transformer with transaction cost awareness
8. `backtester.py` — Evaluate on backtest period with/without transaction costs
9. `evaluate_ml.py` — ML metrics (accuracy, precision, recall, F1)
10. `stability_analysis.py` — Daily accuracy variance over time
11. `visualize.py` — Generate equity/stability plots

**Live trading:**
- `run_live_forward.py` — Real-time forward testing during market hours

## Utility Scripts

**Data utilities:**
- `fetch_mt5.py` — Direct MT5 download helper
- `find_mt5_symbol.py` — Search for MT5 symbols
- `diag_mt5.py` — Diagnose MT5 connection issues
- `process_manual_csv.py` — Process manually exported MT5 CSV
- `auto_fetch_any_nasdaq.py` — Auto-detect Nasdaq symbols

**Model utilities:**
- `tune_lstm.py` — Hyperparameter grid search for LSTM
- `tune_placeholder.py` — Template for hyperparameter tuning
- `ensemble_backtester.py` — Combine LSTM + Transformer predictions
- `forward_test.py` — Alternative forward testing script

## Models

Located in `src/models/`:
- `lstm_model.py` — 3-layer LSTM (128 hidden units)
- `transformer_model.py` — Transformer encoder (128 d_model, 4 heads, 2 layers)

## Configuration

`src/utils.py` — All global constants:
- Data splits (TRAIN_START, BACKTEST_DAYS)
- Trading parameters (TP_ATR_MULT=2.0, SL_ATR_MULT=1.0, TRADE_THRESHOLD=0.4)
- Model architecture (LSTM_HIDDEN=128, TX_DMODEL=128, etc.)
- Paths (DATA_ROOT, EXPERIMENTS_ROOT, RESULTS_ROOT)

## Data Directory Structure

```
data/
├── raw/                          # Source data
│   ├── NQ_5Years_8_11_2024.csv  # Kaggle data (2019-08 to 2024-08-09)
│   ├── NAS100_*.csv              # MT5 export (2024-08-29 to 2026-01-30)
│   └── nasdaq_m5.csv             # Merged output
├── volatility_analysis/          # ATR statistics
├── session_filtered/             # 09:30-11:30 NY data
│   └── nasdaq_open_session.csv
├── features/                     # 43 features
│   └── open_features.csv
├── labeled/                      # 3-class labels
│   └── open_labeled.csv
└── processed/                    # Training-ready
    ├── train.npz                 # 36,473 samples (2019-08 to 2025-08)
    ├── backtest.npz              # 3,139 samples (2025-08 to 2026-01)
    └── feature_cols.csv
```

## Experiments Directory

```
experiments/
├── lstm_open/
│   ├── best.pt                  # Best checkpoint
│   ├── history.json             # Training curves
│   ├── trade_threshold.json     # Optimal threshold
│   └── feature_cols.json
└── transformer_open/
    └── (same structure)
```

## Results Directory

```
results/
├── backtest/                    # Equity curves + metrics
│   ├── lstm_equity.csv
│   ├── lstm_metrics.csv
│   ├── transformer_equity.csv
│   └── transformer_metrics.csv
├── ml_metrics/                  # Classification metrics
├── stability/                   # Daily accuracy variance
├── plots/                       # Visualizations
└── forward_test/                # Live trading logs
    ├── live_signals.csv
    └── session_pnl.csv
```

## Key Files

- `README.md` — Main documentation
- `SETUP.md` — Installation and environment setup
- `requirements.txt` — All dependencies
- `setup.bat` / `setup.ps1` — Automated setup scripts
- `PROJECT_STRUCTURE.md` — This file

## Notebooks

- `notebooks/pipeline_overview.ipynb` — High-level overview (non-executable, see src/ for actual scripts)

## Development Notes

**Python Cache:**
- `src/__pycache__/` and `src/models/__pycache__/` contain compiled bytecode
- Safe to delete; regenerated on next run

**Transaction Cost Training:**
- Models trained with `--transaction-cost 1.5` during validation reward calculation
- Forces selective trading behavior
- Backtest separately evaluates with actual spread costs (`--spread 0.2`)

**Recency Weighting:**
- Training uses exponential weighting (α=2.0): `weight[i] = exp(2.0 × i/n)`
- 2025 data weighted ~7.4× more than 2019 data
- Prevents overfitting to old regimes while maintaining historical context
