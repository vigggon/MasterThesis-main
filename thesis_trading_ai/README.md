# Thesis Trading AI

**Comparing LSTM vs Transformer stability on high-volatility Nasdaq100 opening session (09:30–11:30 NY).**

This project implements transaction-cost-aware deep learning models for intraday trading during the Nasdaq100 market opening. Data spans 2019–2026 (Kaggle + MetaTrader 5), with recency-weighted training, volatility/gap/overnight features, and realistic transaction cost modeling.

## Setup

See **[SETUP.md](SETUP.md)** for:

- Python 3.10–3.13, virtual environment, dependencies
- PyTorch with CUDA (optional)
- `setup.bat` / `setup.ps1` and `scripts/verify_gpu.py`

## Key Configuration

**Trading Parameters:**

- **Timeframe**: 10 Minutes (Resampled from 5m)
- **Stop Loss**: 1.0 × ATR
- **Take Profit**: 3.0 × ATR (Verified optimal)
- **Max Hold**: 12 candles (2 hours)
- **Transaction Cost**: Included in Training & Backtesting
- **Filter**: Min ATR 15.0 (Inference Only)

**Backtest:**

- **Period**: Aug 2025 – Jan 2026 (180 days, 3,139 samples)
- **Evaluation**: With realistic 0.2 unit spread cost

## Data Splits (No Leakage)

| Split | Period | Samples | Use |
| :--- | :--- | :--- | :--- |
| **Train** | Jan 2019 – Aug 2025 | 36,473 | Training + validation (recency-weighted) |
| **Backtest** | Aug 2025 – Jan 30 2026 | 3,139 | Unseen evaluation with transaction costs |
| **Forward** | Live only | N/A | Real-time testing via `run_live_forward.py` |

Data range: **Kaggle (2019-08 to 2024-08-09)** + **MT5 export (2024-08-29 to 2026-01-30)** merged by default (~20 day gap).

Put **Kaggle** `NQ_5Years_8_11_2024.csv` and **MT5 export** `NAS100_*.csv` in `data/raw/`; run `python data_download.py` to merge. Use `--no-merge` to use only Kaggle; forward testing uses live MT5 only (no pre-built file).

## Pipeline (run from `src/`)

Activate the venv, then:

1. **Download** — M5 data → `data/raw/nasdaq_m5.csv`  
   Put **Kaggle** `NQ_5Years_8_11_2024.csv` and **MT5 export** `NAS100_*.csv` (from 2024-08-29) in `data/raw/`. Then:

   ```bash
   python data_download.py
   ```

   The script **merges** Kaggle (to 2024-08-09) + MT5 (from 2024-08-29) → `nasdaq_m5.csv`. Use `--no-merge` to use only Kaggle.

   Optional: `--from-csv path/to/file.csv` or MT5 live: `--symbol NAS100`, `--list-symbols`, `--diagnose`.

2. **Volatility** — ATR(14), Open vs Rest → `data/volatility_analysis/`

   ```bash
   python volatility_validation.py
   ```

3. **Session** — 09:30–11:30 NY → `data/session_filtered/nasdaq_open_session.csv`

   ```bash
   python session_filter.py
   ```

4. **Features** — Returns, ATR, RSI, VWAP, volatility regime, overnight gaps, session context → `data/features/open_features.csv`

   ```bash
   python feature_engineering.py
   ```

5. **Labels** — SL=1×ATR, TP=3.0×ATR (3:1 RR), max 12 bars → `data/labeled/open_labeled.csv`

   ```bash
   python label_generator.py
   ```

6. **Dataset** — Window=24, train/backtest .npz → `data/processed/`

   ```bash
   python dataset_builder.py
   ```

7. **Train** — BiLSTM and Transformer with transaction cost awareness (cost=1.5 by default)

   ```bash
   python train.py lstm
   python train.py transformer
   ```

   Models select best checkpoint by validation reward (penalized by 1.5 unit transaction costs by default).
   Uses recency weighting (α=2.0) to prioritize recent market patterns.
   To disable cost penalty: `python train.py lstm --transaction-cost 0`

8. **Evaluate** — ML metrics, stability, backtest equity with transaction costs (0.2 spread by default unless parameter is specified)

   ```bash
   python evaluate_ml.py
   python stability_analysis.py
   python backtester.py lstm           # Net profit with 0.2 spread (default)
   python backtester.py --both         # Both models with costs
   python backtester.py --both --spread 0.0  # Gross profit (no costs)
   ```
   Transaction costs default to **0.2 units spread** (realistic for NAS100).
   Use `--spread 0.0` for gross profits or `--spread 0.5` for conservative estimates.

9. **Plots** — Backtest equity and stability -> `results/plots/`

   ```bash
   python scripts/generate_plots.py
   ```

10. **Live forward** — During NY open: MT5 + both models, signals and session P&L

    ```bash
    python run_live_forward.py
    ```

    (Optional: `--symbol USTEC`, `--balance 10000`, `--risk 0.01`)

## Outputs

- **data/** — raw, session_filtered, features, labeled, processed
- **experiments/** — `lstm_open/best.pt`, `transformer_open/best.pt`
- **results/** — backtest equity/metrics, plots, `forward_test/live_signals.csv`, `forward_test/session_pnl.csv`

## Documentation

- **[README.md](README.md)** - Main documentation (this file)
- **[SETUP.md](SETUP.md)** - Installation and environment setup
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Codebase organization and file descriptions
