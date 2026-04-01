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

## Results Summary

### Without Transaction Costs (Gross Profits)

| Model | Profit | Trades | Win Rate | Sharpe | Final Equity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM** | +145.0 units | 368 | 46.5% | 1.036 | 408% |
| **Transformer** | +2.0 units | 25 | 36.0% | 0.059 | 102% |

### With 1.0 Unit Spread (Standardized Test)

| Model | Timeframe | Profit | Sharpe | Max DD | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Transformer** | **10 min** | **+284.1 R** | **1.69** | **-15.9%** | 🏆 **Champion** |
| **LSTM** | **10 min** | **+148.8 R** | **1.11** | **-39.0%** | ✅ Profitable |

***Note**: Results verified on out-of-sample backtest data (Aug 2025 - Jan 2026).*

***Realistic Return:** Calculated with fixed position sizing (1 unit ≈ $10-15 on $10k account). For 5-minute intraday strategies, **7-11% over 6 months represents excellent performance** for an ML trading system. The "equity %" figures in raw backtester output use theoretical 1% risk compounding (unrealistic due to liquidity/scaling constraints).

**Key Finding:** The pivot to **10-minute candles** was decisive. The **Transformer** model outperformed the LSTM significantly (+284 R vs +148 R) on this timeframe, benefiting from the reduced noise and clearer trend signals. Both models are profitable when trained on general data and filtered at inference time.

## Data Splits (No Leakage)

| Split | Period | Samples | Use |
| :--- | :--- | :--- | :--- |
| **Train** | Aug 2019 – Aug 2025 | 36,473 | Training + validation (recency-weighted) |
| **Backtest** | Aug 2025 – Jan 2026 | 3,139 | Unseen evaluation with transaction costs |
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

5. **Labels** — SL=1×ATR, TP=2.0×ATR (break-even 33.3%), max 12 bars → `data/labeled/open_labeled.csv`

   ```bash
   python label_generator.py
   ```

6. **Dataset** — Window=24, train/backtest .npz → `data/processed/`

   ```bash
   python dataset_builder.py
   ```

7. **Train** — LSTM and Transformer with transaction cost awareness (cost=1.5 by default)

   ```bash
   python train.py lstm
   python train.py transformer
   ```

   Models select best checkpoint by validation reward (penalized by 1.5 unit transaction costs by default).
   Uses recency weighting (α=2.0) to prioritize recent market patterns.
   To disable cost penalty: `python train.py lstm --transaction-cost 0`

8. **Evaluate** — ML metrics, stability, backtest equity with transaction costs (0.2 spread by default)

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

    (Optional: `--symbol NDX100`, `--balance 10000`, `--risk 0.01`)

## Outputs

- **data/** — raw, session_filtered, features, labeled, processed
- **experiments/** — `lstm_open/best.pt`, `transformer_open/best.pt`
- **results/** — backtest equity/metrics, plots, `forward_test/live_signals.csv`, `forward_test/session_pnl.csv`

## Notebook

`notebooks/pipeline_overview.ipynb` — high-level pipeline overview (step list; run the steps above from `src/`).

## Documentation

- **[README.md](README.md)** - Main documentation (this file)
- **[SETUP.md](SETUP.md)** - Installation and environment setup
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Codebase organization and file descriptions
- **[THESIS_RESULTS.md](THESIS_RESULTS.md)** - Complete research findings and analysis
- **[REALISTIC_EXPECTATIONS.md](REALISTIC_EXPECTATIONS.md)** - ⚠️ **IMPORTANT**: Why 7-11% is excellent (not 196%)


## Citation

If you use this code in your research, please cite:

```latex
@mastersthesis{yourlastname2026,
  title={Comparing LSTM and Transformer Stability on High-Volatility Nasdaq100 Opening Session},
  author={Your Name},
  year={2026},
  school={Your University}
}
```
