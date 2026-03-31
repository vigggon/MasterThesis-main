# Quick Reference Guide

## Training Commands

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Navigate to src/
cd src

# Train models (transaction cost=1.5 by default)
python train.py lstm
python train.py transformer

# Override transaction cost penalty
python train.py lstm --transaction-cost 0.0    # No cost penalty
python train.py lstm --transaction-cost 2.0    # Higher penalty (more selective)
```

## Backtesting Commands

```bash
# Backtest single model (0.2 spread by default)
python backtester.py lstm
python backtester.py transformer

# Backtest both models
python backtester.py --both

# Backtest without transaction costs (gross profit)
python backtester.py --both --spread 0.0

# Backtest with different spread levels
python backtester.py lstm --spread 0.1    # Optimistic
python backtester.py lstm --spread 0.2    # Realistic (DEFAULT)
python backtester.py lstm --spread 0.5    # Conservative

# Threshold sweep (find optimal threshold)
python backtester.py transformer --sweep-threshold
```

## Complete Pipeline

```bash
# 1. Setup (one-time)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Data pipeline
cd src
python data_download.py               # Merge Kaggle + MT5
python volatility_validation.py       # Validate volatility
python session_filter.py              # Filter 09:30-11:30
python feature_engineering.py         # Create 43 features
python label_generator.py             # Generate labels (TP=2.0, SL=1.0)
python dataset_builder.py             # Build train/backtest .npz

# 3. Training (with cost awareness - enabled by default)
python train.py lstm
python train.py transformer

# 4. Evaluation (with 0.2 spread - enabled by default)
python backtester.py --both                # Net profits (realistic)
python backtester.py --both --spread 0.0   # Gross profits (no costs)
python evaluate_ml.py                      # ML metrics
python stability_analysis.py               # Daily variance
python visualize.py                        # Generate plots

# 5. Live forward testing (optional)
python run_live_forward.py
```

## Key Configuration Files

**`utils.py` - Global constants:**
```python
# Data splits
TRAIN_START = "2019-01-01"
BACKTEST_DAYS = 180  # Last 180 days

# Trading parameters
SL_ATR_MULT = 1.0
TP_ATR_MULT = 2.0    # Break-even: 33.3%
MAX_HOLD_CANDLES = 12
TRADE_THRESHOLD = 0.4

# Model architecture
LSTM_HIDDEN = 128
LSTM_LAYERS = 3
TX_DMODEL = 128
TX_NHEAD = 4
TX_LAYERS = 2
```

## Expected Results (Feb 2026 Configuration)

**Training:** 36,473 samples (Aug 2019 - Aug 2025), recency-weighted (α=2.0)  
**Backtest:** 3,139 samples (Aug 2025 - Jan 2026), with 0.2 spread

### LSTM
- **Net Profit**: +71.4 units
- **Trades**: 368
- **Win Rate**: 46.5%
- **Sharpe**: 0.522
- **Max DD**: -30.2%
- **Final Equity**: 196%

### Transformer
- **Net Profit**: -3.0 units
- **Trades**: 25
- **Win Rate**: 36.0%
- **Sharpe**: -0.088
- **Max DD**: -13.5%
- **Final Equity**: 97%

## Troubleshooting

**Problem: Models not trading**
```bash
# Check threshold tuning results
cat experiments/lstm_open/trade_threshold.json
cat experiments/transformer_open/trade_threshold.json

# Try lower threshold
python train.py lstm --tune-threshold
```

**Problem: Too many trades**
```bash
# Increase transaction cost penalty during training
python train.py lstm --transaction-cost 2.0
```

**Problem: Low win rate**
```bash
# Increase TP/SL ratio in utils.py
TP_ATR_MULT = 2.5  # Break-even: 28.6%

# Regenerate labels and retrain
python label_generator.py
python dataset_builder.py
python train.py lstm --transaction-cost 1.5
```

**Problem: High losses with costs**
```bash
# Check break-even spread
python -c "print(f'Break-even spread: {71.4/368:.3f} units')"

# Reduce spread or increase training cost penalty
python train.py lstm --transaction-cost 2.0  # More selective
```

**Problem: Data not found**
```bash
# Verify data pipeline ran
ls data/raw/nasdaq_m5.csv
ls data/session_filtered/nasdaq_open_session.csv
ls data/features/open_features.csv
ls data/labeled/open_labeled.csv
ls data/processed/train.npz
ls data/processed/backtest.npz
```

## Useful Analysis Commands

```bash
# Check training history
python -c "import json; print(json.load(open('experiments/lstm_open/history.json', 'r'))['val_reward'][-10:])"

# Check data splits
python -c "import numpy as np; d=np.load('data/processed/train.npz', allow_pickle=True); print(f'Train: {d[\"times\"][0]} to {d[\"times\"][-1]} ({len(d[\"times\"])} samples)')"

# Calculate recency weights
python -c "import numpy as np; n=36473; w=np.exp(2*np.arange(n)/n); print(f'2025 vs 2019: {w[-1]/w[0]:.1f}x')"

# Break-even analysis
python -c "from utils import TP_ATR_MULT, SL_ATR_MULT; be=SL_ATR_MULT/(SL_ATR_MULT+TP_ATR_MULT); print(f'Break-even win rate: {be:.1%}')"
```

## File Locations

**Models**: `experiments/{model}_open/best.pt`  
**Metrics**: `results/backtest/{model}_metrics.csv`  
**Equity**: `results/backtest/{model}_equity.csv`  
**Plots**: `results/plots/`  
**Logs**: `results/forward_test/`

## Documentation

- `README.md` - Main documentation
- `SETUP.md` - Installation guide
- `PROJECT_STRUCTURE.md` - Codebase structure
- `THESIS_RESULTS.md` - Research findings
- `QUICK_REFERENCE.md` - This file
