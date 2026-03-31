# Changelog

## Version 1.0.0 (February 2026) - Transaction Cost Aware Release

### Major Changes

#### Transaction Cost Defaults
- **Training**: Default `--transaction-cost` changed from `1.0` to `1.5` units
  - Models now train with realistic cost penalties by default
  - Use `--transaction-cost 0` to disable cost-aware training
  
- **Backtesting**: Default `--spread` changed from `0.0` to `0.2` units
  - Backtest results now show net profits (after costs) by default
  - Use `--spread 0.0` to see gross profits

#### Configuration Updates
- **TP/SL Ratio**: Changed from 1.6:1 to 2.0:1
  - Take Profit: 2.0 × ATR (was 1.6)
  - Stop Loss: 1.0 × ATR (unchanged)
  - Break-even win rate: 33.3% (was 38.5%)

- **Recency Weighting**: Implemented exponential weighting (α=2.0)
  - 2025 data weighted 7.4× more than 2019 data
  - Prioritizes recent market patterns while maintaining historical context

#### Feature Engineering
Added new features for better market context:
- **Volatility Regime**: ATR vs 20-period MA ratio
- **Overnight Gaps**: Gap percentage and direction
- **Session Context**: Distance from previous session high/low
- **Range Expansion**: Current vs average bar range

#### Data Pipeline
- **Training Period**: Extended to Aug 2019 - Aug 2025 (6 years, 36,473 samples)
- **Backtest Period**: Aug 2025 - Jan 2026 (180 days, 3,139 samples)
- **Data Sources**: Kaggle (2019-08 to 2024-08-09) + MT5 (2024-08-29 to 2026-01-30)
- **Default Merge**: Automatically merges Kaggle + MT5 exports

### Results

#### Final Performance (With 0.2 Spread)
| Model | Net Profit | Trades | Win Rate | Sharpe | Max DD | Final Equity |
|-------|-----------|--------|----------|--------|--------|--------------|
| **LSTM** | +71.4 units | 368 | 46.5% | 0.522 | -30.2% | 196% |
| **Transformer** | -3.0 units | 25 | 36.0% | -0.088 | -13.5% | 97% |

**Key Finding**: LSTM demonstrates superior robustness to transaction costs with balanced trade frequency, while Transformer becomes overly conservative under cost pressure.

### Bug Fixes
- Fixed `visualize.py` equity column reference (`equity` → `equity_pct`)
- Fixed PyTorch `enable_nested_tensor` warning in Transformer model
- Fixed PowerShell command compatibility issues

### Documentation
- **NEW**: `PROJECT_STRUCTURE.md` - Complete codebase organization guide
- **NEW**: `THESIS_RESULTS.md` - Research findings and academic summary
- **NEW**: `QUICK_REFERENCE.md` - Command reference and troubleshooting
- **NEW**: `CHANGELOG.md` - This file
- **UPDATED**: `README.md` - Current configuration and results
- **UPDATED**: All docstrings reflect TP=2.0 configuration

### Breaking Changes
⚠️ **Models trained before this version are incompatible** due to:
- Different TP/SL ratio (labels regenerated with TP=2.0)
- New feature count (43 vs previous 37-40)
- Different transaction cost penalty during training

**Action Required**: Regenerate all data and retrain models:
```bash
cd src
python label_generator.py
python dataset_builder.py
python train.py lstm
python train.py transformer
python backtester.py --both
```

### Command Changes

**Before:**
```bash
# Training (no cost awareness)
python train.py lstm

# Backtesting (gross profits)
python backtester.py lstm
```

**After:**
```bash
# Training (cost=1.5 by default)
python train.py lstm

# Backtesting (0.2 spread by default)
python backtester.py lstm

# To disable costs:
python train.py lstm --transaction-cost 0
python backtester.py lstm --spread 0.0
```

---

## Version 0.1.0 (Initial Development)

### Initial Implementation
- Basic LSTM and Transformer models
- Kaggle data pipeline (2019-2024)
- TP=1.6, SL=1.0 configuration
- No transaction cost modeling
- Basic feature set (ATR, RSI, VWAP, returns)

### Known Issues (Resolved in v1.0.0)
- ❌ Models unprofitable with transaction costs
- ❌ No recency weighting
- ❌ Limited feature set
- ❌ Gross profits misleading
- ❌ Overtrading without cost penalties

---

## Migration Guide (v0.1.0 → v1.0.0)

### If You Have Old Models:

**Option 1: Full Regeneration (Recommended)**
```bash
cd src
python label_generator.py        # Regenerate with TP=2.0
python dataset_builder.py        # Rebuild datasets
python train.py lstm             # Retrain with cost=1.5
python train.py transformer
python backtester.py --both      # Evaluate with 0.2 spread
```

**Option 2: Keep Old Results for Comparison**
```bash
# Backup old experiments
cp -r experiments/ experiments_backup/

# Generate new results
cd src
python label_generator.py
python dataset_builder.py
python train.py lstm
python train.py transformer

# Compare old vs new
python backtester.py lstm --spread 0.0  # New model, no costs
# vs your old results
```

### Configuration File Updates

**`utils.py` - Update these constants:**
```python
# OLD
TP_ATR_MULT = 1.6
TRAIN_START = "2023-01-01"

# NEW
TP_ATR_MULT = 2.0
TRAIN_START = "2019-01-01"
```

### Expected Improvements

After migration to v1.0.0:
- ✅ Realistic profitability assessment (transaction costs included)
- ✅ Better generalization (recency weighting)
- ✅ More informed predictions (gap/volatility features)
- ✅ Reduced overtrading (cost-aware training)
- ✅ Clearer risk/reward (2:1 TP/SL ratio)

---

## Future Roadmap

### Planned Features
- [ ] Multi-timeframe analysis (5m, 15m, 1h)
- [ ] Ensemble model (LSTM + Transformer)
- [ ] Live forward testing automation
- [ ] VPS deployment scripts
- [ ] Real-time dashboard
- [ ] Alternative architectures (TCN, attention-LSTM)

### Research Directions
- [ ] Multi-regime testing (bear markets, high VIX)
- [ ] Feature ablation studies
- [ ] Hyperparameter optimization
- [ ] Risk-adjusted position sizing
- [ ] Adaptive thresholding

---

For questions or issues, see `README.md` and `QUICK_REFERENCE.md`.
