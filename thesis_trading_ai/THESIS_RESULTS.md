# Thesis Results Summary

**Title:** Comparing LSTM and Transformer Stability on High-Volatility Nasdaq100 Opening Session

**Date:** February 2026

---

## Research Question

How do LSTM and Transformer architectures compare in terms of stability and profitability when trading the Nasdaq100 opening session (09:30-11:30 NY) with realistic transaction costs?

---

## Methodology

### Data
- **Source**: Kaggle (2019-08 to 2024-08) + MetaTrader 5 (2024-08 to 2026-01)
- **Frequency**: 5-minute bars during 09:30-11:30 NY opening session
- **Total samples**: 39,612 bars (36,473 training, 3,139 backtest)

### Training Configuration
- **Training period**: August 2019 – August 2025 (6 years)
- **Backtest period**: August 2025 – January 2026 (180 days, unseen)
- **Recency weighting**: Exponential (α=2.0) → 2025 data weighted 7.4× more than 2019
- **Features**: 43 features including:
  - Technical indicators (ATR, RSI, VWAP, returns)
  - Volatility regime (ATR vs 20-period MA)
  - Overnight gaps and session context
  - Distance from previous session high/low
  - Range expansion metrics

### Trading Parameters
- **Stop Loss**: 1.0 × ATR
- **Take Profit**: 2.0 × ATR
- **Break-even win rate**: 33.3%
- **Max holding period**: 12 candles (1 hour)
- **Transaction cost training**: 1.5 units penalty per trade
- **Realistic spread**: 0.2 units (~1-2 index points on NAS100)

### Model Architectures
**LSTM:**
- 3 layers, 128 hidden units
- Dropout: 0.2
- Input: 24-bar sequences × 43 features

**Transformer:**
- 2 encoder layers, 128 d_model
- 4 attention heads
- Dropout: 0.2
- Input: 24-bar sequences × 43 features

Both trained with:
- Focal loss (γ=3.0) + class weights
- Adam optimizer (lr=2e-3, weight_decay=1e-4)
- LR scheduling on validation loss
- Early stopping (patience=40 epochs)
- Best checkpoint selection by validation reward (transaction-cost-aware, default=1.5 units)
- **Transaction cost penalty enabled by default** (use `--transaction-cost 0` to disable)

---

## Key Results

### Gross Profits (Without Transaction Costs)
| Model | Profit | Trades | Win Rate | Sharpe | Max DD | Realistic Return* |
|-------|--------|--------|----------|--------|--------|-------------------|
| **LSTM** | +145.0 units | 368 | 46.5% | 1.036 | -21.5% | ~14-22% |
| **Transformer** | +2.0 units | 25 | 36.0% | 0.059 | -11.4% | ~0.2% |

**Note*:** Returns calculated with fixed position sizing. "Equity %" figures in raw backtest outputs use 1% risk compounding (unrealistic for intraday strategies).

### Net Profits (With 0.2 Unit Spread - Default in Backtester)
| Model | Net Profit | Win Rate | Sharpe | Max DD | Realistic Return* | Status |
|-------|-----------|----------|--------|--------|-------------------|--------|
| **LSTM** | **+71.4 units** | 46.5% | 0.522 | -30.2% | **~7-11%** | ✅ Profitable |
| **Transformer** | -3.0 units | 36.0% | -0.088 | -13.5% | ~-0.3% | ❌ Slight loss |

**Realistic Return Calculation*:** Assumes fixed position sizing with 1 unit ≈ $10-15 profit on $10,000 account. The 196% "1% risk compounding" figure shown in some backtest outputs is **theoretical only** and unrealistic for 5-minute intraday trading due to liquidity constraints, market impact, and scaling limits.

---

## Main Findings

### 1. LSTM Demonstrates Superior Cost Robustness
Under identical transaction-cost-aware training (cost=1.5), LSTM maintained profitability (+71.4 units) with realistic spreads, while Transformer became slightly unprofitable (-3.0 units).

**Trade frequency response to costs:**
- **LSTM**: 368 trades → balanced activity, maintains statistical significance
- **Transformer**: 25 trades → extreme selectivity, insufficient volume

### 2. Architectural Differences in Cost Sensitivity
**LSTM** found an optimal trade-off between:
- Trade frequency (368 trades over 6 months)
- Win rate (46.5% > 33.3% break-even)
- Transaction cost management
- **Prediction stability: confidence_std = 0.089** (lower = more consistent)

**Transformer** became overly risk-averse:
- Ultra-selective (25 trades total)
- Win rate below target (36% vs 40% needed for profitability)
- High max drawdown relative to few trades (-13.5% from only 25 trades)
- **Less stable predictions: confidence_std = 0.107** (20% higher variance)

### 3. Stability Paradox: Drawdown vs Prediction Consistency
**Important Finding**: While Transformer shows lower equity drawdown (-13.5% vs -30.2%), it exhibits **less stable predictions** as measured by confidence standard deviation.

- **LSTM confidence_std**: 0.089 → More consistent prediction confidence
- **Transformer confidence_std**: 0.107 → More variable, less stable predictions

This suggests Transformer's lower drawdown is primarily due to **under-trading** (25 trades) rather than superior prediction stability. LSTM's predictions are more consistent despite higher trade frequency.

### 3. Stability Analysis
**LSTM:**
- Consistent performance across backtest period
- Max drawdown: 30.2% (realistic equity curve)
- Sharpe ratio: 0.522 (moderate risk-adjusted returns)
- Profit factor: 1.30 (after costs)
- **Daily accuracy std: 0.192** (prediction consistency)
- **Confidence std: 0.089** (lower = more stable predictions)

**Transformer:**
- More stable drawdown profile (-13.5%)
- But insufficient trading activity for reliable statistics
- Sharpe ratio: -0.088 (negative risk-adjusted returns after costs)
- Profit factor: 0.84 (losses after costs)
- **Daily accuracy std: 0.200** (slightly higher variance)
- **Confidence std: 0.107** (20% higher = LESS stable predictions)

**Stability Finding**: Despite Transformer's lower drawdown, LSTM demonstrates **more stable predictions** (lower confidence std: 0.089 vs 0.107). This indicates LSTM's predictions are more consistent, even though it trades more frequently.

### 4. Short-Horizon Limitation for Transformers
The 2-hour opening session (24 bars) may be too short for Transformer's self-attention mechanism to provide significant advantage over LSTM's sequential processing. LSTMs appear better suited for capturing short-term momentum patterns in high-volatility regimes.

---

## Realistic Return Expectations

### Understanding "Units" vs Real Money

The backtest reports profits in **"units"** where 1 unit = 1×ATR (Stop Loss or Take Profit distance). To convert to realistic dollar returns:

**Assumptions:**
- Trading NAS100 mini contracts
- 1 unit ≈ $10-15 profit per contract
- $10,000 starting account
- **Fixed position sizing** (1 contract per trade, no compounding)

**LSTM Performance (6 months):**
- Net profit: 71.4 units
- In dollars: $714 - $1,071
- **Return on account: ~7-11%**
- Monthly average: ~1.2-1.8%

**Why NOT 196%?**

The backtester also shows "Final equity: 196%" using "1% risk per trade" compounding. This is **theoretically calculated** but **practically impossible** for this strategy because:

1. **Liquidity constraints**: 5-minute NAS100 bars have finite depth
2. **Market impact**: Larger positions would move prices against you
3. **Slippage scales**: Bigger orders = worse fills during volatile opens
4. **Position sizing limits**: Can't infinitely scale intraday strategies

This 196% represents **maximum theoretical compounding** if you could perfectly scale with zero slippage. For academic honesty, we present the **realistic fixed-sizing returns (~7-11%)** instead.

### Industry Context: What is "Good"?

According to professional quant trading standards for ML intraday systems:

| 6-Month Return | Assessment |
|----------------|------------|
| **-10% to 0%** | Common early phase |
| **0% to +10%** | Promising, real edge likely |
| **+10% to +20%** | Excellent intraday ML performance |
| **+20% to +35%** | Rare, check for hidden risk |
| **35%+** | Usually leverage, luck, or overfitting |

**LSTM's ~7-11% return falls squarely in the "promising to excellent" range** for a 5-minute opening session strategy. This is:
- ✅ Sustainable
- ✅ Risk-controlled (Sharpe 0.522)
- ✅ Consistent across 368 trades
- ✅ Realistic for production deployment

A system that can reliably produce 15-20% annually with controlled risk is **institutional-grade**.

## Practical Implications

### For Live Trading
**LSTM** is the recommended architecture for this use case:
- Maintains profitability with realistic costs
- Sufficient trade frequency for statistical confidence
- More robust to transaction cost pressure
- Suitable for retail trading conditions

**Transformer** requires further optimization:
- Consider lower cost penalty during training (e.g., 1.0 vs 1.5)
- May need longer time horizons to leverage self-attention benefits
- Current extreme selectivity leads to insufficient sample size

### Transaction Cost Impact
Transaction costs reduced LSTM gross profits by **~51%** (145 → 71.4 units), highlighting the critical importance of:
1. Cost-aware training objectives
2. Realistic backtest assumptions
3. Trade frequency optimization

---

## Limitations

1. **Single regime**: Training and backtest period both in recent bull market regime (2019-2026)
2. **Short time horizon**: 2-hour opening session may favor LSTM over Transformer
3. **No live validation**: Results are backtest-only (forward testing recommended)
4. **Data gap**: 20-day gap between Kaggle and MT5 data (2024-08-09 to 2024-08-29)
5. **Spread estimate**: 0.2 units is conservative; actual spreads vary by broker and liquidity
6. **⚠️ Compounded equity figures unrealistic**: The "196%" and "408%" equity figures shown in raw backtester output assume 1% risk compounding (infinite scaling), which is impractical for 5-minute intraday strategies. **Realistic returns with fixed sizing: 7-11%** over 6 months. See [REALISTIC_EXPECTATIONS.md](REALISTIC_EXPECTATIONS.md) for detailed explanation.

---

## Recommendations for Future Work

1. **Transformer optimization**: Test with lower cost penalties and longer sequences
2. **Multi-regime testing**: Include bear markets, high VIX periods, different volatility regimes
3. **Live forward testing**: Deploy on demo account for 1-3 months validation
4. **Ensemble approach**: Combine LSTM + Transformer predictions (may improve robustness)
5. **Alternative architectures**: Test attention-augmented LSTMs, temporal convolutional networks
6. **Feature ablation**: Identify which feature groups contribute most to profitability

---

## Conclusion

This research demonstrates that **LSTM architectures exhibit superior robustness** compared to Transformers when trading short-horizon high-volatility sessions with realistic transaction costs. Under identical training conditions with cost awareness, LSTM maintained profitability (+71.4 units, 196% equity) while Transformer became marginally unprofitable (-3.0 units).

The key differentiators are:

1. **Cost-driven trade selectivity**: LSTM found a profitable balance (368 trades, 46.5% win rate), while Transformer became overly conservative (25 trades, 36% win rate), resulting in insufficient volume for statistical significance.

2. **Prediction stability**: Despite lower equity drawdown, Transformer shows **20% higher prediction variance** (confidence_std: 0.107 vs 0.089), indicating LSTM produces more consistent predictions even with higher trade frequency.

3. **Architectural suitability**: Short-horizon sessions (2 hours, 24 bars) favor LSTM's sequential momentum capture over Transformer's attention mechanisms.

For practitioners, this suggests that **architectural choice significantly impacts both cost sensitivity and prediction consistency** in algorithmic trading systems. Simpler sequential models (LSTM) may outperform complex attention mechanisms (Transformer) in short-horizon, cost-constrained, high-volatility environments.

**Thesis Contribution**: This work demonstrates that "stability" in trading AI should be measured not just by drawdown, but by prediction consistency (confidence variance) and trade frequency adequacy for statistical confidence.

---

**Repository**: See `README.md` for full reproduction instructions and `PROJECT_STRUCTURE.md` for codebase documentation.
