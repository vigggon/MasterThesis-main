# Chapter 4 — Main Results

This chapter is divided into two sections as specified. Section 4.1 provides a tutorial-style exposition of the technical implementation, enabling full reproducibility. Section 4.2 presents the experimental results with tables, figures, and detailed discussion.

For reproducibility purposes, all code is available at: **<https://github.com/vigggon/MasterThesis-main>**. A permanent link for the code can also be found at: `[SOFTWARE_HERITAGE_LINK — to be added upon submission]`.

---

## 4.1 Technical Development

This section describes the end-to-end implementation of the trading system, from raw tick data ingestion through model deployment. The codebase is implemented in Python 3.11 using PyTorch 2.1.0, and is structured as a sequential pipeline where each stage produces intermediate outputs consumed by the next. The system was not derived from an existing tutorial or open-source trading framework; all components were developed from scratch for this thesis.

## 4.1.1 Project Architecture

The project follows a modular structure organised around a central configuration file and six sequential pipeline stages:

```text
thesis_trading_ai/
├── config/config.yaml           # Central parameter source
├── src/
│   ├── data/                    # Stage 1: Data acquisition
│   ├── features/                # Stages 2-4: Session filter, features, labels
│   ├── models/                  # Architecture definitions
│   ├── training/                # Stage 5: Training & sweep
│   ├── backtesting/             # Stage 6: Event-driven backtester
│   └── live/                    # Stage 7: MT5 live execution
├── scripts/                     # High-level entrypoints
├── experiments/                 # Model checkpoints per configuration
└── results/                     # Output CSVs and plots
```

All tuneable parameters — including ATR periods, stop-loss/take-profit multiples, trading session boundaries, and risk allocation — are centralised in `config/config.yaml` and loaded at runtime through a singleton configuration manager. This design ensures that experimental conditions can be modified in a single location without touching source code.

## 4.1.2 Data Acquisition and Preprocessing (Stage 1–2)

The pipeline begins by merging two data sources into a continuous 5-minute OHLCV series. The Kaggle dataset (August 2019 – August 2024) is loaded first, followed by the MetaTrader 5 export (August 2024 – February 2026). Column names are normalised to a canonical schema, and the two DataFrames are concatenated chronologically:

```python
# data_download.py — Merge logic (simplified)
df_kaggle = pd.read_csv(kaggle_path)
df_mt5 = pd.read_csv(mt5_export_path)

# Normalise column names: "Time" → "datetime", "Open" → "open", etc.
for c in df.columns:
    col_map[c] = c.strip().lower()

df = pd.concat([df_kaggle, df_mt5]).sort_values("datetime").reset_index(drop=True)
```

The merged dataset is then filtered to the New York Open session (09:30–11:30 EST) using timezone-aware datetime comparison. This stage eliminates approximately 80% of bars, retaining only the high-volatility window targeted by the research question:

```python
# session_filter.py — Core filter
def filter_open_session(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.tz_convert(NY_TZ).dt.time
    start = pd.Timestamp("09:30").time()
    end = pd.Timestamp("11:30").time()
    mask = (t >= start) & (t <= end)
    return df.loc[mask].copy()
```

## 4.1.3 Feature Engineering (Stage 3)

Session-filtered bars are resampled from 5-minute to 10-minute resolution using standard OHLCV aggregation rules (first open, max high, min low, last close, sum volume). Over 30 scale-invariant features are then computed. The design principle is that all features are expressed as ratios, percentages, or normalised values to prevent the model from being dominated by raw price magnitudes.

Key feature groups include:

**Volatility indicators** — ATR(14) serves as the primary volatility estimate and is used downstream for position sizing and barrier placement. The implementation follows the standard True Range definition:

```python
def atr(high, low, close, period):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()
```

**Cyclical time encodings** — Rather than feeding raw timestamps, which would create artificial discontinuities at midnight, sinusoidal encodings preserve the circular nature of time:

```python
minutes = dt.dt.hour * 60 + dt.dt.minute
df["time_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
df["time_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
```

Additional features include RSI(14), Bollinger Band width and normalised position, VWAP deviation, EMA-9 and EMA-21 distances, volume moving average ratios, overnight gap percentage, and multi-scale cumulative returns (1-hour and 2-hour windows). A volatility regime ratio compares current ATR to the value 6 bars (1 hour) prior to capture regime transitions.

## 4.1.4 Triple-Barrier Labelling (Stage 4)

Labels are generated using the Triple-Barrier Method. For each bar, the labeller simulates both a hypothetical long and short position forward through subsequent bars, checking whether the take-profit barrier (3.0 × ATR above/below entry), stop-loss barrier (1.0 × ATR), or maximum holding period (12 candles = 2 hours) is reached first:

```python
def label_candle_3class(row, future, atr_col="atr_14",
                        sl_mult=1.0, tp_mult=3.0, max_hold=12):
    long_win, long_bar = _simulate_long(row, future, ...)
    short_win, short_bar = _simulate_short(row, future, ...)
    if long_win and not short_win:
        return 1   # Long
    if short_win and not long_win:
        return 2   # Short
    if long_win and short_win:
        return 1 if long_bar <= short_bar else 2  # Earlier wins
    return 0       # Hold
```

The asymmetric 3:1 reward-to-risk ratio means the theoretical break-even win rate is 25%, creating a profitable framework even with modest directional accuracy. The three-class output (Hold=0, Long=1, Short=2) allows the model to abstain from trading when confidence is insufficient.

## 4.1.5 Sequence Construction and Temporal Split (Stage 5)

Labelled bars are assembled into overlapping sequences of length $W = 24$ (4 hours of 10-minute bars). Each sequence produces a feature tensor of shape $(W, F)$ where $F$ is the number of engineered features, paired with the label at the final timestep. A gap detector prevents sequences from spanning interruptions greater than 14 days:

```python
# dataset_builder.py — Gap-aware sequence construction
dt64 = dates.astype("datetime64[ns]")
diffs = dt64[1:] - dt64[:-1]
gap_indices = np.where(diffs > np.timedelta64(14, 'D'))[0] + 1

for i in range(window, len(df)):
    # Skip if a gap > 14 days falls inside this window
    if current_gap_ptr < len(gap_indices) and gap_indices[current_gap_ptr] < i:
        continue
    X.append(block[feature_cols].values)
    y.append(int(lab))
```

The temporal split allocates all data chronologically preceding August 29, 2025 for training, with the subsequent 180 days (August 29, 2025 – February 23, 2026) reserved as the strict out-of-sample backtest set. Within the training set, the last 15% is held out as a validation subset for checkpoint selection and threshold tuning.

## 4.1.6 Model Architectures

The two architectures differ substantially in both inductive bias and parameter count. The winning BiLSTM configuration (1 layer, 128 units, bidirectional with temporal attention) totals **210,820 trainable parameters**, while the winning Transformer (3 layers, 8 heads, d_model=64) totals **103,427 parameters**. Notably, the BiLSTM is the higher-capacity model — approximately twice the parameter count — despite the Transformer being architecturally more complex. This difference arises from the bidirectional design doubling the hidden state and the temporal attention projection layer. Both architectures were subject to identical regularisation, ensuring performance differences are attributed to architectural inductive bias rather than capacity advantage.

**Bidirectional LSTM with Temporal Attention** — The BiLSTM processes the input sequence bidirectionally, producing hidden states of size $2H$. A learned temporal attention mechanism computes a weighted sum over timesteps to isolate the most informative signals within the lookback window:

```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        scores = self.attn(lstm_out)          # (B, T, 1)
        weights = F.softmax(scores, dim=1)    # (B, T, 1)
        context = (weights * lstm_out).sum(dim=1)  # (B, H)
        return context
```

This attention mechanism allows the model to prioritize relevant historical bars over the final hidden state, which is particularly beneficial in high-noise financial data.

**Transformer Encoder** — The Transformer projects features to a $d_{model}$-dimensional embedding and uses stacked encoder layers with Pre-Layer Normalisation (`norm_first=True`) to stabilize training. Classification utilizes the last-token representation:

```python
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=8, num_layers=3, ...):
        self.proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=dropout,
            batch_first=True, norm_first=True  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Last-token pooling
        return self.fc(x)
```

The Pre-LN design follows Xiong et al. (2020), which demonstrated that placing layer normalisation before the attention and feedforward sublayers reduces gradient variance at initialisation and eliminates the need for learning rate warm-up schedules.

## 4.1.7 Training Protocol

Both architectures are trained with identical hyperparameters to isolate architectural effects. The loss function is Focal Loss with $\gamma = 3.0$ and label smoothing $\epsilon = 0.02$, combined with inverse-frequency class weighting:

```python
# Dynamic class weights based on training set distribution
count_tensor = torch.ones(num_classes)
for c, cnt in zip(uniq.tolist(), counts.tolist()):
    count_tensor[c] = float(cnt)
class_weights = (y_train.size(0) / (num_classes * count_tensor))

criterion = FocalLoss(weight=class_weights, gamma=3.0, label_smoothing=0.02)
```

**Recency-weighted sampling** ensures that late-stage observations are sampled more frequently, reflecting the non-stationary nature of financial series. **Checkpoint selection** utilizes a cost-aware validation reward rather than classification loss, saving the model with the highest net profit after transaction costs. Post-training, a decision threshold $\tau$ is tuned via grid search on the validation set.

## 4.1.8 Event-Driven Backtesting Engine (Stage 6)

The backtester replays the out-of-sample period bar-by-bar, enforcing a strict single-position constraint: a new trade cannot open until the previous one closes. For each bar where the model predicts Long or Short with probability exceeding the threshold $\tau$, the engine resolves the trade by walking forward through subsequent bars:

```python
# backtester.py — Trade resolution (simplified)
if direction == 1:  # Long
    sl = entry - current_atr * SL_ATR_MULT   # 1.0 × ATR below
    tp = entry + current_atr * TP_ATR_MULT   # 3.0 × ATR above
    for step in range(len(future)):
        if future_high[step] >= tp:
            raw_pnl = tp - entry; break
        if future_low[step] <= sl:
            raw_pnl = sl - entry; break
    # Convert to R-multiples and deduct transaction cost
    net_r = (raw_pnl / current_atr) - (spread_points / current_atr)
```

Trade results are expressed in R-multiples (profit normalised by the ATR-based stop distance), enabling comparison across varying volatility regimes. A daily circuit breaker halts trading after -4R cumulative loss, and an ATR floor filter (minimum 15 points) prevents trading in low-volatility conditions.

## 4.1.9 Live Forward Testing (Stage 7)

The live system connects to the MetaTrader 5 terminal via the `MetaTrader5` Python API, polling for new 10-minute candles during the New York Open session. When a new candle forms, the system computes features over the trailing 24-bar window, runs both models via GPU forward pass, and transmits trade orders with millisecond-latency execution. Each trade is logged with full indicator snapshots for post-hoc verification:

```python
# run_live_forward.py — Core polling loop (simplified)
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M10, 1, count)
df = pd.DataFrame(rates)

# Build features over trailing window
features = build_features(df)
X = features[-WINDOW:].values  # Last 24 bars

# Predict with both models
with torch.no_grad():
    logits = model(torch.tensor(X).unsqueeze(0).float())
    probs = torch.softmax(logits, dim=1)
```

Position sizing follows fixed fractional risk: each trade risks exactly 0.5% of current account equity, with the lot size calculated as $\text{lot} = (\text{equity} \times 0.005) / (\text{ATR} \times \text{point\_value})$.

---

## 4.2 Experimental Results

This thesis presents evidence suggesting that models optimised for offline Sharpe Ratio may be systematically biased toward higher trade frequency, which degrades real-world performance under execution constraints. The primary contribution of this work is the empirical demonstration that optimizing trading models for offline Sharpe Ratio induces a bias toward high trade frequency, which leads to performance degradation under real-world execution constraints. While the Transformer architecture consistently outperforms the BiLSTM in validation and out-of-sample backtests, this ranking reversed under live market conditions. Within the context of this study, these results indicate a potential bias in conventional financial machine learning evaluation pipelines toward high-frequency, execution-fragile models.

This section presents the experimental results obtained from the implementation described in Section 4.1. Results are organised chronologically: hyperparameter selection, out-of-sample backtesting, live forward testing, and cross-environment comparison.

## 4.2.1 Model Selection via Hyperparameter Sweep

A systematic grid search was conducted on the validation subset to identify the optimal configuration for each architectural paradigm. The Transformer search space varied encoder layers (1–3), attention heads (4–8), and embedding dimensions (64–128), yielding 12 configurations. The BiLSTM search space varied recurrent layers (1–3) and hidden units (32–128), yielding 9 configurations. The configuration maximising the validation Sharpe Ratio was selected as the representative prototype for each paradigm.

*Table 1: Hyperparameter Sweep Results — Top Configurations.*

| Model Type | Layers | Heads/Units | d_model | Total R | Trades | Win % | Sharpe | Realistic Return % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformer (Winner) | 3 | 8 heads | 64 | 54.08 | 187 | 48.1% | 0.820 | 27.0% – 40.6% |
| Transformer | 1 | 8 heads | 128 | 45.22 | 178 | 46.6% | 0.713 | 22.6% – 33.9% |
| Transformer | 1 | 4 heads | 128 | 43.06 | 176 | 46.6% | 0.679 | 21.5% – 32.3% |
| Transformer | 1 | 4 heads | 64 | 41.77 | 188 | 48.4% | 0.655 | 20.9% – 31.3% |
| BiLSTM | 1 | 128 units | — | 34.36 | 174 | 45.4% | 0.554 | 17.2% – 25.8% |
| BiLSTM | 2 | 128 units | — | 28.80 | 190 | 44.2% | 0.459 | 14.4% – 21.6% |
| BiLSTM | 3 | 64 units | — | 26.10 | 166 | 41.6% | 0.437 | 13.1% – 19.6% |

In Table 1, we present the top-performing configurations from the hyperparameter sweep, ranked by validation Sharpe Ratio. Several observations are noteworthy. First, the Transformer consistently outperforms the BiLSTM across all comparable configurations, with the winning Transformer (L3/H8/D64) achieving a Sharpe of 0.820 versus 0.554 for the winning BiLSTM (L1/U128), representing a 48% improvement. Second, deeper Transformer architectures (3 layers) outperform shallow ones (1 layer), consistent with findings by Vaswani et al. (2017) that deeper stacks capture longer-range dependencies. Third, the BiLSTM benefits from wider hidden states (128 units) but not from additional layers, suggesting that recurrent depth provides diminishing returns in this high-noise regime. The "Realistic Return" column assumes 0.5%–0.75% of account equity risked per trade, consistent with the fixed fractional position sizing described in the Methodology.

## 4.2.2 Out-of-Sample Backtest Evaluation

Following model selection, the winning configurations were evaluated on the strict 180-day out-of-sample holdout window (August 29, 2025 through February 23, 2026). A fixed transaction cost of 2.0 index points per round-trip was applied, consistent with realistic NAS100 execution friction during the New York Open session.

*Table 2: Out-of-Sample Backtest Performance Metrics (August 2025 – February 2026).*

| Metric | Transformer | BiLSTM |
| :--- | :--- | :--- |
| Total R-multiples | +46.10 | +26.10 |
| Trades (n) | 161 | 152 |
| Win Rate | 47.8% | 44.1% |
| Sharpe Ratio (95% CI) | **0.813** [0.30 – 1.38] | **0.492** [-0.04 – 1.00] |
| Profit Factor (95% CI) | 1.65 [1.27 – 2.67] | 1.36 [1.03 – 2.17] |
| Max Drawdown (%) | -2.63% | -4.29% |
| CVaR 95% | -1.00 R | -1.02 R |
| Realistic Return (0.5% Risk) | +23.1% | +13.1% |
| Realistic Return (0.75% Risk) | +34.6% | +19.6% |

In Table 2, we present the formal out-of-sample results on the 180-day holdout window that neither model has seen during training or hyperparameter selection. The 95% confidence intervals for Sharpe Ratio and Profit Factor are derived via bootstrap resampling (n=1,000 iterations), confirming that while the Transformer's edge is statistically stronger (lower bound roughly 0.30 Sharpe), both models exhibit high variance inherent to financial returns. While the confidence intervals overlap, the Transformer's distribution is clearly shifted toward higher values. The Transformer achieves a Sharpe ratio of 0.813, exhibiting negligible decay from its validation benchmark of 0.820. The BiLSTM exhibits a more pronounced decay from 0.554 to 0.492, though it remains consistently profitable.

### 4.2.2.1 Critical Assessment of Backtest Reality

While the out-of-sample results establish a strong baseline, the backtest remains structurally optimistic. First, performance is highly sensitive to the dynamically tuned decision threshold $\tau$ and the ATR multiplier parameters. Because these were optimised on the validation set, there remains a risk of overfitting to the validation Sharpe — the true optimal parameters in live, forward-moving conditions will inevitably drift. Second, the triple-barrier labelling framework introduces a subtler, but pervasive, structural advantage offline by assuming perfect, instantaneous liquidity at the barrier price levels. This introduces a significant upward bias in expected returns that is inherently asymmetric; because 'Long' and 'Short' labels are only assigned to trades that reach their take-profit barrier before their stop-loss, the backtest primarily inflates winning trades while under-representing the practical difficulty of capturing the full 3R payoff amidst real-market slippage.

Both architectures demonstrate maximum drawdowns well within acceptable bounds. The Transformer's peak-to-trough decline of -2.63% is notably conservative, indicating that its higher Sharpe is achieved without proportionally increased risk. The CVaR 95% values near -1.0 R for both models confirm that the worst 5% of trade outcomes are, on average, equivalent to exactly one stop-loss hit — validating that the triple-barrier framework and ATR-based position sizing successfully prevent catastrophic tail losses.

### Equity Curves

![Figure 1: Backtest Equity Curve — Transformer vs BiLSTM.](plots/backtest_equity_curve.pdf)
*Figure 1: Cumulative equity curves for both architectures over the 180-day out-of-sample holdout period (September 2025 – January 2026). The Transformer (blue) demonstrates a steeper equity trajectory with fewer prolonged flat periods compared to the BiLSTM (orange).*

The equity trajectories in Figure 1 reinforce the hypothesis that the Transformer extracts more consistent signals across the out-of-sample period. Its steeper accumulation, rather than a reliance on a few outlier wins, is consistent with the significantly higher trade frequency (161 vs 152) and win rate (47.8% vs 44.1%) reported in Table 2. The BiLSTM's prolonged flat period between October and November 2025 supports the earlier observation regarding its higher decision threshold and lower overall trade frequency. This visually supports the Transformer's superior pattern recognition capabilities within the simulated environment.

### Drawdown Analysis

![Figure 2: Backtest Drawdown Curve — Transformer vs BiLSTM.](plots/backtest_drawdown_curve.pdf)
*Figure 2: Drawdown profiles for both architectures. The Transformer exhibits a shallower maximum drawdown (-2.63%) compared to the BiLSTM (-4.29%).*

Figure 2 further delineates the difference in capital preservation behaviour between architectures. The BiLSTM's sustained drawdown of -4.29% near the end of the evaluation window contrasts with the Transformer's shallower maximum of -2.63%. This visually suggests that the Transformer’s attention mechanism is not merely an alpha generator but also an effective risk-management tool in simulated environments, achieving higher absolute returns without proportionally increasing downside volatility.

### Return Distributions

![Figure 3: Backtest Return Distribution — Transformer.](plots/returns_hist_transformer_backtest.pdf)
*Figure 3: Distribution of per-trade R-multiples for the Transformer. The right-skewed tail reflects the 3:1 take-profit-to-stop-loss ratio inherent in the triple-barrier labelling.*

![Figure 4: Backtest Return Distribution — BiLSTM.](plots/returns_hist_bilstm_backtest.pdf)
*Figure 4: Distribution of per-trade R-multiples for the BiLSTM. The distribution exhibits similar asymmetry but with a slightly higher concentration of trades near the -1R stop-loss boundary.*

In Figures 3 and 4, the return distributions for both models display the characteristic asymmetry expected from the 3:1 reward-to-risk barrier structure. Winning trades cluster near +3R (the take-profit boundary) while losing trades cluster near -1R (the stop-loss). The Transformer distribution shows a slightly higher density in the +3R region relative to the -1R region, consistent with its higher profit factor of 1.65 versus 1.36. This reinforces the conclusion that both models successfully learned to exploit the asymmetric payoffs of the triple-barrier framework during the backtest period.

### Rolling Sharpe Ratio

The rolling Sharpe ratio in Figure 5 provides a diagnostic of regime-specific alpha stability. The Transformer maintains a consistently positive rolling Sharpe, indicating high robustness to the moderate volatility shifts in the backtest window. The BiLSTM's dip below zero reflects its sensitivity to cyclical drift, where its recurrent state may be "over-memorizing" transient regimes that the Transformer's attention mechanism avoids. This reinforces the conclusion that the Transformer maintains a more stable risk-adjusted signal during the out-of-sample simulation.

### Statistical Significance (Monte Carlo Permutation)

To assess whether the observed performance is statistically distinguishable from random trading, 1,000 random permutations of each model's return sequence were generated.

![Figure 6: Monte Carlo Sharpe Distribution — Transformer.](plots/backtest_monte_carlo_sharpe_transformer.pdf)
*Figure 6: Monte Carlo null distribution of Sharpe Ratio for the Transformer. The observed Sharpe (0.813) falls significantly above the null mean.*

![Figure 7: Monte Carlo Sharpe Distribution — BiLSTM.](plots/backtest_monte_carlo_sharpe_bilstm.pdf)
*Figure 7: Monte Carlo null distribution of Sharpe Ratio for the BiLSTM. The observed Sharpe (0.492) also exceeds the null distribution.*

The Monte Carlo simulations in Figures 6 and 7 strongly indicate that the performance of both models is statistically distinguishable from random trading (p < 0.05). The Transformer’s Sharpe ratio is separated from the null mean by multiple standard deviations, providing robust evidence against the null hypothesis that these strategies could be replicated through random noise.

## 4.2.3 Classification Performance

While the primary evaluation metric for this study is financial performance (Sharpe Ratio, Profit Factor), classification metrics provide additional diagnostic insight into how each architecture distributes its predictions across the three output classes.

*Table 3: Per-Class Classification Metrics on the Out-of-Sample Backtest Set.*

| Class | Metric | Transformer | BiLSTM |
| :--- | :--- | :--- | :--- |
| **Hold** | Precision | 0.502 | 0.584 |
| | Recall | 0.552 | 0.213 |
| | F1-Score | 0.526 | 0.312 |
| **Long** | Precision | 0.414 | 0.409 |
| | Recall | 0.458 | 0.570 |
| | F1-Score | 0.435 | 0.476 |
| **Short** | Precision | 0.435 | 0.321 |
| | Recall | 0.268 | 0.631 |
| | F1-Score | 0.332 | 0.425 |
| **Overall** | Accuracy | 0.465 | 0.405 |
| | Macro F1 | 0.431 | 0.405 |

In Table 3, several instructive patterns emerge. The Transformer achieves higher overall accuracy (46.5% vs 40.5%) and is more conservative in its trade predictions — it classifies 753 of 1,374 samples as Hold compared to only 250 for the BiLSTM. This selectivity is consistent with its higher backtest Sharpe: by abstaining from marginal opportunities, it concentrates its trades on higher-conviction signals. Conversely, the BiLSTM predicts trades far more aggressively (1,124 trade signals vs 621 for the Transformer), achieving notably higher recall for both Long (0.570) and Short (0.631) classes at the cost of lower precision.

Crucially, these classification accuracies — both below 50% — might appear poor in a standard machine learning context. However, under the 3:1 reward-to-risk framework, a model needs only ~25% directional accuracy to break even. The combination of 47.8% win rate with 3:1 payoff asymmetry is what produces the Transformer's 0.813 Sharpe, demonstrating that classification accuracy alone is an insufficient proxy for trading system profitability.

## 4.2.4 Live Forward Testing (MT5 Execution)

The live forward test is integrated not as a tool for statistical inference, but rather to evaluate execution robustness under real market conditions. It is critical to acknowledge that the 35-day live period (February 24 – March 31, 2026) yielded an extremely small sample size (31 trades for the Transformer and 19 for the BiLSTM). Consequently, these results must be interpreted primarily as indicative of execution dynamics rather than statistically robust performance estimates.

Both architectures were deployed in parallel to a dedicated VPS connected to IC Markets (EU) demo accounts. Each system was allocated $10,000 USD initial capital with 0.5% risk per trade.

*Table 4: Live Forward Testing Performance Metrics (March 2026).*

| Metric | Transformer | BiLSTM |
| :--- | :--- | :--- |
| Total R-multiples | +3.32 | +4.68 |
| Trades (n) | 31 | 19 |
| Win Rate | 48.4% | 63.2% |
| Profit Factor | 1.59 | 3.50 |
| Actual PnL (%) | -1.07% | +0.10% |

In Table 4, which follows the same metric-per-row format as Table 2 for direct comparability, the live results reveal a striking reversal from the backtested rankings. The BiLSTM, which ranked second in backtesting (0.492 vs 0.813 Sharpe), demonstrates notably superior live performance: a 63.2% win rate with a profit factor of 3.50, compared to the Transformer's 48.4% win rate and 1.59 profit factor.

Sharpe ratios and maximum drawdown percentages are omitted from Table 4; as previously noted, the limited sample size renders such statistics highly sensitive to individual trade outcomes. Profit Factor and Win Rate are presented as more appropriate measures of execution behavior for this window.

The Transformer's higher trade frequency (31 vs 19 trades) exposes it to proportionally greater execution friction. This higher trade frequency is consistent with the Transformer's broader classification recall (0.458 for Long) observed in Table 3. Although it generated more R-units in absolute terms (+3.32 R from 31 trades vs +4.68 from 19 trades for the BiLSTM), the increased friction from more frequent position entry and exit resulted in a net account loss of -1.07% after broker costs.

### Live Equity Curves

![Figure 8: Live Equity Curve — Transformer vs BiLSTM.](plots/live_equity_curve.pdf)
*Figure 8: Cumulative equity curves from actual MT5 broker executions during March 2026.*

The equity curves in Figure 8 illustrate the divergence between architectures under live market conditions. While both systems benefited from strong directional moves on March 9-10, the Transformer's more frequent trading subsequent to this period generated alternating losses that eroded its early gains. This supports the hypothesis that higher signal sensitivity converts into a net liability during extreme regime shifts, as the BiLSTM's higher threshold preserved capital by simply omitting marginal entries.

### Live Drawdown Profiles

![Figure 9: Live Drawdown Curve — Transformer vs BiLSTM.](plots/live_drawdown_curve.pdf)
*Figure 9: Drawdown profiles from live execution. Both architectures exhibit drawdown periods during the volatile March 2026 regime.*

This visually confirms that while both models entered significant peak-to-trough declines during the regime shift, the BiLSTM maintained a shallower profile, consistent with its more conservative trade frequency.

### MT5 Actual Execution Traces

![Figure 10: Actual MT5 Cumulative PnL — Transformer.](plots/mt5_actual_transformer.pdf)
*Figure 10: Actual MT5 cumulative net PnL for the Transformer, tracking official broker execution records.*

![Figure 11: Actual MT5 Cumulative PnL — BiLSTM.](plots/mt5_actual_bilstm.pdf)
*Figure 11: Actual MT5 cumulative net PnL for the BiLSTM, tracking official broker execution records.*

In Figures 10 and 11, the actual broker execution traces show the trade-by-trade account evolution as recorded by the MetaTrader 5 terminal. These plots confirm that the BiLSTM maintained positive cumulative equity throughout the live period, finishing with a marginal gain, while the Transformer's account peaked early and subsequently declined below its starting capital. This provides the primary empirical evidence for the BiLSTM's superior real-world robustness.

### Sequential Trade Analysis

![Figure 12: Sequential Cumulative PnL — Live Comparison.](plots/live_trade_sequence_pnl.pdf)
*Figure 12: Trade-by-trade sequential cumulative PnL comparison between architectures in live conditions.*

In Figure 12, the trade-by-trade sequential comparison illustrates the divergence point between the two architectures. Both models track closely for the first 10 trades, after which the BiLSTM establishes a consistent lead. The Transformer's more aggressive trading frequency generates a higher variance trajectory, with multiple consecutive losses in the mid-March period that the BiLSTM avoids by simply not trading.

### Live Return Distributions

![Figure 13: Live Return Distribution — Transformer.](plots/returns_hist_transformer_live.pdf)
*Figure 13: Distribution of per-trade R-multiples for the Transformer during live forward testing (March 2026).*

![Figure 14: Live Return Distribution — BiLSTM.](plots/returns_hist_bilstm_live.pdf)
*Figure 14: Distribution of per-trade R-multiples for the BiLSTM during live forward testing (March 2026).*

The live return distributions in Figures 13 and 14 support the hypothesis that extreme regime shifts can fundamentally break reward-to-risk assumptions. Most notably, the characteristic +3R cluster visible in backtesting is entirely absent under live conditions. As volatility expands, ATR-scaled barriers may reach levels that are statistically unreachable within an intraday window, forcing profitable trades into early session-end exits. Consequently, winning trades yielded smaller, variable gains that failed to offset the systematic friction of frequent execution.

## 4.2.5 Regime Sensitivity Analysis

To evaluate the relationship between market volatility and model performance, trade outcomes were plotted against the normalised ATR at the time of entry.

![Figure 15: Volatility vs. Performance — Transformer.](plots/volatility_overlay_transformer.pdf)
*Figure 15: Volatility (ATR %) vs. trade return for the Transformer.*

![Figure 16: Volatility vs. Performance — BiLSTM.](plots/volatility_overlay_bilstm.pdf)
*Figure 16: Volatility (ATR %) vs. trade return for the BiLSTM.*

In Figures 15 and 16, the scatter plots with fitted trendlines suggest a positive correlation between normalised volatility and model return for both architectures. This provides preliminary evidence consistent with the hypothesis that these models capture predictive signal specifically from elevated volatility regimes — the ATR-filtered environment they were designed to exploit. The relationship appears slightly stronger for the Transformer, which is consistent with its higher overall Sharpe: the self-attention mechanism may be better equipped to recognise structural patterns within high-volatility candles.

## 4.2.6 Execution Fidelity Analysis

To quantify the impact of real-world market friction, we compare the theoretical signals (backtest engine replay of the March 2026 period) against actual MT5 broker outcomes.

*Table 5: Theoretical vs. Actual Execution Gap (March 2026).*

| Metric | Transformer | BiLSTM |
| :--- | :--- | :--- |
| Theoretical PnL (%) | +3.11% | +4.18% |
| Actual MT5 PnL (%) | -1.07% | +0.10% |
| **Execution Gap** | **-4.18%** | **-4.08%** |

In Table 5, both architectures exhibit a consistent execution gap of approximately 4%. This degradation is attributable to the difference between the idealised transaction cost model used in backtesting (fixed 2.0 points) and the variable real-world execution dynamics: widened spreads during volatile opens, partial fills, and order queue positioning.

The consistency of this ~4% execution gap suggests the presence of a structural lower bound imposed by market microstructure. This provides evidence that predictive modelling enhancements alone are insufficient to overcome execution-layer constraints. While per-trade slippage is structurally similar across both models, the Transformer's Final PnL is more heavily impacted due to its increased trade count.

![Figure 17: Execution Fidelity — Transformer.](plots/execution_fidelity_transformer.pdf)
*Figure 17: Execution fidelity analysis for the Transformer — theoretical signals (blue) vs. actual MT5 execution (red).*

![Figure 18: Execution Fidelity — BiLSTM.](plots/execution_fidelity_bilstm.pdf)
*Figure 18: Execution fidelity analysis for the BiLSTM — theoretical signals (blue) vs. actual MT5 execution (red).*

In Figures 17 and 18, the divergence between theoretical (blue) and actual (red) equity trajectories is clearly visible. For both models, the theoretical and actual curves track closely for the first few trades before gradually separating. The BiLSTM's smaller trade count results in fewer friction events, enabling it to maintain positive territory despite the systematic execution gap.

## 4.2.7 Backtest vs. Live Comparison

The following overlay plots directly compare the backtest equity and drawdown profiles with the live forward test trajectories.

![Figure 19: Equity Comparison — Transformer (Backtest vs. Live).](plots/comparison_transformer.pdf)
*Figure 19: Transformer cumulative equity — backtest simulation (blue) overlaid with live MT5 execution (red).*

![Figure 20: Equity Comparison — BiLSTM (Backtest vs. Live).](plots/comparison_bilstm.pdf)
*Figure 20: BiLSTM cumulative equity — backtest simulation (blue) overlaid with live MT5 execution (red).*

In Figures 19 and 20, the overlay visualisations illustrate the performance continuity across the backtest-to-live transition. The backtest equity curves (blue) demonstrate sustained growth across the August 2025 – February 2026 window, while the live segments (red) cover the shorter March 2026 forward testing period. The transition point (February 24, 2026) is visible as the junction between the two curves. The BiLSTM shows smoother continuity at this transition, while the Transformer's live trajectory diverges more sharply from its backtest trend. This reinforces the core finding that theoretical backtest performance is an insufficient predictor of live execution stability.

![Figure 21: Drawdown Comparison — Transformer.](plots/drawdown_comparison_transformer.pdf)
*Figure 21: Transformer drawdown comparison — backtest (blue) vs. live (red).*

![Figure 22: Drawdown Comparison — BiLSTM.](plots/drawdown_comparison_bilstm.pdf)
*Figure 22: BiLSTM drawdown comparison — backtest (blue) vs. live (red).*

In Figures 21 and 22, the drawdown comparison shows that live-period drawdowns remain generally bounded within the same envelope established during backtesting, suggesting no fundamental regime breakdown. The live drawdowns tend to be noisier due to the smaller sample size.

### The Robustness Gap: Frequency vs. Return

To synthesize the divergence between environments, Figure 23 illustrates the relationship between trade frequency and net profitability across the transition from backtest to live execution.

![Figure 23: The Robustness Gap — Trade Frequency vs. Net Return.](results/plots/robustness_gap_analysis.pdf)
*Figure 23: Comparison of total trade count versus net account return. The 'Robustness Gap' is visible in the Transformer's (blue) sharp downward shift as its higher signal sensitivity in backtesting converts into excessive friction-cost exposure during live execution.*

## 4.2.8 Summary of Findings

The primary contribution of this work is the empirical demonstration that optimizing trading models for offline Sharpe Ratio induces a bias toward high trade frequency, which leads to performance degradation under real-world execution constraints. The experimental results present evidence suggesting a fundamental trade-off:

1. **Theoretical Superiority vs. Practical Fragility**: While the Transformer achieves a 65% higher backtest Sharpe (0.813 vs 0.492), its superior pattern recognition leads to a "sensitivity bias." By capturing marginal signals that later fail under live execution friction, its theoretical edge converts into a net loss (-1.07%). The BiLSTM, by missing these marginal trades, maintains superior resilience (+0.10%).

2. **Structural Execution Constraints**: Both architectures exhibit a consistent ~4% execution gap relative to backtest expectations. This suggests that the performance degradation is a structural property of the NAS100 market microstructure, rather than a failure of model logic.

3. **Regime Sensitivity**: The mass failure of 3:1 take-profit barriers during live deployment provides evidence that the triple-barrier framework's profitability is highly dependent on volatility regime stability.

The transition from backtest to live results suggests a fundamental mismatch between the training distribution and the live environment. While keeping in mind the statistical limitations of the forward test, the observed mechanism of degradation remains instructive:

### Mechanism: Volatility Regime Expansion

The training dataset (2019–2025) is predominantly composed of low-to-moderate volatility regimes (VXN < 25). In March 2026, however, geopolitical instability drove volatility significantly above the training distribution. This shift fundamentally altered the ATR scaling used for barrier placement.

### Evidence: Take-Profit Barrier Failure

The most revealing sign of this mismatch is the **zero hit-rate** of take-profit targets during live trading. In an extreme regime, the 3.0 × ATR barrier expands to distances (400+ points) that are statistically unlikely to be traversed within the 2-hour session window. Consequently, 100% of winning trades were forced to close via the session-end auto-exit, yielding smaller gains that could not offset the frequency-driven friction.

The BiLSTM's relative live stability suggests that its recurrent structure provides implicit temporal smoothing, causing it to ignore the "weak" signals that the Transformer's high sensitivity triggers upon. In a noise-dominated regime, trading less frequently is inherently advantageous. This indicates that for financial ML, **robustness is often inversely proportional to signal sensitivity**, and Sharpe-optimised models may be systematically incentivised to capture marginal signals as variance scales sublinearly ($\propto \sqrt{n}$).

### The Upper ATR Filter Dilemma

A natural mitigation strategy would be to introduce an upper ATR filter — complementing the existing minimum ATR threshold of 15 points with a maximum threshold that prevents trading when volatility exceeds a defined ceiling. In principle, such a filter would have excluded the most extreme days of March 2026 and prevented the majority of losing trades. In practice, however, applying a reasonable upper ATR filter to the live period would have eliminated nearly all trading opportunities, resulting in zero or near-zero live trades. This presents a methodological tension: while the filter would improve simulated performance, it would simultaneously eliminate the empirical evidence needed to evaluate live execution behaviour. The decision to trade without an upper filter was therefore made deliberately to ensure that the live forward test produced a meaningful, if imperfect, dataset for analysis.

### Noise Robustness Considerations

As described in the Methodology (Section 3.4), Gaussian noise ($\sigma_{\text{noise}} = 0.01$) was injected into input features during training as a data augmentation technique to quantify model sensitivity to input perturbations. While this noise injection improves generalisation to minor distributional shifts — such as broker-specific pricing differences or rounding artefacts — the March 2026 regime shift represents a fundamentally different challenge: not noise in the input features, but a structural change in the underlying data-generating process itself. The noise robustness mechanism, calibrated to $\sigma = 0.01$, was insufficient to prepare the models for the order-of-magnitude volatility increase observed during the live period. This suggests that future work should explore regime-aware augmentation strategies — for instance, synthetically generating extreme-volatility training samples using historical VXN spike periods — rather than relying on isotropic Gaussian perturbation alone.

The observations above suggest that a key limitation of both architectures is their inability to distinguish between genuine directional signals and noise-driven false breakouts in extreme volatility regimes. Future research could explore State Space Models (SSMs) such as Mamba (Gu and Dao, 2023) or S4 (Gu et al., 2022) as an alternative architectural paradigm. SSMs operate by learning a continuous-time latent state representation that explicitly separates signal dynamics from observation noise through a structured state transition matrix. This inductive bias may hypothetically confer advantages in super-volatile regimes where the primary challenge is not pattern recognition — which both the Transformer and BiLSTM handle well — but rather the filtering of model predictions through a noise-aware lens. Additionally, SSMs theoretically achieve linear-time complexity in sequence length, potentially enabling longer lookback windows that capture broader regime context without the quadratic cost of self-attention.
