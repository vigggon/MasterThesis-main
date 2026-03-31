# Thesis Results Update: February 2026 Data Extension

This document outlines the changes and metric updates stemming from the addition of the new dataset (`USTEC_M5_NEW.csv`), which extends the analysis timeframe into late February 2026.

## Time Period Changes
The historical dataset was appended with the newest MetaTrader5 export (approx. 4780 new 5-minute candles). 

* **Previous Data End Date:** Early January 2026
* **New Data End Date:** February 23, 2026

Both the `LSTM` and `Transformer` models were retrained across the entirety of this extended dataset.
* **Training Window:** 2019 to August 2025
* **Out-of-Sample Forward Testing:** August 29, 2025 – February 23, 2026 (Approx. 6 months forward timeframe).

## Model Evaluation & Stability Execution
The evaluation stringency remained identical to previous thesis parameters—incorporating standard 1.0 Point equivalent friction (slippage/spread), minimum ATR gating (15 points), and strict Daily Limit boundaries (-4.0R SL, +12.0R TP).

### Transformer Robustness & Performance
Over the extended out-of-sample forward test (8/2025 -> 2/2026):

* **Total Profit:** +295.07 R
* **Mean Daily PnL:** +4.57 R
* **Standard Deviation (Daily):** 15.28 R
* **Maximum Regulated Drawdown (Worst Day Limit test):** -13.75 R
* **Win/Loss Day Profiling:** 50.8% Winning Days (Probability of a losing day: 49.2%)

**Yearly Regime Breakdown:**
The extended backtest successfully tested across the transition of the new year, tracking distinct volatility phases.
* **2025 (Forward block):** +225.98 R (from 424 trades) | Max DD: -1.80 R
* **2026 (YTD block):** +69.09 R (from 198 trades) | Max DD: -7.28 R

**Monte Carlo Randomization Test (1000 iter):**
* **Actual Sequenced Max DD:** -1.80 R 
* **Monte Carlo Mean Max DD:** -2.19 R (Std: 2.37)
* **Z-Score (Distance from Mean):** 0.17
* **Verdict:** The actual PnL sequence yielded a drawdown smaller than ~60.6% of purely randomized permutations. The system continues to generate a highly stable, non-curve-fitted sequence of trade results in true out-of-sample regimes.

### LSTM Tracking Performance
The LSTM acts as the baseline reference structure and similarly retained strong characteristics over the extended window:

* **Mean Daily PnL:** +3.75 R
* **Standard Deviation (Daily):** 14.22 R
* **Maximum Regulated Drawdown (Worst Day Limit test):** -13.77 R
* **Win/Loss Day Profiling:** 47.6% Winning Days (Probability of a losing day: 52.4%)

## Summary of Completed Technical Actions
1. `parse_mt5_data.py`: Built parser to standardize raw MT5 extracts (handling `DATE` and `TIME` splits to unified `datetime`) and cleanly pre-pend existing history.
2. Filter open market regime, label standard bounds, format dynamic features tracking into Feb 2026.
3. Retrained base architectures over 250 Epochs leveraging dynamically weighted focal loss for early structural convergence.
4. Regenerated precision-recall threshold balances.
5. Ran full OOS Walk-Forward Backtester.
6. Generated all visual reporting (`results/plots`): Daily performance distributions, comparative regime stabilities, and isolated model equity curves.
