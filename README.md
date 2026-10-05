# Binance Grid Optimizer

Research-only project to evaluate conservative, easily configurable improvements to Binance Spot Grid bots.

## Reference
- Baseline bot: Binance Grid Bot **9161957**
- Baseline is preserved as the control configuration.
- No live trading or automatic deployment is performed by this repository.

## Objective
Test small configuration changes that may improve ROI while preserving a low maximum drawdown (MDD) and robust performance.

Candidate controls:
- stop loss (SL)
- take profit (TP)
- trailing stop
- grid count
- lower/upper range
- investment
- optional trailing-up behavior

## Method
Every experiment is compared against the unchanged 9161957 baseline. A higher ROI alone is not sufficient: MDD, PNL, trade count, stability and robustness are also evaluated.

## Safety
This repository is for research/backtesting. Results are not guarantees of future performance.
