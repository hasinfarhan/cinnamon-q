# Hierarchical Reinforcement Learning for Cryptocurrency Portfolio Allocation

Code, trained models and results for the master's thesis

> **Adaptive Rationality in Non-Stationary Markets: A Hierarchical Reinforcement Learning Approach with Constrained Optimization for Cryptocurrency Portfolio Allocation**
> Kashfia Sarony, University of Warsaw, Faculty of Economic Sciences (Data Science and Business Analytics), 2026.
> Supervisor: dr Marcin Gruszczyński.

The final model is **v50**. Seed 7 is the reported model; seed 42 is a second, independently trained instance of the same specification.

## Architecture

The system separates learned prediction from rule-based risk control:

1. **Signal generation (learned).** A recurrent PPO agent (`sb3-contrib` RecurrentPPO, single-layer LSTM with 128 units, 128x128 MLP heads) reads a 226-value observation every four hours and outputs one conviction score per asset in (-1, 1).
2. **Portfolio construction (analytical).** Scores are converted into weights with the inverse of a 20-day downside semicovariance matrix (10% diagonal shrinkage). Positions whose weight would contradict the sign of their score are dropped; initial gross exposure equals 80% of capital times the mean absolute conviction, and three hard limits can only reduce it: 80% gross long, 60% gross short, 10% per asset. A 1% no-trade band suppresses small rebalancing trades.
3. **Risk gates (rules).** A drawdown circuit breaker moves the book to cash for one day after a 15% decline from its reference peak and then allows re-entry only on the side of Bitcoin's 200-day trend; a macro gate scales positions down when the VIX-based macro score signals panic; a regime-uncertainty gate holds cash while Bitcoin trades within 2.5% of its 200-day average.

Inputs: 14 Binance spot assets (AAVE, AVAX, BAT, BCH, BTC, CRV, ETH, GRT, LINK, LTC, SOL, SUSHI, UNI, YFI) on four-hour bars; technical indicators, multi-horizon returns, correlation with Bitcoin and EWMA volatility per asset; the Crypto Fear & Greed Index; VIX, DXY and the 10-year Treasury yield; Bitcoin's and each coin's own 200-day trend. All daily inputs are lagged by one day.

Costs: 0.1% of traded notional on every rebalancing and 3% per annum on short positions.

## Results

### Held-out test year (25 Jun 2025 to 25 Jun 2026, 2,190 four-hour returns)

The model was trained only on data up to 24 June 2025, 20:00 UTC. Scaling ranges are fitted on the training period.

| Strategy | Total return % | Ann. return % | Sharpe | Sortino | Max DD % | Calmar | Win rate % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v50 (seed 7)** | **+30.44** | **+30.47** | **1.242** | **1.788** | **-16.97** | **1.795** | 48.4 |
| v50 (seed 42) | +17.45 | +17.46 | 0.778 | 1.117 | -22.56 | 0.774 | 47.5 |
| Buy-and-hold BTC | -42.77 | -42.79 | -1.138 | -1.539 | -52.19 | -0.820 | 49.9 |
| Equal weight (14 assets) | -54.81 | -54.84 | -0.924 | -1.232 | -70.69 | -0.776 | 50.3 |

Benchmarks are computed on the same bar-aligned returns without transaction costs; all costs are charged to the model.

- The model was net long (about 40 to 50% of capital on monthly average) while Bitcoin traded above its 200-day average in July to September 2025, and net short (33 to 48%) from November 2025 when the market turned down. Long positions contributed +8.0 and short positions +26.1 percentage points.
- Moving-block bootstrap (blocks of 5, 10 and 20 trading days, 5,000 resamples, same blocks for model and BTC): the mean advantage over BTC is +66 to +70 pp of total return, +2.2 to +2.3 of annualized Sharpe and +3.0 to +3.2 of annualized Sortino ratio, and 92 to 94% of resampled paths favour the model on every metric.

Outputs of `scripts/11_backtest.py` for both seeds are in [`reports/`](reports/).

### Training period (Mar 2018 to Jun 2025, in-sample)

These figures describe the risk profile of the architecture over a full market cycle (two bear markets, several bull markets, the March 2020 crash); they are not a test of predictive ability.

| Strategy | Total return % | Ann. return % | Sharpe | Sortino | Max DD % | Calmar | Win rate % |
|---|---:|---:|---:|---:|---:|---:|---:|
| v50 (seed 7) | +325.47 | +21.93 | 0.927 | 1.310 | -33.26 | 0.659 | 47.4 |
| Buy-and-hold BTC | +817.03 | +35.45 | 0.795 | 1.132 | -77.04 | 0.460 | 51.3 |
| Equal weight (14 assets) | +282.75 | +20.18 | 0.649 | 0.890 | -85.39 | 0.236 | 52.4 |
| 50% BTC + 50% v50 | +841.71 | +35.92 | 1.049 | 1.483 | -44.89 | 0.800 | 51.6 |

The model exceeds Bitcoin on every risk-adjusted measure with less than half of its drawdown. Its four-hour returns are almost uncorrelated with Bitcoin's over the training period (+0.07) and negatively correlated over the test year (-0.42), so it works best as a complement to a market position: the 50/50 combination earned more than Bitcoin alone in-sample and lost 10.2% in the test year while Bitcoin lost 42.8%. In strong bull markets interrupted by corrections the strategy participates less than a long-only holding.

### Selection adjustment

Several configurations were evaluated on the test window during development. Counting the eleven evaluations made with the final data pipeline, the Deflated Sharpe Ratio (Bailey and López de Prado, 2014) of the final model is **0.58** (expected maximum Sharpe ratio of eleven trials by chance: about 1.05); counting nineteen evaluations, including discarded earlier development versions, it is 0.26. Before adjustment, the probabilistic Sharpe ratio (probability that the true Sharpe ratio is positive) is 0.89. The return series has a skewness of -0.36 and a kurtosis of 12.1.

## Reproducing

Python 3.12. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

**Evaluate the committed models** (requires the dataset, see below):

```bash
python scripts/11_backtest.py
python scripts/11_backtest.py --model models/v50/seed_42
```

Seed 7 ends the test year at $130,444.12 from $100,000.

**Build the dataset** (about 270 MB, not stored in the repository):

```bash
python scripts/01_fetch_prices.py
python scripts/02_fetch_fear_greed.py
python scripts/03_build_dataset.py
```

The build writes `data/processed/crypto_deep_v6.parquet` and stops if any data check fails (missing values, frozen prices, spike-and-reverse glitches). Macro series are downloaded from Yahoo Finance during the build.

**Train** (1.3 million steps, CPU, several hours per seed):

```bash
python scripts/10_train.py --seed 7
python scripts/10_train.py --seed 42
```

## Repository layout

| Path | Content |
|---|---|
| `src/hdrl/data.py` | Four-hour bars, input scaling (fitted on training rows only), listing mask, trend inputs |
| `src/hdrl/env.py` | Trading environment: portfolio construction, risk gates, costs, training reward |
| `scripts/01_fetch_prices.py` | Binance 15-minute candles |
| `scripts/02_fetch_fear_greed.py` | Crypto Fear & Greed Index |
| `scripts/03_build_dataset.py`, `scripts/feature_functions.py` | Dataset build and indicators |
| `scripts/10_train.py` | Training with the train/test split |
| `scripts/11_backtest.py` | Test-year backtest, benchmarks and metrics |
| `models/v50/` | Trained models (seed 7 final, seed 42) and the [model card](models/v50/MODEL_CARD.md) |
| `reports/` | Backtest outputs for both seeds |

## License

Apache 2.0, see [LICENSE](LICENSE).
