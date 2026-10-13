# Model card: v50

## Summary

v50 is the signal-generation layer of a hierarchical long/short allocation system for 14 cryptocurrencies on four-hour bars (see the [README](../../README.md) and thesis Chapters II and III). It is trained with recurrent PPO; portfolio construction and risk gates are analytical and rule-based and are implemented in `src/hdrl/env.py`.

| | |
|---|---|
| Final model | `seed_7/` |
| Robustness check | `seed_42/` (same specification, different seed) |
| Training data | 5 Mar 2018 to 24 Jun 2025, 20:00 UTC (episodes start after a 200-day warm-up from 17 Aug 2017) |
| Test window | 25 Jun 2025 to 25 Jun 2026 (2,191 four-hour bars), never seen in training |
| Dataset | `crypto_deep_v6.parquet`, built by `scripts/03_build_dataset.py` |

## Files

Each seed folder contains `final_model.zip` (policy weights), `vec_normalize.pkl` (observation scaling statistics, required for inference) and `holdout.json` (the train/test split and training settings).

| Seed | `final_model.zip` SHA-256 | `vec_normalize.pkl` SHA-256 |
|---|---|---|
| 7 (final) | `76488832d6499e682e1d13a383b26e2d3e0ed68f9e750a71f09fd9d36308999b` | `7f7a08806b71fce67bfa412d05e5031f621c5eaee71a2d7c35ad90db1f18806d` |
| 42 | `caf91f5eed9ad57e10119786b0ea1ed315dbb7ca98181546a9f841799fdd9040` | `7afb130e85ec9788b88d9aeacefea8e1d7360ebcc2378ff27dbee1d666d328d6` |

## Training

| Setting | Value |
|---|---|
| Algorithm | RecurrentPPO, `MlpLstmPolicy` (sb3-contrib 2.8.0) |
| Network | LSTM 128 units, 1 layer (actor); MLP 128x128 for actor and critic; ReLU |
| Steps | 1,300,000, 4 parallel environments, 512-bar rollouts |
| Learning rate, gamma, GAE lambda | 3e-5, 0.99, 0.95 |
| Batch size, epochs, clip range | 512, 4, 0.2 |
| Entropy coefficient, value coefficient, max grad norm | 0.015, 0.5, 0.5 |
| Initial exploration noise | log std -1.0 (std 0.37) |
| Observation / reward scaling | VecNormalize, observations clipped at 10 |
| Input scaling | per-asset ranges at the 0.5th and 99.5th percentiles of the training rows |
| Episodes | random start after the warm-up, at least 1,000 bars long |

Reward per step (thesis Section 3.3), with r the net portfolio return after costs:

- C1 = r, or 1.35 r when r < 0 and the portfolio is more than 0.5% below its peak (loss aversion);
- C2 = -0.005 x drawdown (drawdown drag);
- C6 = 2.0 x mean over listed coins of the trend term applied to portfolio weights, and C4 = 0.005 x the same mean applied to the conviction scores. Bitcoin's 200-day trend sets the market direction D; for a coin whose own 200-day trend agrees with D, a position x earns |s_coin| x D x x; for a coin whose trend disagrees, only a position against the market is penalised (|s_BTC| x min(x D, 0)); a coin without 200 days of history is scored on Bitcoin's trend.

The trend-side re-entry after the circuit breaker and the regime-uncertainty gate are active only at evaluation.

## Evaluation

Held-out test year (`python scripts/11_backtest.py`; benchmarks without transaction costs):

| | Seed 7 | Seed 42 | BTC buy-and-hold | Equal weight |
|---|---:|---:|---:|---:|
| Total return | +30.44% | +17.45% | -42.77% | -54.81% |
| Sharpe | 1.24 | 0.78 | -1.14 | -0.92 |
| Sortino | 1.79 | 1.12 | -1.54 | -1.23 |
| Max drawdown | -16.97% | -22.56% | -52.19% | -70.69% |
| Calmar | 1.80 | 0.77 | -0.82 | -0.78 |

Training period, in-sample (seed 7): +325.47%, Sharpe 0.93, Sortino 1.31, maximum drawdown -33.26%, against +817.03%, 0.80, 1.13 and -77.04% for Bitcoin.

Selection adjustment: several configurations were evaluated on the test window during development. Deflated Sharpe Ratio 0.58 counting the eleven evaluations made with the final data pipeline, 0.26 counting nineteen including discarded earlier versions; probabilistic Sharpe ratio before adjustment 0.89.

## Intended use and limitations

- Research artefact for the thesis; not investment advice and not a production trading system.
- The strategy targets return per unit of downside risk. In strong bull markets interrupted by corrections it participates less than a long-only holding, and it is best understood as a complement to a market position.
- The test window is a single, predominantly bearish year; the block bootstrap measures uncertainty within that path, not across independent bear markets.
- Gross exposure can reach 140% of capital (80% long plus 60% short). Short positions are simulated with a flat 3% annual borrowing cost; funding-rate variation, slippage and liquidity limits are not modelled.
- The developer-activity input is a single snapshot per coin and is constant over time.
