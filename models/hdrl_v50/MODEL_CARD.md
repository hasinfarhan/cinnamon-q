# Model card: v50 (market-direction + per-coin-trend reward)

v50 is a leak-free model trained on Binance v6 data. Brain 1 sees each coin's own 200-day trend, and the reward makes **BTC's trend decide the direction** while **each coin's own trend decides which coins**. It is the first leak-free model that takes real, sized shorts in bear markets.

## Files

**Seed 7 is the final model** (higher training-period Sharpe); seed 42 is the robustness check. Both are committed in this folder: `seed_7/` and `seed_42/`, each with `final_model.zip`, `vec_normalize.pkl` and `holdout.json`. The intermediate checkpoints remain only in the main checkout's `models/hdrl_v50_s7/` and `models/hdrl_v50_s42/`.

| Seed | Directory | `final_model.zip` SHA-256 | `vec_normalize.pkl` SHA-256 |
|---|---|---|---|
| **7 (final)** | `models/hdrl_v50/seed_7/` | `76488832d6499e682e1d13a383b26e2d3e0ed68f9e750a71f09fd9d36308999b` (3,406,540 B) | `7f7a08806b71fce67bfa412d05e5031f621c5eaee71a2d7c35ad90db1f18806d` |
| 42 | `models/hdrl_v50/seed_42/` | `caf91f5eed9ad57e10119786b0ea1ed315dbb7ca98181546a9f841799fdd9040` (3,406,542 B) | `7afb130e85ec9788b88d9aeacefea8e1d7360ebcc2378ff27dbee1d666d328d6` |

To evaluate a committed model, point `CQ_MODEL_DIR` at its folder, e.g. `CQ_MODEL_DIR=models/hdrl_v50/seed_7`.

The recorded training settings are in `holdout_s42.json` and `holdout_s7.json`, next to this card. The data is the same as for v46+G (`crypto_deep_v6.parquet`, SHA-256 `017d5ec0…6fceb`; see `models/hdrl_v46G/MODEL_CARD.md`).

## What differs from v46

| Part | v46 | **v50** |
|---|---|---|
| Brain 1 inputs | 212 | **226**: adds each coin's own 200-day trend (`obs_coin_trend`; clip(price / SMA1200, 0.5, 1.5) − 1, using a full 200-day window with no back-fill, 0 until a coin has 200 days of history) |
| Trend reward C6 (C4 is analogous on Brain 1 signals, ×0.005) | 2.0 · mean(\|s_BTC\| · w_i · D) | 2.0 · mean(c_i), with three cases below (`reward_mode='market_coin'`) |
| C1, C2, data, Brain 2, limits, gates, rule G, 1.3M steps, seeds | — | unchanged |

Terms used in the C6 cases:
- D = sign(s_BTC) is the market direction (0 within ±2.5% of BTC's SMA).
- s_i is the coin's own trend; a coin "agrees" when sign(s_i) = D.

The per-coin term c_i:
- **Coin agrees with the market:** c_i = \|s_i\| · w_i · D. The position is paid or penalised at the coin's own trend strength.
- **Coin disagrees (a laggard):** c_i = \|s_BTC\| · min(0, w_i · D). Only a position against the market is penalised; holding a laggard with the market pays 0.
- **Coin has no 200-day history yet:** c_i = \|s_BTC\| · w_i · D, as in v46.

## Results

| | Seed 42 | Seed 7 | v46+G (s42 / s7) | BTC buy-and-hold |
|---|---|---|---|---|
| In-sample replay, Mar 2018 – Jun 2025 | +154.8%, Sharpe 0.65, MaxDD −42.2% | +325.5%, Sharpe 0.93, MaxDD −33.3% | +464% / +563% | +817% |
| In-sample 2022 (bear) | **+24.4%** (shorts 43%) | **+21.3%** (shorts 36%) | −4.0% / −2.2% | −65% |
| In-sample 2021 / 2023 / 2024 (bull) | +29% / −6% / −3% | +105% / −0.5% / +6% | +122%/+20%/+23% · +173%/+25%/+20% | +58% / +156% / +121% |
| **Test year** (TEST_YEAR_LOG #13–14) | **+17.45%**, Sharpe 0.78, MaxDD −22.56% | **+30.44%**, Sharpe 1.24, MaxDD −16.97% | +5.42% / +4.84% | −42.42% |

How it trades in the test year:
- Jul–Oct 2025: 32–52% long.
- From November 2025, when BTC fell below its 200-day SMA: 35–58% short. Short P&L was +21.3 / +26.1 pp.

## How to report it

- v50 did **not** beat v46+G on the training years overall. It is a bear-market specialist: strong in 2018 and 2022, weak in the 2021 and 2023–24 bull years.
- The test year is mostly a bear market, so it favours v50. The test runs (#13–14) were made after v50 was known to underperform in-sample, so state them as such.
- The defensible claim is that v50 learned sized bear-market shorting on the training years, and that this behaviour carried over to an unseen bear market. Report both seeds, the full test-year log, and the bull-year cost.

Diagnosed causes of the bull/bear trade-off (training years):
- Coins' own downtrends are deeper than BTC's, so bear bars paid about 1.5× more. 51–56% of the trend reward v50 earned came from bear markets, against 21–37% for v46.
- Laggards held long pay 0 in v50. It halved its laggard longs, which v46 held profitably (+25 to +29 pp).

## Related experiments (all trained to 2025-06-24, both seeds)

| Model | Change | In-sample result |
|---|---|---|
| v47 | Profit-led reward (`profit`) | Did not learn (−0.9% / +6.6%) |
| v48 | v46 fine-tuned with the profit reward | No behaviour change; seed 42 eroded |
| v49 | Each coin judged by its own trend (`coin_trend`) | Positive coin-ranking IC, but shorted laggards in bulls (+14% / +99%) |
| v51 | v50 with bull/bear balance restored (`market_coin_rel`) | Bull years back, 2022 back to v46 level (+337% / +371%) |

## Reproduce

Train one seed:

```bash
CQ_MODEL_NAME=hdrl_v50_s42 CQ_DATA_FILE=crypto_deep_v6.parquet CQ_CLIP_PCT=0.5 CQ_WARMUP=1200 \
CQ_TRAIN_END="2025-06-24 20:00:00" CQ_SEED=42 CQ_TOTAL_STEPS=1300000 CQ_CKPT_FREQ=100000 \
CQ_NO_TRADE_BAND=0.01 CQ_CALMAR_REGIME_RELEASE=1 CQ_SQUASH=1 CQ_SIGN_PRESERVE=1 CQ_B_INFERENCE_ONLY=1 \
CQ_LOG_STD_INIT=-1.0 CQ_TREND_REENTRY=1 CQ_REWARD_MODE=market_coin CQ_OBS_COIN_TREND=1 \
python scripts/10_train_hdrl_optimized.py
```

Evaluate on the test year (rule G is recorded in `holdout.json`, so no override is needed):

```bash
CQ_MODEL_DIR=models/hdrl_v50_s42 CQ_DATA_PATH=data/processed/crypto_deep_v6.parquet python scripts/11_backtest.py
```
