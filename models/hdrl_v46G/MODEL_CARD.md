# Model card: v46 + rule G (current candidate)

**v46 + G** = the trained v46 models (seeds 42 and 7), evaluated with the drawdown-gate re-entry rule **G**.
G is an evaluation-only rule: training is identical to v46, so the models were not retrained.

## Files (binaries are git-ignored; they live in the main checkout's `models/`)

| Seed | Directory | `final_model.zip` SHA-256 | `vec_normalize.pkl` SHA-256 |
|---|---|---|---|
| 42 | `models/hdrl_v46_s42/` | `37b07b1c398a6e9aea9ffbec2494aff4fdd38c2254a868119568e115c697c7bf` (3,297,936 B) | `4d57ae63ce419414059120f3fa07b0c53d5f8d6939ec7230c0f98c4d126a2185` |
| 7 | `models/hdrl_v46_s7/` | `a17a4b23d3f0bbba3acaffebe6a537dc6d110c249ab960a06e3f4962a3b6be57` (3,297,934 B) | `2a65df31b9afbb2a63d1aa5ce79c43322ac8677646ffa9b0997b52a0ced9f8eb` |

The recorded training settings are in `holdout_s42.json` and `holdout_s7.json`, next to this card.

**Data:** `data/processed/crypto_deep_v6.parquet`, SHA-256 `017d5ec0cc5136c1252461622b6338cf90cd3baed733e4dd7db735971de6fceb`.
It is built by `scripts/08_fetch_binance_v6.py` (raw prices) and `scripts/_build_v6.py` (features and data checks).

## What it is

| Part | Setting |
|---|---|
| Data (v6) | Binance spot USDT 15-min candles for 14 coins, from each coin's listing (BTC/ETH Aug 2017) to 2026-06-25; no price filling. Features are as in v5 except MACD (as % of price). Crypto Fear & Greed Index replaces per-coin news sentiment. Macro and Fear & Greed are lagged one day. |
| Listing mask | A coin is tradable 20 days after its first candle. Unlisted coins get no weight and no C4/C6 credit. |
| Loader | 4h bars. Min-max ranges are fitted on training rows only, and MACD, BB width and volatility use the 0.5–99.5 percentile range (`clip_pct=0.5`). |
| Brain 1 | RecurrentPPO, MlpLstmPolicy (LSTM 128, pi/vf [128,128]), tanh-squashed actions, `log_std_init=-1.0`, 1.3M steps, lr 3e-5 |
| Brain 2 | Sortino solver (20-day semicovariance), sign-preserving sizing. Limits: max 80% long, max 60% short, max 10% per coin. No-trade band 1%. |
| Reward | v40: C1 (losses ×1.35 in drawdown) + C2 (−0.005·DD) + C4 + C6 (2.0·\|s\|·mean(w·d)) |
| Gates | Calmar breaker at 15% drawdown with 1-day cooldown. **Rule G at evaluation:** re-entry is allowed only on the BTC 200-day-trend side until the trend flips. Gate 3 (cash within ±2.5% of the SMA, evaluation only). Macro panic gate. |
| Split | Train ≤ 2025-06-24 20:00 (episodes start ≥ 2018-03, after a 1,200-bar warm-up); test 2025-06-25 → 2026-06-25 (2,191 bars) |

## Results

| | Seed 42 | Seed 7 | BTC buy-and-hold |
|---|---|---|---|
| **Test year** (TEST_YEAR_LOG #11–12) | **+5.42%**, Sharpe 0.448, MaxDD −17.02% | **+4.84%**, Sharpe 0.382, MaxDD −15.39% | −42.42%, MaxDD −52.19% |
| In-sample replay, Mar 2018 – Jun 2025 (diagnostic only) | +463.7%, Sharpe 1.21, MaxDD −26.2% | +563.4%, Sharpe 1.18, MaxDD −31.6% | +817%, MaxDD −77% |

## Known weaknesses (from diagnostics)

- **Weak bearish conviction.** In 2022 the average signal strength is 0.10–0.20 and shorts are 2–10% of the portfolio, against a 60% cap.
- **Seed-specific stray coins.** Some coins stay long in bear markets (seed 42: AAVE, CRV, SUSHI; seed 7: AAVE, BAT, LINK, SOL).
- **Under-invested in bull markets** (net exposure about 0.35–0.40).
- **No coin-ranking skill** (IC ≈ 0).
- **Reward dominated by trend alignment** (C6 is about 85% of reward variance).
- **Rule G whipsaw at turning points** (Feb 2019: −9 pp for seed 42).

**Negative result, v47:** a profit-led reward (`reward_mode='profit'`) ranks strategies like money does (ρ 0.92), but it did not learn: in-sample returns were −0.9% / +6.6%, with a random hedged book.

## Reproduce

Train one seed:

```bash
CQ_MODEL_NAME=hdrl_v46_s42 CQ_DATA_FILE=crypto_deep_v6.parquet CQ_CLIP_PCT=0.5 CQ_WARMUP=1200 \
CQ_TRAIN_END="2025-06-24 20:00:00" CQ_SEED=42 CQ_TOTAL_STEPS=1300000 CQ_CKPT_FREQ=100000 \
CQ_NO_TRADE_BAND=0.01 CQ_CALMAR_REGIME_RELEASE=1 CQ_SQUASH=1 CQ_SIGN_PRESERVE=1 \
CQ_B_INFERENCE_ONLY=1 CQ_LOG_STD_INIT=-1.0 python scripts/10_train_hdrl_optimized.py
```

Evaluate on the test year with rule G:

```bash
CQ_MODEL_DIR=models/hdrl_v46_s42 CQ_DATA_PATH=data/processed/crypto_deep_v6.parquet \
CQ_ENV_OVERRIDE='{"calmar_trend_reentry": true}' python scripts/11_backtest.py
```

Regression anchors for the code: v45 seed 42 on `crypto_deep_v5_clean.parquet` returns $102,731.09 on the test year. v46 seed 42's in-sample replay (2018-03-05 → 2025-06-24) returns $459,371.00 without G and $563,727.75 with G.
