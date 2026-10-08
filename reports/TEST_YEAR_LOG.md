# Test-year evaluation log

Test year: **2025-06-25 → 2026-06-25** (2,190 four-hour bars). BTC buy-and-hold −42.53%, equal-weight (14 coins) −54.40%.

Every evaluation of any model on this window is listed here, so the number of looks at the test year can be reported honestly.

| # | Date (approx.) | Model | Trained on data up to | Purpose | Total return | Sharpe | Max DD | Notes |
|---|---|---|---|---|---|---|---|---|
| — | before 2026-06-26 | v35, v37, v38, v39, v41 | full v5 file (incl. test year) | Development iterations | −0.17%, −8.16%, −14.31%, −16.53%, −9.51% | — | — | Selection pool for v40 (6 same-window trials incl. v40) |
| 1 | 2026-06-26 | v40 | **full v5 file (incl. test year)** | Original thesis model | +13.54% | 0.706 | −18.56% | **Leaky**: trained on the test year; same-day macro/news; full-sample normalization |
| 2 | 2026-10-07 | v40 + lagged inputs | same as v40 | Leak attribution (diagnostic, not a candidate) | +11.85% | 0.632 | −19.79% | |
| 3 | 2026-10-07 | v40 + lagged inputs + train-only normalization | same as v40 | Leak attribution (diagnostic, not a candidate) | +8.04% | 0.475 | −22.19% | |
| 4 | 2026-10-07 | v43 | 2025-06-24 | v40 config retrained without leaks (seed 42) | −32.01% | −1.428 | −48.02% | Its test-year behaviour was later diagnosed; A+B was partly motivated by that diagnosis |
| 5 | 2026-10-07 | v44dev A+B, seed 42 | 2024-06-24 | A+B chosen on validation year (Jun 2024 – Jun 2025) | +3.41% | 0.260 | −19.94% | In cash 44% of steps (drawdown gate held cash from Feb 2026) |
| 6 | 2026-10-07 | v44dev A+B, seed 7 | 2024-06-24 | Same, second seed (both seeds reported) | +4.15% | 0.303 | −16.97% | In cash 65% of steps |
| 7 | 2026-10-07 | v45, seed 42 | 2025-06-24 | A+B + tanh squash + sign-preserving sizing + B inference-only + log_std_init −1; 1.3M steps; checked on training data only before this run | +2.73% | 0.259 | −15.34% | Never net short; Calmar gate fired 2025-11-13 and held cash to end of test year (61% of steps); fees 2.15 pp |
| 8 | 2026-10-07 | v45, seed 7 | 2025-06-24 | Same, second seed (both seeds reported) | +3.90% | 0.310 | −15.16% | Never net short; gate fired 2025-11-14, cash to end (61% of steps); fees 2.35 pp |
| 9 | 2026-10-08 | v46, seed 42 | 2025-06-24 | v45 settings retrained on **v6 data** (Binance prices from 2017, no filled prices, Fear & Greed instead of news sentiment, listing mask, MACD as % of price, 0.5–99.5% clipped scaling, episodes from 2018-03); checked on training data only before this run | +5.48% | 0.453 | −16.97% | Net short 61% of bars from Nov 2025 (small: 6–11% short); short P&L +4.90 pp, long P&L +5.04 pp; gate on 0% (one day, 2025-10-10); fees 3.46 pp. Benchmarks on v6 prices: BTC −42.42%, equal-weight −54.58% |
| 10 | 2026-10-08 | v46, seed 7 | 2025-06-24 | Same, second seed (both seeds reported) | +3.93% | 0.326 | −15.38% | Weakly bearish from Nov 2025; gate fired 2026-01-20 and held cash to end (43% of bars); short P&L +0.35 pp; fees 1.69 pp |
| 11 | 2026-10-08 | v46 + rule G, seed 42 | 2025-06-24 | v46 weights unchanged; drawdown-gate re-entry rule G at evaluation (after a 15% breach, re-enter only on the BTC 200-day trend's side instead of cash until BTC recovers). G chosen on training years only (in-sample +359% → +464%). Not retrained. | +5.42% | 0.448 | −17.02% | Breaker fired 2025-10-10 (BTC above SMA) → longs-only lock for ~3 weeks; otherwise identical to #9; short P&L +4.69 pp |
| 12 | 2026-10-08 | v46 + rule G, seed 7 | 2025-06-24 | Same, second seed | +4.84% | 0.382 | −15.39% | Breaker fired 2026-01-20 → shorts-only lock instead of cash to end; net short 52% of bars (shorts 1–4%); short P&L +1.30 pp (vs +0.35 without G) |

**Commitment (2026-10-07):** results #5–#6 will not be used to change the model design. Further design work uses the validation year only; any later test-year evaluation is appended here.

**Protocol change (2026-10-07):** validation year dropped by the author; v45 trained to 2025-06-24 and evaluated once (#7–#8) after in-sample checks only.
