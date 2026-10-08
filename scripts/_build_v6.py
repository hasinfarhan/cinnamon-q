"""
Build crypto_deep_v6.parquet (same columns as crypto_deep_v5_clean.parquet) from
  - data/raw/binance_15m/<COIN>.parquet   (08_fetch_binance_v6.py; unfilled 15-min candles)
  - data/raw/fear_greed_daily.csv         (alternative.me Crypto Fear & Greed Index, daily)
  - yfinance daily VIX / DXY / TNX

Differences from v5:
  * prices: Binance spot USDT pairs instead of Alpaca; each coin has rows only where Binance
    printed a candle - no forward/back filling of prices (the v5 SOL/CRV defect)
  * utility_score: Fear & Greed / 100 (one market-wide value) instead of per-coin LLM news
    sentiment; bars on date D use the index of D-1 (the last completed day)
  * macro (vix, dxy, tnx, macro_trend_score): bars on date D use date D-1 values, as v5_clean
  * commit_count_4_weeks: the same per-coin constants as v5 (carries no information)
Features are computed exactly as in 08_fetch_crypto_v5.py, on each coin's own candles, except
  * MACD is divided by the close (MACD as a fraction of price), so its scale is comparable
    between a $3k and a $100k BTC; v5's dollar MACD was squashed flat in low-price years.
The build stops if any data check fails.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf

MAIN = Path(r"C:\E\trading_thesis\cinnamon-q")
RAW = MAIN / "data" / "raw" / "binance_15m"
DST = MAIN / "data" / "processed" / "crypto_deep_v6.parquet"
V5 = MAIN / "data" / "processed" / "crypto_deep_v5_clean.parquet"
sys.path.insert(0, str(MAIN / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("v5f", MAIN / "scripts" / "08_fetch_crypto_v5.py")
v5f = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v5f)   # reuse feature functions
COINS = sorted(["BTC", "ETH", "SOL", "LINK", "UNI", "AVAX", "LTC", "BCH", "AAVE", "GRT", "BAT", "YFI", "SUSHI", "CRV"])
END = pd.Timestamp("2026-06-25 03:00:00")
fail = []

# ── inputs ─────────────────────────────────────────────────────────────────
raw = {c: pd.read_parquet(RAW / f"{c}.parquet").set_index("timestamp").sort_index() for c in COINS}
commit = pd.read_parquet(V5, columns=["ticker", "commit_count_4_weeks"]).groupby("ticker").commit_count_4_weeks.first()

fg = pd.read_csv(MAIN / "data" / "raw" / "fear_greed_daily.csv", parse_dates=["date"]).set_index("date")["value"]
fg_days = pd.date_range(fg.index.min(), END.normalize(), freq="D")
fg_lag = (fg.reindex(fg_days).ffill() / 100.0).shift(1)            # bars on D use D-1

print("Fetching macro (VIX, DXY, TNX) from yfinance ...")
mac = {}
for name, sym in {"vix": "^VIX", "dxy": "DX-Y.NYB", "tnx": "^TNX"}.items():
    h = yf.Ticker(sym).history(start="2017-06-01", end="2026-06-27", interval="1d")["Close"]
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize(); mac[name] = h[~h.index.duplicated()]
    print(f"  {sym}: {len(h)} days {h.index.min().date()} -> {h.index.max().date()}")
mac = pd.DataFrame(mac)
m_days = pd.date_range("2017-06-01", END.normalize(), freq="D")
mac = mac.reindex(m_days).ffill()                                   # weekends/holidays: last published close
mac["macro_trend_score"] = 1.0 - np.clip((mac["vix"] - 10.0) / 70.0, 0.0, 1.0)
mac_lag = mac.shift(1)                                              # bars on D use D-1
assert mac_lag.loc["2017-08-17":].notna().all().all(), "macro gaps"

# ── features per coin, on its own candles ───────────────────────────────────
btc = raw["BTC"]; btc_ret = btc["close"].pct_change().fillna(0.0)
btc_sma = btc["close"].rolling(200).mean().bfill()
rows = []
for c in COINS:
    d = raw[c]; close, high, low = d["close"], d["high"], d["low"]
    rets = close.pct_change().fillna(0.0)
    date = d.index.normalize()
    f = pd.DataFrame({
        "timestamp": d.index, "ticker": c,
        "open": d["open"].values, "high": high.values, "low": low.values, "close": close.values, "volume": d["volume"].values,
        "RSI_14": v5f._rsi(close).values, "MACD": (v5f._macd(close) / close).values, "BB_WIDTH": v5f._bb_width(close).values,
        "ATR_NORM": v5f._atr_norm(high, low, close).values, "SMA_200_RATIO": v5f._sma_ratio(close).values,
        "corr_btc": v5f._corr_btc(rets, btc_ret.reindex(d.index)).values,
        "garch_volatility": v5f._ewma_vol(rets).values,
        "commit_count_4_weeks": float(commit[c]),
        "utility_score": fg_lag.reindex(date).fillna(0.5).values,   # 0.5 only before Feb 2018
        "btc_sma_200": btc_sma.reindex(d.index).ffill().bfill().values,
        "macro_trend_score": mac_lag["macro_trend_score"].reindex(date).values.astype(np.float32),
        "vix": mac_lag["vix"].reindex(date).values, "dxy": mac_lag["dxy"].reindex(date).values,
        "tnx": mac_lag["tnx"].reindex(date).values,
    })
    rows.append(f)
    print(f"  {c:5s} {len(f):7,} rows  {f.timestamp.min()} -> {f.timestamp.max()}")
df = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "ticker"]).reset_index(drop=True)
v5cols = list(pd.read_parquet(V5).columns)
df = df[v5cols]

# ── data checks ─────────────────────────────────────────────────────────────
print("\n=== DATA CHECKS ===")
feat = [c for c in v5cols if c not in ("timestamp", "ticker")]
nan = df[feat].isna().sum(); nan = nan[nan > 0]
if len(nan): fail.append(f"NaNs in features: {nan.to_dict()}")
print("  NaNs:", "none" if not len(nan) else nan.to_dict())
for c in COINS:
    s = raw[c]["close"].resample("4h").last().dropna()
    r = s.pct_change()
    same = (s.diff() == 0).astype(int); run = same.groupby((same == 0).cumsum()).cumsum().max()
    big = r[(r.abs() > 0.35)]
    # the first 20 days after listing are not tradable (listing mask), so launch-day swings are exempt
    tradable_from = s.index[0] + pd.Timedelta(days=20)
    glitch = [t for t in big.index if t >= tradable_from and abs(r.shift(-1).get(t, 0)) > 0.25
              and np.sign(r.shift(-1)[t]) != np.sign(r[t])]
    print(f"  {c:5s} longest run of identical 4h closes: {run:2d} bars | 4h moves >35%: {len(big)} "
          f"{[f'{t:%Y-%m-%d %H}h {r[t]:+.0%}' for t in big.index]} | spike-and-reverse: {len(glitch)}")
    if run > 12: fail.append(f"{c}: frozen for {run} 4h bars")
    if glitch: fail.append(f"{c}: spike-and-reverse bars {glitch}")
# cross-source: Binance vs Alpaca 4h closes where Alpaca actually traded (v5 rows that changed price)
v5 = pd.read_parquet(V5, columns=["timestamp", "ticker", "close"])
print("  Binance vs Alpaca (v5) 4h close, |difference| where both have real prices:")
for c in COINS:
    a = v5[v5.ticker == c].set_index("timestamp")["close"].resample("4h").last()
    a = a[a.diff() != 0]                                             # drop frozen (filled) stretches
    b = raw[c]["close"].resample("4h").last()
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna(); dev = (j.b / j.a - 1).abs()
    top = dev.nlargest(2)
    print(f"    {c:5s} n={len(j):5d}  median {dev.median()*100:.3f}%  99th pct {dev.quantile(.99)*100:.2f}%  max {dev.max()*100:.1f}%  "
          f"largest at: " + ", ".join(f"{t:%Y-%m-%d %H}h Alpaca {j.a[t]:.4g} vs Binance {j.b[t]:.4g}" for t in top.index))
    if dev.quantile(.99) > 0.02: fail.append(f"{c}: Binance and Alpaca disagree (99th pct {dev.quantile(.99):.1%})")
print(f"  Fear & Greed lagged: first value used {fg_lag.first_valid_index().date()}, range {df.utility_score.min():.2f}-{df.utility_score.max():.2f}")

if fail:
    print("\nBUILD STOPPED - checks failed:"); [print("  -", x) for x in fail]; sys.exit(1)
df.to_parquet(DST, index=False, compression="snappy")
print(f"\nSaved {DST}  ({len(df):,} rows, {df.ticker.nunique()} coins, {df.timestamp.min()} -> {df.timestamp.max()}, "
      f"{DST.stat().st_size / 1e6:.0f} MB)")
