"""
Build data/processed/crypto_deep_v6.parquet (15-minute rows, one per coin and candle) from
  - data/raw/binance_15m/<COIN>.parquet   (scripts/01_fetch_prices.py)
  - data/raw/fear_greed_daily.csv         (scripts/02_fetch_fear_greed.py)
  - daily VIX, DXY and 10-year Treasury yield from Yahoo Finance (yfinance)

  * prices: Binance spot USDT pairs; each coin has rows only where Binance printed a candle
  * utility_score (sentiment): Fear & Greed / 100, one market-wide value; bars on date D
    use the index of D-1 (the last completed day)
  * vix, dxy, tnx, macro_trend_score: bars on date D use the values of D-1
  * commit_count_4_weeks (developer activity): one CoinGecko snapshot per coin, constant
    over time (it therefore carries no time-varying information)
  * indicators (scripts/feature_functions.py) on each coin's own candles; MACD is divided
    by the close so that its scale is comparable across price levels

The build stops if any data check fails.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(os.environ.get("CQ_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "data" / "raw" / "binance_15m"
DST = ROOT / "data" / "processed" / "crypto_deep_v6.parquet"
_spec = importlib.util.spec_from_file_location("ff", Path(__file__).resolve().parent / "feature_functions.py")
ff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ff)

COINS = sorted(["BTC", "ETH", "SOL", "LINK", "UNI", "AVAX", "LTC", "BCH", "AAVE", "GRT", "BAT", "YFI", "SUSHI", "CRV"])
END = pd.Timestamp("2026-06-25 03:00:00")
# CoinGecko developer-activity snapshot (commits over four weeks), one value per coin
COMMITS = {"AAVE": 13.373279520344804, "AVAX": 43.93322451793489, "BAT": 20.128802693183125,
           "BCH": 14.069910840209642, "BTC": 404.3634266563654, "CRV": 49.793899024564226,
           "ETH": 476.79433091370646, "GRT": 1007.8746718119779, "LINK": 17.275530934370288,
           "LTC": 0.10456406594945777, "SOL": 1484.3788362743253, "SUSHI": 213.25888739829898,
           "UNI": 18.19161028419901, "YFI": 0.6954452406458339}
COLUMNS = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "RSI_14", "MACD", "BB_WIDTH",
           "ATR_NORM", "SMA_200_RATIO", "corr_btc", "garch_volatility", "commit_count_4_weeks", "utility_score",
           "btc_sma_200", "macro_trend_score", "vix", "dxy", "tnx"]
fail = []

# ── inputs ─────────────────────────────────────────────────────────────────
raw = {c: pd.read_parquet(RAW / f"{c}.parquet").set_index("timestamp").sort_index() for c in COINS}

fg = pd.read_csv(ROOT / "data" / "raw" / "fear_greed_daily.csv", parse_dates=["date"]).set_index("date")["value"]
fg_days = pd.date_range(fg.index.min(), END.normalize(), freq="D")
fg_lag = (fg.reindex(fg_days).ffill() / 100.0).shift(1)            # bars on D use D-1

print("Fetching macro (VIX, DXY, TNX) from yfinance ...")
mac = {}
for name, sym in {"vix": "^VIX", "dxy": "DX-Y.NYB", "tnx": "^TNX"}.items():
    h = yf.Ticker(sym).history(start="2017-06-01", end="2026-06-27", interval="1d")["Close"]
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    mac[name] = h[~h.index.duplicated()]
    print(f"  {sym}: {len(h)} days {h.index.min().date()} -> {h.index.max().date()}")
mac = pd.DataFrame(mac)
mac = mac.reindex(pd.date_range("2017-06-01", END.normalize(), freq="D")).ffill()   # weekends: last close
mac["macro_trend_score"] = 1.0 - np.clip((mac["vix"] - 10.0) / 70.0, 0.0, 1.0)
mac_lag = mac.shift(1)                                              # bars on D use D-1
assert mac_lag.loc["2017-08-17":].notna().all().all(), "macro gaps"

# ── features per coin, on its own candles ───────────────────────────────────
btc = raw["BTC"]
btc_ret = btc["close"].pct_change().fillna(0.0)
btc_sma = btc["close"].rolling(200).mean().bfill()
rows = []
for c in COINS:
    d = raw[c]
    close, high, low = d["close"], d["high"], d["low"]
    rets = close.pct_change().fillna(0.0)
    date = d.index.normalize()
    f = pd.DataFrame({
        "timestamp": d.index, "ticker": c,
        "open": d["open"].values, "high": high.values, "low": low.values, "close": close.values, "volume": d["volume"].values,
        "RSI_14": ff._rsi(close).values, "MACD": (ff._macd(close) / close).values, "BB_WIDTH": ff._bb_width(close).values,
        "ATR_NORM": ff._atr_norm(high, low, close).values, "SMA_200_RATIO": ff._sma_ratio(close).values,
        "corr_btc": ff._corr_btc(rets, btc_ret.reindex(d.index)).values,
        "garch_volatility": ff._ewma_vol(rets).values,
        "commit_count_4_weeks": float(COMMITS[c]),
        "utility_score": fg_lag.reindex(date).fillna(0.5).values,   # 0.5 only before Feb 2018
        "btc_sma_200": btc_sma.reindex(d.index).ffill().bfill().values,
        "macro_trend_score": mac_lag["macro_trend_score"].reindex(date).values.astype(np.float32),
        "vix": mac_lag["vix"].reindex(date).values, "dxy": mac_lag["dxy"].reindex(date).values,
        "tnx": mac_lag["tnx"].reindex(date).values,
    })
    rows.append(f)
    print(f"  {c:5s} {len(f):7,} rows  {f.timestamp.min()} -> {f.timestamp.max()}")
df = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "ticker"]).reset_index(drop=True)[COLUMNS]

# ── data checks ─────────────────────────────────────────────────────────────
print("\n=== DATA CHECKS ===")
nan = df[COLUMNS[2:]].isna().sum()
nan = nan[nan > 0]
if len(nan):
    fail.append(f"NaNs in features: {nan.to_dict()}")
print("  NaNs:", "none" if not len(nan) else nan.to_dict())
for c in COINS:
    s = raw[c]["close"].resample("4h").last().dropna()
    r = s.pct_change()
    same = (s.diff() == 0).astype(int)
    run = same.groupby((same == 0).cumsum()).cumsum().max()
    big = r[(r.abs() > 0.35)]
    # the first 20 days after listing are not tradable (listing mask), so launch-day swings are exempt
    tradable_from = s.index[0] + pd.Timedelta(days=20)
    glitch = [t for t in big.index if t >= tradable_from and abs(r.shift(-1).get(t, 0)) > 0.25
              and np.sign(r.shift(-1)[t]) != np.sign(r[t])]
    print(f"  {c:5s} longest run of identical 4h closes: {run:2d} bars | 4h moves >35%: {len(big)} "
          f"{[f'{t:%Y-%m-%d %H}h {r[t]:+.0%}' for t in big.index]} | spike-and-reverse: {len(glitch)}")
    if run > 12:
        fail.append(f"{c}: frozen for {run} 4h bars")
    if glitch:
        fail.append(f"{c}: spike-and-reverse bars {glitch}")
print(f"  Fear & Greed lagged: first value used {fg_lag.first_valid_index().date()}, "
      f"range {df.utility_score.min():.2f}-{df.utility_score.max():.2f}")

if fail:
    print("\nBUILD STOPPED - checks failed:")
    for x in fail:
        print("  -", x)
    sys.exit(1)
DST.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(DST, index=False, compression="snappy")
print(f"\nSaved {DST}  ({len(df):,} rows, {df.ticker.nunique()} coins, {df.timestamp.min()} -> {df.timestamp.max()}, "
      f"{DST.stat().st_size / 1e6:.0f} MB)")
