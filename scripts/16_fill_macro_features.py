"""
Step 2 of Clean Rebuild: Fill the 5 macro features that were 100% NaN.

Features filled:
  vix              — CBOE Volatility Index (fear gauge)
  dxy              — US Dollar Index (dollar strength)
  tnx              — 10-Year Treasury Yield (risk-free rate proxy)
  btc_sma_200      — Bitcoin 200-day SMA (bull/bear market regime)
  macro_trend_score — Composite 0-1 score (1=risk-on, 0=risk-off)

Method:
  VIX, DXY, TNX  → yfinance daily history, forward-filled to 15-min bars
  BTC SMA 200    → computed from BTC close prices in the dataset (200*96 bars)
  macro_trend_score → normalised weighted average of the four signals:
    score = 0.30 * (1 - vix_norm)   high VIX  = fear   = low score
          + 0.20 * (1 - dxy_norm)   high DXY  = strong $ = bad for crypto
          + 0.20 * (1 - tnx_norm)   high TNX  = high rates = risk-off
          + 0.30 * btc_trend        above SMA200 = bull = high score

Input : data/processed/crypto_deep_v4_sentiment.parquet
Output: data/processed/crypto_deep_v4_sentiment.parquet  (in-place update)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))

DATA_PATH = root / "data" / "processed" / "crypto_deep_v4_sentiment.parquet"

# ── Normalisation bounds (based on historical ranges) ────────
VIX_LO,  VIX_HI  = 10.0, 80.0   # VIX: 10=calm, 80=crash
DXY_LO,  DXY_HI  = 88.0, 115.0  # DXY range 2023-2026
TNX_LO,  TNX_HI  =  0.5,   5.5  # 10-yr yield range
BTC_SMA_CLIP     =  0.5          # clip ratio to [-50%, +50%] of SMA

print("=" * 65)
print("  MACRO FEATURES FILL — Step 2")
print("=" * 65)

# ── 1. Load dataset ───────────────────────────────────────────
print(f"\n[1/6] Loading dataset...")
df = pd.read_parquet(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
print(f"  Rows: {len(df):,}  |  Assets: {sorted(df['ticker'].unique())}")

ts_min = df["timestamp"].min()
ts_max = df["timestamp"].max()
print(f"  Period: {ts_min.date()} to {ts_max.date()}")

# Fetch with 1-year buffer before start for SMA warmup
fetch_start = (ts_min - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
fetch_end   = (ts_max + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

# ── 2. Fetch macro data from yfinance ─────────────────────────
print(f"\n[2/6] Fetching macro data from yfinance ({fetch_start} to {fetch_end})...")

tickers = {"^VIX": "vix", "DX-Y.NYB": "dxy", "^TNX": "tnx"}
macro_daily = {}

for yticker, col in tickers.items():
    try:
        hist = yf.download(yticker, start=fetch_start, end=fetch_end,
                           interval="1d", progress=False, auto_adjust=True)
        if hist.empty:
            print(f"  {yticker}: NO DATA returned")
            continue
        # Handle MultiIndex columns from yfinance
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        series = hist["Close"].dropna()
        series.index = pd.to_datetime(series.index, utc=True).normalize()
        macro_daily[col] = series
        print(f"  {yticker:<12} ({col}): {len(series)} daily bars  "
              f"range=[{series.min():.2f}, {series.max():.2f}]  "
              f"latest={series.iloc[-1]:.2f}")
    except Exception as e:
        print(f"  {yticker}: FETCH ERROR — {e}")

if len(macro_daily) < 3:
    print("\nERROR: Could not fetch enough macro data. Check internet/API.")
    sys.exit(1)

# ── 3. Compute BTC SMA 200 (daily) ────────────────────────────
# Use yfinance BTC history for warmup (dataset starts May 2023,
# SMA200 needs 200 days before that → fetch from April 2022)
print(f"\n[3/6] Computing BTC 200-day SMA (with yfinance warmup)...")
try:
    btc_hist = yf.download("BTC-USD", start=fetch_start, end=fetch_end,
                           interval="1d", progress=False, auto_adjust=True)
    if isinstance(btc_hist.columns, pd.MultiIndex):
        btc_hist.columns = btc_hist.columns.get_level_values(0)
    btc_yf = btc_hist["Close"].dropna()
    btc_yf.index = pd.to_datetime(btc_yf.index, utc=True).normalize()
    print(f"  yfinance BTC bars: {len(btc_yf)}  ({btc_yf.index[0].date()} to {btc_yf.index[-1].date()})")
except Exception as e:
    print(f"  yfinance BTC fetch failed ({e}) — falling back to dataset prices")
    btc_yf = pd.Series(dtype=float)

# Also get BTC from dataset (for any gaps yfinance misses)
btc_ds = df[df["ticker"] == "BTC"][["timestamp", "close"]].copy()
btc_ds["date"] = btc_ds["timestamp"].dt.normalize()
btc_ds_daily = btc_ds.groupby("date")["close"].last()
btc_ds_daily.index = pd.to_datetime(btc_ds_daily.index, utc=True)

# Merge: prefer yfinance for warmup, use dataset for any remaining gaps
if len(btc_yf) > 0:
    btc_daily = btc_yf.combine_first(btc_ds_daily).sort_index()
else:
    btc_daily = btc_ds_daily.sort_index()

btc_sma200 = btc_daily.rolling(window=200, min_periods=200).mean()
# BTC trend: how far above/below the 200-day SMA (as a ratio)
btc_trend_raw = (btc_daily / btc_sma200 - 1.0).clip(-BTC_SMA_CLIP, BTC_SMA_CLIP)

print(f"  Combined BTC bars: {len(btc_daily)}")
print(f"  SMA200 range     : [{btc_sma200.dropna().min():.0f}, {btc_sma200.dropna().max():.0f}]")
print(f"  SMA200 NaN count : {btc_sma200.isna().sum()} (should be < 200)")
print(f"  Trend range      : [{btc_trend_raw.dropna().min():.3f}, {btc_trend_raw.dropna().max():.3f}]")

macro_daily["btc_sma200_raw"]  = btc_sma200
macro_daily["btc_trend_raw"]   = btc_trend_raw

# ── 4. Forward-fill daily → 15-min timestamps ─────────────────
print(f"\n[4/6] Forward-filling daily macro → 15-min bars...")

# Build a date lookup for each 15-min bar
all_ts = df["timestamp"].drop_duplicates().sort_values()
ts_dates = all_ts.dt.normalize()   # UTC midnight of each bar

def daily_to_15min(series: pd.Series, name: str) -> pd.Series:
    """Map a daily series onto the 15-min timestamp index via forward-fill."""
    # Reindex to all unique dates in our dataset, then ffill
    all_dates = sorted(ts_dates.unique())
    reindexed = series.reindex(all_dates, method="ffill")

    # Build timestamp → value mapping
    ts_to_val = dict(zip(all_dates, reindexed.values))
    mapped = ts_dates.map(ts_to_val)
    result = pd.Series(mapped.values, index=all_ts.values, name=name)

    n_nan = result.isna().sum()
    n_total = len(result)
    print(f"  {name:<20} {n_total - n_nan:,}/{n_total:,} filled  "
          f"NaN={n_nan} ({n_nan/n_total*100:.1f}%)")
    return result

vix_15m       = daily_to_15min(macro_daily["vix"],            "vix")
dxy_15m       = daily_to_15min(macro_daily["dxy"],            "dxy")
tnx_15m       = daily_to_15min(macro_daily["tnx"],            "tnx")
sma200_15m    = daily_to_15min(macro_daily["btc_sma200_raw"], "btc_sma_200")
trend_15m     = daily_to_15min(macro_daily["btc_trend_raw"],  "btc_trend")

# Build ts→value dicts for fast row-wise lookup
vix_map    = dict(zip(all_ts.values, vix_15m.values))
dxy_map    = dict(zip(all_ts.values, dxy_15m.values))
tnx_map    = dict(zip(all_ts.values, tnx_15m.values))
sma200_map = dict(zip(all_ts.values, sma200_15m.values))
trend_map  = dict(zip(all_ts.values, trend_15m.values))

# ── 5. Map onto df rows ───────────────────────────────────────
print(f"\n[5/6] Mapping values onto {len(df):,} rows...")
ts_vals = df["timestamp"].values

df["vix"]         = [vix_map.get(t, np.nan)    for t in ts_vals]
df["dxy"]         = [dxy_map.get(t, np.nan)    for t in ts_vals]
df["tnx"]         = [tnx_map.get(t, np.nan)    for t in ts_vals]
df["btc_sma_200"] = [sma200_map.get(t, np.nan) for t in ts_vals]

# Compute macro_trend_score from normalised components
vix_arr   = df["vix"].values
dxy_arr   = df["dxy"].values
tnx_arr   = df["tnx"].values
trend_arr = np.array([trend_map.get(t, np.nan) for t in ts_vals])

# Normalise each to 0-1
vix_norm   = np.clip((vix_arr - VIX_LO)  / (VIX_HI  - VIX_LO),  0, 1)
dxy_norm   = np.clip((dxy_arr - DXY_LO)  / (DXY_HI  - DXY_LO),  0, 1)
tnx_norm   = np.clip((tnx_arr - TNX_LO)  / (TNX_HI  - TNX_LO),  0, 1)
btc_norm   = np.clip((trend_arr + BTC_SMA_CLIP) / (2 * BTC_SMA_CLIP), 0, 1)

# Composite (weights: VIX 30%, DXY 20%, TNX 20%, BTC trend 30%)
macro_score = (0.30 * (1 - vix_norm) +
               0.20 * (1 - dxy_norm) +
               0.20 * (1 - tnx_norm) +
               0.30 * btc_norm)

df["macro_trend_score"] = macro_score

# ── 6. Verify and save ────────────────────────────────────────
print(f"\n[6/6] Verification + save...")

for col in ["vix", "dxy", "tnx", "btc_sma_200", "macro_trend_score"]:
    s = df[col]
    n_nan = s.isna().sum()
    print(f"  {col:<22} NaN={n_nan:,}  "
          f"range=[{s.min():.3f}, {s.max():.3f}]  "
          f"mean={s.mean():.3f}  std={s.std():.3f}")

df.to_parquet(DATA_PATH, index=False)
size_mb = DATA_PATH.stat().st_size / 1024 / 1024
print(f"\n  Saved: {DATA_PATH}")
print(f"  Size : {size_mb:.1f} MB")

print("\n" + "=" * 65)
print("  Step 2 DONE — macro features filled.")
print("  Next: retrain the model on crypto_deep_v4_sentiment.parquet")
print("=" * 65)
