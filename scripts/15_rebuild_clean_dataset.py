"""
Step 1 of Clean Rebuild: Remove dead assets, recompute broken features.

Dead assets removed: ALGO, MKR, PAXG
  - ALGO : only 25 days of data (Jun 2023), zero test-period rows
  - MKR  : ends Sep 2025 (Sky rebrand), zero test-period rows, zero volume
  - PAXG : 968-day gap in middle, zero volume throughout

Broken features recomputed:
  - corr_btc : was stuck at 1.0 for every asset → recomputed as
               rolling 96-bar (24h) Pearson correlation of RETURNS vs BTC
  - garch_volatility : was all 0.0 → recomputed as 96-bar EWM std of returns
                       (RiskMetrics-style EWMA volatility, a standard GARCH proxy)

Output: data/processed/crypto_deep_v4.parquet  (14 assets, clean features)
        → next step: run 09b_historical_sentiment.py on this file
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))

INPUT  = root / "data" / "processed" / "crypto_deep_v3.parquet"
OUTPUT = root / "data" / "processed" / "crypto_deep_v4.parquet"

REMOVE_ASSETS  = {"ALGO", "MKR", "PAXG"}
CORR_WINDOW    = 96       # 24 hours of 15-min bars
EWMA_HALFLIFE  = 96       # ~24h half-life for volatility decay

print("=" * 65)
print("  CLEAN DATASET REBUILD — Step 1")
print("=" * 65)

# ── 1. Load ───────────────────────────────────────────────────
print(f"\n[1/6] Loading {INPUT.name}...")
df = pd.read_parquet(INPUT)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

before_assets = sorted(df["ticker"].unique())
before_rows   = len(df)
print(f"  Rows   : {before_rows:,}")
print(f"  Assets : {before_assets}")

# ── 2. Remove dead assets ─────────────────────────────────────
print(f"\n[2/6] Removing dead assets: {sorted(REMOVE_ASSETS)}")
df = df[~df["ticker"].isin(REMOVE_ASSETS)].copy()
df = df.reset_index(drop=True)

after_assets = sorted(df["ticker"].unique())
print(f"  Rows after : {len(df):,}  (removed {before_rows - len(df):,})")
print(f"  Assets now : {after_assets}")

# ── 3. Compute corr_btc properly ──────────────────────────────
print(f"\n[3/6] Recomputing corr_btc (rolling {CORR_WINDOW}-bar correlation of returns vs BTC)...")

# Build a wide price table indexed by timestamp
price_wide = df.pivot_table(
    index="timestamp", columns="ticker", values="close", aggfunc="first"
)
price_wide = price_wide.sort_index().ffill()

# Returns
returns_wide = price_wide.pct_change()

btc_ret = returns_wide["BTC"]

corr_dict = {}
for ticker in after_assets:
    if ticker == "BTC":
        # BTC corr with itself is always 1 — store as 1 then let normaliser handle
        corr_dict["BTC"] = pd.Series(1.0, index=returns_wide.index)
    else:
        # Rolling Pearson correlation
        asset_ret = returns_wide[ticker]
        rolling_corr = asset_ret.rolling(window=CORR_WINDOW, min_periods=CORR_WINDOW // 2).corr(btc_ret)
        # Fill NaNs at the start with 0.5 (neutral)
        rolling_corr = rolling_corr.fillna(0.5)
        corr_dict[ticker] = rolling_corr
        sample = rolling_corr.dropna()
        print(f"  {ticker:<8}  mean={sample.mean():+.3f}  min={sample.min():+.3f}  max={sample.max():+.3f}")

print(f"  BTC      corr=1.000 (reference)")

# Map back to long-format df
df["corr_btc"] = df.apply(
    lambda row: corr_dict[row["ticker"]].get(row["timestamp"], 0.5), axis=1
)
print(f"  Done. corr_btc range: [{df['corr_btc'].min():.3f}, {df['corr_btc'].max():.3f}]  std={df['corr_btc'].std():.4f}")

# ── 4. Compute GARCH proxy (EWMA volatility) ──────────────────
print(f"\n[4/6] Recomputing garch_volatility (EWMA volatility, halflife={EWMA_HALFLIFE} bars)...")

vol_dict = {}
for ticker in after_assets:
    asset_ret = returns_wide[ticker]
    # Exponentially weighted std — the RiskMetrics EWMA model
    ewma_vol = asset_ret.ewm(halflife=EWMA_HALFLIFE, min_periods=10).std()
    ewma_vol = ewma_vol.fillna(ewma_vol.median())
    vol_dict[ticker] = ewma_vol
    sample = ewma_vol.dropna()
    print(f"  {ticker:<8}  mean={sample.mean():.5f}  max={sample.max():.5f}")

df["garch_volatility"] = df.apply(
    lambda row: float(vol_dict[row["ticker"]].get(row["timestamp"], 0.0)), axis=1
)
print(f"  Done. garch_volatility range: [{df['garch_volatility'].min():.6f}, {df['garch_volatility'].max():.6f}]  std={df['garch_volatility'].std():.6f}")

# ── 5. Verify no constants remain ─────────────────────────────
print(f"\n[5/6] Verification check...")
problems = []
for col in ["corr_btc", "garch_volatility"]:
    s = df[col]
    if s.std() < 1e-6:
        problems.append(f"  STILL CONSTANT: {col}")
    else:
        print(f"  {col:<20}  std={s.std():.6f}  range=[{s.min():.4f}, {s.max():.4f}]  OK")

for ticker in after_assets:
    sub = df[df["ticker"] == ticker]
    n_test = (sub["timestamp"] >= pd.Timestamp("2025-10-12", tz="UTC")).sum()
    if n_test == 0:
        problems.append(f"  ZERO TEST ROWS: {ticker}")

if problems:
    print("  PROBLEMS FOUND:")
    for p in problems:
        print(p)
else:
    print("  All checks passed.")

# ── 6. Save ───────────────────────────────────────────────────
print(f"\n[6/6] Saving to {OUTPUT.name}...")
df.to_parquet(OUTPUT, index=False)
size_mb = OUTPUT.stat().st_size / 1024 / 1024
print(f"  Saved: {OUTPUT}")
print(f"  Size : {size_mb:.1f} MB")
print(f"  Rows : {len(df):,}")
print(f"  Assets ({len(after_assets)}): {after_assets}")

print("\n" + "=" * 65)
print("  Step 1 DONE.")
print("  Next: run scripts/09b_historical_sentiment.py")
print("        (update DATA_PATH and OUTPUT_PATH to v4 files)")
print("=" * 65)
