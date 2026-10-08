"""
Build crypto_deep_v5_clean.parquet from crypto_deep_v5.parquet, removing two
look-ahead leaks found in the v5 pipeline (the original file is not modified):

1. Macro (vix, dxy, tnx, macro_trend_score): v5 attaches each date's daily close
   (US market close, ~20:00-21:00 UTC) to every bar of that same date, from 00:00 UTC.
   Fix: each bar on date D uses the value of date D-1, which is fully published
   before any bar of D.
2. News sentiment (utility_score): v5 gives each bar the score built from all of
   that calendar date's headlines, including later ones. Fix: each bar on date D
   uses the score of date D-1, the last fully completed day.

The very first date has no previous day; it keeps its own value (training data only).
"""
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(r"C:\E\trading_thesis\cinnamon-q")
SRC = MAIN / "data" / "processed" / "crypto_deep_v5.parquet"
DST = MAIN / "data" / "processed" / "crypto_deep_v5_clean.parquet"
MACRO = ["vix", "dxy", "tnx", "macro_trend_score"]

df = pd.read_parquet(SRC)
orig_cols, orig_len, orig_dtypes = list(df.columns), len(df), df.dtypes.copy()
ts = pd.to_datetime(df["timestamp"])
date = ts.dt.normalize()
print(f"Loaded {SRC.name}: {orig_len:,} rows, {df['ticker'].nunique()} tickers, {date.min().date()} -> {date.max().date()}")

# --- the fix is only valid if each value is constant within its date --------
macro_spread = df.groupby(date)[MACRO].agg(lambda s: s.max() - s.min()).abs().max()
util_spread = (df.groupby([df["ticker"], date])["utility_score"].agg(lambda s: s.max() - s.min())).abs().max()
print("max within-date spread  macro:", macro_spread.round(6).to_dict(), "| utility:", round(float(util_spread), 6))
assert (macro_spread < 1e-6).all(), "macro values vary within a date - day-lag logic would be wrong"
assert util_spread < 1e-6, "utility_score varies within a (ticker, date) - day-lag logic would be wrong"

full_days = pd.date_range(date.min(), date.max(), freq="D")

# --- 1. macro: one global series per date, lagged one calendar day ----------
daily_macro = df.groupby(date)[MACRO].first().reindex(full_days).ffill()
lagged_macro = daily_macro.shift(1)
lagged_macro.iloc[0] = daily_macro.iloc[0]
for col in MACRO:
    new = date.map(lagged_macro[col])
    assert new.notna().all(), f"{col}: NaN after lag"
    changed = float((new.values != df[col].values).mean())
    df[col] = new.astype(orig_dtypes[col]).values
    print(f"  {col:18s} lagged one day  ({changed:.1%} of rows changed)")

# --- 2. news sentiment: per-ticker daily series, lagged one calendar day -----
daily_util = (df.assign(_d=date).groupby(["_d", "ticker"])["utility_score"].first()
                .unstack("ticker").reindex(full_days).ffill())
lagged_util = daily_util.shift(1)
lagged_util.iloc[0] = daily_util.iloc[0]
stacked = lagged_util.stack()
key = pd.MultiIndex.from_arrays([date, df["ticker"]])
new_util = pd.Series(stacked.reindex(key).values, index=df.index)
assert new_util.notna().all(), "utility_score: NaN after lag"
changed = float((new_util.values != df["utility_score"].values).mean())
df["utility_score"] = new_util.astype(orig_dtypes["utility_score"]).values
print(f"  {'utility_score':18s} lagged one day  ({changed:.1%} of rows changed)")

# --- spot check: a bar on D now carries D-1's original value ----------------
orig = pd.read_parquet(SRC, columns=["timestamp", "ticker", "vix", "utility_score"])
orig["_d"] = pd.to_datetime(orig["timestamp"]).dt.normalize()
d = pd.Timestamp("2025-07-15")
o_prev = orig[(orig["_d"] == d - pd.Timedelta(days=1)) & (orig["ticker"] == "BTC")].iloc[0]
c_now = df[(date == d) & (df["ticker"] == "BTC")].iloc[0]
print(f"spot check BTC {d.date()}: clean vix={c_now['vix']:.3f} == orig vix on {(d - pd.Timedelta(days=1)).date()}={o_prev['vix']:.3f};"
      f" clean utility={c_now['utility_score']:.4f} == orig prev-day utility={o_prev['utility_score']:.4f}")
assert np.isclose(c_now["vix"], o_prev["vix"]) and np.isclose(c_now["utility_score"], o_prev["utility_score"])

assert list(df.columns) == orig_cols and len(df) == orig_len
df.to_parquet(DST, index=False)
print(f"Saved {DST} ({len(df):,} rows, columns unchanged)")
