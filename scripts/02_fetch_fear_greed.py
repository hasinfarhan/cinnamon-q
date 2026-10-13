"""
Fetch the daily Crypto Fear & Greed Index (Alternative.me), available from February 2018.

Output: data/raw/fear_greed_daily.csv with columns date, value (0-100), value_classification.
The dataset build lags the index by one day (bars on date D use the value for D-1).
"""
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(os.environ.get("CQ_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "data" / "raw" / "fear_greed_daily.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

j = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=30).json()
df = pd.DataFrame(j["data"])
df = pd.DataFrame({"date": pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize(),
                   "value": df["value"].astype(int),
                   "value_classification": df["value_classification"]})
df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
print(f"Saved {OUT}: {len(df):,} days, {df.date.min().date()} -> {df.date.max().date()}")
