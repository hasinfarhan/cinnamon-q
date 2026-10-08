"""
v6 raw price fetch: 15-min Binance spot candles (USDT pairs) for the 14 coins, from each
pair's first listed candle (BTC/ETH: Aug 2017) to the v5 end (2026-06-25 03:00 UTC).

Why: the v5 Alpaca feed has no SOL bars 2023-07-06 -> 2024-08-26 and lists CRV only from
2023-08-18 (AVAX, YFI, AAVE, GRT, SUSHI late in 2021); v5's reindex().ffill().bfill() turned
those holes into frozen prices. Nothing is filled here - each coin keeps only the candles
Binance actually printed. Output: data/raw/binance_15m/<COIN>.parquet (re-runs resume).

Source: Binance public market-data mirror (data-api.binance.vision), no API key.
"""
import time
from pathlib import Path
import requests
import pandas as pd

MAIN = Path(r"C:\E\trading_thesis\cinnamon-q")
OUT = MAIN / "data" / "raw" / "binance_15m"; OUT.mkdir(parents=True, exist_ok=True)
HOST = "https://data-api.binance.vision"
COINS = ["BTC", "ETH", "SOL", "LINK", "UNI", "AVAX", "LTC", "BCH", "AAVE", "GRT", "BAT", "YFI", "SUSHI", "CRV"]
START_MS = int(pd.Timestamp("2017-08-01").timestamp() * 1000)
END = pd.Timestamp("2026-06-25 03:00:00")          # exclusive; matches v5's last bar 02:45
END_MS = int(END.timestamp() * 1000)
STEP = 15 * 60 * 1000

def get(params):
    for attempt in range(8):
        try:
            r = requests.get(f"{HOST}/api/v3/klines", params=params, timeout=30)
            if r.status_code in (418, 429):
                time.sleep(int(r.headers.get("Retry-After", 30))); continue
            r.raise_for_status(); return r.json()
        except requests.RequestException:
            if attempt == 7: raise
            time.sleep(2 ** attempt)

for coin in COINS:
    f = OUT / f"{coin}.parquet"
    if f.exists():
        print(f"skip {coin} (have {len(pd.read_parquet(f)):,} candles)"); continue
    rows, s = [], START_MS
    while s < END_MS:
        k = get({"symbol": f"{coin}USDT", "interval": "15m", "startTime": s, "endTime": END_MS - 1, "limit": 1000})
        if not k: break
        rows += k; s = k[-1][0] + STEP
        if len(rows) % 50000 < 1000: print(f"  {coin}: {len(rows):,} candles, up to {pd.to_datetime(k[-1][0], unit='ms')}")
        time.sleep(0.05)
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tb", "tq", "ig"])
    df = pd.DataFrame({"timestamp": pd.to_datetime(df.t, unit="ms"), **{c: df[c].astype(float) for c in ["open", "high", "low", "close", "volume"]},
                       "trades": df.n.astype(int)})
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    assert df.timestamp.max() < END and df.timestamp.is_monotonic_increasing
    df.to_parquet(f, index=False)
    print(f"[done] {coin}: {len(df):,} candles {df.timestamp.min()} -> {df.timestamp.max()}")
