"""
Market data preparation for the trading environment.

Turns the 15-minute dataset built by scripts/03_build_dataset.py into time-aligned
[n_bars, n_assets] float32 arrays on 4-hour bars:

  * per-asset inputs (returns, multi-horizon returns, technical indicators, correlation
    with BTC, EWMA volatility, sentiment, developer activity), scaled to comparable ranges;
  * market-wide inputs (VIX, DXY, 10-year yield, macro score; BTC 1-day and 7-day returns,
    market 7-day return, BTC relative to its 200-day SMA);
  * each coin's own 200-day trend;
  * a listing mask: a coin becomes tradable 20 days after its first candle.

Min-max scaling ranges are fitted on rows up to `fit_end` only (the last training bar),
so no statistic of the test period enters the inputs.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

TA_COLUMNS = ['RSI_14', 'MACD', 'BB_WIDTH', 'ATR_NORM', 'SMA_200_RATIO']
MACRO_COLS = ['vix', 'dxy', 'tnx', 'macro_trend_score']
REQUIRED = ['timestamp', 'ticker', 'close', 'commit_count_4_weeks', 'utility_score',
            'corr_btc', 'garch_volatility', *TA_COLUMNS, *MACRO_COLS]

# Time-indexed arrays returned by prepare_data (all have n_bars rows)
ARRAY_KEYS = ['returns', 'volatility', 'commit', 'utility', 'price', 'corr', 'garch',
              'rsi', 'macd', 'bbw', 'atr', 'sma_ratio', 'ret_4h', 'ret_1d', 'ret_7d',
              'macro', 'regime', 'coin_trend', 'avail']


def _fit_range(df: pd.DataFrame, col: str, fit_end: Optional[str], clip_pct: Optional[float] = None):
    """Per-asset scaling range of `col`, fitted only on rows up to `fit_end`.

    With clip_pct (e.g. 0.5) the range is the clip_pct and (100 - clip_pct) percentiles
    instead of the minimum and maximum, so a single extreme event does not set the scale;
    the caller clips the scaled value to [0, 1].
    """
    src = df if fit_end is None else df[pd.to_datetime(df['timestamp']) <= pd.Timestamp(fit_end)]
    if clip_pct:
        g = src.groupby('ticker')[col]
        stats = pd.DataFrame({'min': g.quantile(clip_pct / 100.0), 'max': g.quantile(1.0 - clip_pct / 100.0)})
    else:
        stats = src.groupby('ticker')[col].agg(['min', 'max'])
    return df['ticker'].map(stats['min']), df['ticker'].map(stats['max'])


def _scaled(df: pd.DataFrame, col: str, fit_end, clip_pct, default: float) -> pd.Series:
    lo, hi = _fit_range(df, col, fit_end, clip_pct)
    rng = hi - lo
    out = pd.Series(np.where(rng > 0, (df[col] - lo) / rng, default), index=df.index).fillna(default)
    return out.clip(0.0, 1.0) if clip_pct else out


def prepare_data(data_path: Path, fit_end: Optional[str] = None, clip_pct: Optional[float] = None,
                 resample_freq: str = '4h') -> Dict:
    """Load the dataset and return aligned arrays plus metadata.

    Returns a dict with the arrays named in ARRAY_KEYS and 'asset_list', 'timestamps',
    'n_steps', 'n_assets', 'bars_per_day'.
    """
    df = pd.read_parquet(data_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"dataset is missing columns {missing}"
    assert resample_freq == '4h', "the model is defined on 4-hour bars"
    df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)

    # ── 15-minute candles -> 4-hour bars (per coin, only where the coin traded) ──
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    parts = []
    for ticker, grp in df.groupby('ticker', sort=False):
        g = grp.set_index('timestamp')
        agg = {}
        for col in g.columns:
            cl = col.lower()
            if cl == 'open':       agg[col] = 'first'
            elif cl == 'high':     agg[col] = 'max'
            elif cl == 'low':      agg[col] = 'min'
            elif 'volume' in cl:   agg[col] = 'sum'
            else:                  agg[col] = 'last'
        r = g.resample(resample_freq).agg(agg).dropna(subset=['close'])
        r['ticker'] = ticker
        parts.append(r.reset_index())
    df = pd.concat(parts, ignore_index=True).sort_values(['timestamp', 'ticker']).reset_index(drop=True)

    bars_per_day = 6
    mtf_windows = [(1, 'ret_4h', 0.02), (6, 'ret_1d', 0.05), (42, 'ret_7d', 0.15)]
    w_1d, w_7d = 6, 42

    asset_list = sorted(df['ticker'].unique().tolist())
    timestamps = sorted(df['timestamp'].unique())
    n_steps, n_assets = len(timestamps), len(asset_list)
    print(f"  {n_steps:,} four-hour bars, {n_assets} assets"
          + (f"; scaling ranges fitted on rows up to {fit_end}" if fit_end else ""))

    # ── per-asset features ──────────────────────────────────────────────────
    df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    df['returns'] = df.groupby('ticker')['close'].pct_change().fillna(0.0)
    for window, col, scale in mtf_windows:   # cumulative return over the window, tanh-compressed
        raw = df.groupby('ticker')['close'].pct_change(periods=window).fillna(0.0)
        df[col + '_norm'] = np.tanh(raw / scale).astype(np.float32)
    df['volatility'] = df.groupby('ticker')['returns'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()).fillna(0.0)

    lo, hi = _fit_range(df, 'commit_count_4_weeks', fit_end)
    rng = hi - lo
    df['commit_count_norm'] = np.where(rng > 0, (df['commit_count_4_weeks'] - lo) / rng, 0.5)
    df['utility_score'] = df['utility_score'].fillna(0.5).clip(0.0, 1.0)
    df['corr_btc_norm'] = ((df['corr_btc'] + 1.0) / 2.0).fillna(0.5).clip(0.0, 1.0)

    df['RSI_14_norm'] = (df['RSI_14'] / 100.0).fillna(0.5).clip(0.0, 1.0)
    df['MACD_norm'] = _scaled(df, 'MACD', fit_end, clip_pct, 0.0)
    df['BB_WIDTH_norm'] = _scaled(df, 'BB_WIDTH', fit_end, clip_pct, 0.0)
    df['ATR_NORM'] = df['ATR_NORM'].fillna(0.0).clip(0.0, 1.0)
    df['SMA_200_RATIO'] = df['SMA_200_RATIO'].fillna(1.0).clip(0.0, 2.0) / 2.0
    lo, hi = _fit_range(df, 'garch_volatility', fit_end, clip_pct)
    rng = hi - lo
    df['garch_volatility_norm'] = pd.Series(
        np.where(rng > 0, (df['garch_volatility'] - lo) / rng, 0.5), index=df.index).fillna(0.5).clip(0.0, 1.0)

    # ── macro inputs (one value per bar; fixed scaling bounds) ───────────────
    macro_ts = (df.sort_values('timestamp').groupby('timestamp')[MACRO_COLS].first()
                  .reindex(pd.Index(timestamps, name='timestamp')).ffill().bfill())
    macro = np.stack([
        np.clip((macro_ts['vix'].values - 10.0) / (80.0 - 10.0), 0, 1),
        np.clip((macro_ts['dxy'].values - 88.0) / (115.0 - 88.0), 0, 1),
        np.clip((macro_ts['tnx'].values - 0.5) / (5.5 - 0.5), 0, 1),
        np.clip(macro_ts['macro_trend_score'].values, 0, 1),
    ], axis=1).astype(np.float32)

    # ── wide [bar, asset] tables ───────────────────────────────────────────
    idx = pd.Index(timestamps, name='timestamp')
    cols = pd.Index(asset_list, name='ticker')

    def wide(col, fill):
        p = df.pivot(index='timestamp', columns='ticker', values=col).reindex(index=idx, columns=cols)
        return p.ffill().fillna(fill)

    returns = df.pivot(index='timestamp', columns='ticker', values='returns') \
                .reindex(index=idx, columns=cols).fillna(0.0)      # a missing bar has no return
    price = df.pivot(index='timestamp', columns='ticker', values='close').reindex(index=idx, columns=cols)

    # listing mask: tradable once a coin has 20 days of its own history (immediately if it
    # is present from the first bar)
    first = price.notna().values.argmax(axis=0)
    lag = np.where(first > 0, bars_per_day * 20, 0)
    avail = np.arange(n_steps)[:, None] >= (first + lag)[None, :]
    price = price.ffill().fillna(0.0)
    price_listed = price.where(price > 0)                            # NaN before listing

    # ── market-wide trend inputs ───────────────────────────────────────────
    btc = price['BTC']
    btc_1d = btc.pct_change(w_1d).fillna(0.0).values
    btc_7d = btc.pct_change(w_7d).fillna(0.0).values
    mkt_7d = price_listed.pct_change(w_7d, fill_method=None).mean(axis=1).fillna(0.0).values
    btc_sma = btc.rolling(window=1200, min_periods=100).mean().bfill()          # 200 days of 4h bars
    btc_trend = ((btc / btc_sma).clip(0.5, 1.5) - 0.5).values.astype(np.float32)  # 0.5 = at the SMA
    regime = np.stack([
        np.tanh(btc_1d / 0.05).astype(np.float32),
        np.tanh(btc_7d / 0.15).astype(np.float32),
        np.tanh(mkt_7d / 0.15).astype(np.float32),
        np.clip(btc_trend, 0.0, 1.0),
    ], axis=1)

    # each coin's own 200-day trend: a full 1,200-bar window, neutral (0.5) until available
    coin_trend = np.stack([
        ((price_listed[c] / price_listed[c].rolling(window=1200, min_periods=1200).mean())
         .clip(0.5, 1.5) - 0.5).fillna(0.5).values.astype(np.float32)
        for c in price.columns], axis=1)

    f32 = lambda p: p.values.astype(np.float32)
    data = {
        'returns': f32(returns),
        'volatility': f32(wide('volatility', 0.0)),
        'commit': f32(wide('commit_count_norm', 0.5)),
        'utility': f32(wide('utility_score', 0.5)),
        'price': f32(price),
        'corr': f32(wide('corr_btc_norm', 0.5)),
        'garch': f32(wide('garch_volatility_norm', 0.5)),
        'rsi': f32(wide('RSI_14_norm', 0.5)),
        'macd': f32(wide('MACD_norm', 0.0)),
        'bbw': f32(wide('BB_WIDTH_norm', 0.0)),
        'atr': f32(wide('ATR_NORM', 0.0)),
        'sma_ratio': f32(wide('SMA_200_RATIO', 1.0)),
        'ret_4h': f32(wide('ret_4h_norm', 0.0)),
        'ret_1d': f32(wide('ret_1d_norm', 0.0)),
        'ret_7d': f32(wide('ret_7d_norm', 0.0)),
        'macro': macro,
        'regime': regime,
        'coin_trend': coin_trend,
        'avail': avail,
        'asset_list': asset_list,
        'timestamps': timestamps,
        'n_steps': n_steps,
        'n_assets': n_assets,
        'bars_per_day': bars_per_day,
    }
    return data


def slice_data(data: Dict, start: int, end: int) -> Dict:
    """The same data restricted to bars [start, end)."""
    out = {k: (np.array(v[start:end]) if k in ARRAY_KEYS else v) for k, v in data.items()}
    out['timestamps'] = data['timestamps'][start:end]
    out['n_steps'] = end - start
    return out
