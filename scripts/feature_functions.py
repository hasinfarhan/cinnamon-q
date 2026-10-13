"""
Technical indicators used to build the dataset (scripts/03_build_dataset.py).

Each function takes pandas Series of one asset's 15-minute candles and returns a Series
aligned with the input. Kept as plain functions so the dataset build is self-contained.
"""
import numpy as np
import pandas as pd


def _rsi(close, period=14):
    """Relative strength index (14 periods), from rolling mean gains and losses; 50 where undefined."""
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _macd(close, fast=12, slow=26):
    """MACD line: EMA(12) minus EMA(26) of the close (divided by the close in scripts/03_build_dataset.py)."""
    return close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()


def _bb_width(close, period=20, std_dev=2.0):
    """Bollinger Band width: (upper - lower) / middle band, 20 periods, 2 standard deviations."""
    sma   = close.rolling(window=period).mean()
    std   = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return ((upper - lower) / sma.replace(0, np.nan)).fillna(0.0)


def _atr_norm(high, low, close, period=14):
    """Average true range (14 periods) divided by the close, clipped to [0, 1]."""
    prev_close = close.shift(1)
    tr  = pd.concat([high - low,
                     (high - prev_close).abs(),
                     (low  - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return (atr / close.replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)


def _sma_ratio(close, period=200):
    """Close divided by its 200-period simple moving average (200 x 15 min = 50 hours)."""
    sma = close.rolling(window=period).mean()
    return (close / sma.replace(0, np.nan)).fillna(1.0)


def _corr_btc(returns, btc_returns, window=96):
    """Rolling correlation with Bitcoin returns over 96 periods (one day of 15-minute bars)."""
    return returns.rolling(window=window, min_periods=10).corr(btc_returns).fillna(0.0)


def _ewma_vol(returns, span=24):
    """Exponentially weighted volatility of returns (span 24), a RiskMetrics-type GARCH proxy."""
    return returns.ewm(span=span, adjust=False).std().fillna(0.0)

