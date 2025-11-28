"""
Technical Analysis Indicators - Deep Feature Engineering

This module provides centralized technical analysis feature engineering for both
training and inference. By keeping the logic in one place, we prevent "Training/Serving
Skew" - a common AI bug where features are calculated differently during training vs. inference.

All calculations are done per-ticker (groupby) to prevent data leakage between assets.

Note: We implement indicators manually (not using pandas-ta) for Python 3.11 compatibility.
"""

from typing import Optional

import numpy as np
import pandas as pd


class TechnicalAnalysis:
    """
    Centralized technical analysis feature engineering.
    
    This class provides professional-grade technical indicators that enrich
    the raw OHLCV data with signals that help the AI understand:
    - Momentum (RSI)
    - Trend (MACD)
    - Volatility (Bollinger Bands, ATR)
    - Market Regime (SMA Ratio)
    
    All calculations are done per-ticker to prevent data leakage.
    """
    
    def __init__(self):
        """Initialize the technical analysis calculator."""
        pass
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical analysis features to the DataFrame.
        
        This method enriches the input DataFrame with professional TA indicators:
        - RSI (14): Momentum indicator (overbought/oversold)
        - MACD (12, 26, 9): Trend and momentum changes
        - Bollinger Band Width: Volatility squeeze indicator
        - ATR (Normalized): True range volatility relative to price
        - SMA Ratio: Macro regime indicator (bull vs bear)
        
        **CRITICAL:** All calculations are done per-ticker (groupby) to prevent
        data leakage between assets. This ensures that RSI for BTC doesn't leak
        into ETH calculations.
        
        Args:
            df: DataFrame with OHLCV data.
                Required columns: 'ticker', 'open', 'high', 'low', 'close', 'volume'
                Optional: 'timestamp' (for sorting)
        
        Returns:
            DataFrame with added technical analysis columns:
            - RSI_14: Relative Strength Index (14 period)
            - MACD: MACD line
            - MACD_signal: MACD signal line
            - MACD_hist: MACD histogram
            - BB_WIDTH: Bollinger Band Width (normalized)
            - ATR_NORM: Normalized Average True Range
            - SMA_200_RATIO: Close price / SMA(200) ratio
        
        Raises:
            ValueError: If required columns are missing
        """
        # Validate required columns
        required_cols = ['ticker', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Make a copy to avoid modifying the original
        df_enriched = df.copy()
        
        # Ensure data is sorted by ticker and timestamp (if available)
        if 'timestamp' in df_enriched.columns:
            df_enriched = df_enriched.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
        else:
            df_enriched = df_enriched.sort_values('ticker').reset_index(drop=True)
        
        print("  Calculating technical indicators per ticker...")
        
        # Group by ticker and apply technical indicators
        # This prevents data leakage between assets
        ticker_groups = df_enriched.groupby('ticker')
        
        enriched_parts = []
        for ticker, group_df in ticker_groups:
            # Calculate indicators for this ticker only
            ticker_df = self._calculate_indicators(group_df.copy())
            enriched_parts.append(ticker_df)
        
        # Combine all tickers back together
        df_enriched = pd.concat(enriched_parts, ignore_index=True)
        
        # Sort by ticker and timestamp (if available)
        if 'timestamp' in df_enriched.columns:
            df_enriched = df_enriched.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
        else:
            df_enriched = df_enriched.sort_values('ticker').reset_index(drop=True)
        
        print(f"  ✓ Added {len(self._get_feature_columns())} technical indicators")
        
        return df_enriched
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a single ticker.
        
        This is an internal method that processes one ticker at a time.
        All calculations are done manually using pandas/numpy.
        
        Args:
            df: DataFrame for a single ticker with OHLCV data
        
        Returns:
            DataFrame with added indicator columns
        """
        # Ensure we have the required columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("Missing required OHLCV columns")
        
        # Extract OHLCV series
        open_price = df['open']
        high_price = df['high']
        low_price = df['low']
        close_price = df['close']
        volume = df['volume']
        
        # ============================================================
        # 1. RSI (14) - Momentum Indicator
        # ============================================================
        # RSI measures momentum: >70 = overbought, <30 = oversold
        df['RSI_14'] = self._calculate_rsi(close_price, period=14)
        
        # ============================================================
        # 2. MACD (12, 26, 9) - Trend and Momentum Changes
        # ============================================================
        # MACD provides three signals:
        # - MACD line: Fast EMA - Slow EMA
        # - Signal line: EMA of MACD line
        # - Histogram: MACD - Signal (momentum strength)
        macd, macd_signal, macd_hist = self._calculate_macd(close_price, fast=12, slow=26, signal=9)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd_hist
        
        # ============================================================
        # 3. Bollinger Band Width (BBW) - Volatility Squeeze Indicator
        # ============================================================
        # Bollinger Bands: Upper, Middle (SMA), Lower
        # BBW = (Upper - Lower) / Middle (normalized width)
        # High BBW = high volatility, Low BBW = volatility squeeze (predicts big moves)
        df['BB_WIDTH'] = self._calculate_bb_width(close_price, period=20, std_dev=2)
        
        # ============================================================
        # 4. ATR (Normalized) - True Range Volatility
        # ============================================================
        # ATR measures true range (high-low, high-prev_close, low-prev_close)
        # Normalized ATR = ATR / Close (relative volatility)
        atr = self._calculate_atr(high_price, low_price, close_price, period=14)
        df['ATR_NORM'] = atr / close_price
        
        # ============================================================
        # 5. SMA Ratio (Close / SMA200) - Macro Regime Indicator
        # ============================================================
        # SMA200 is a long-term trend indicator
        # Ratio > 1.0 = bull market, < 1.0 = bear market
        sma_200 = self._calculate_sma(close_price, period=200)
        df['SMA_200_RATIO'] = close_price / sma_200
        
        # ============================================================
        # Cleanup: Fill NaNs
        # ============================================================
        # Technical indicators often have NaNs at the beginning (need history)
        # Strategy: Forward fill first (carry last known value), then fill remaining with 0
        indicator_cols = self._get_feature_columns()
        for col in indicator_cols:
            if col in df.columns:
                # Forward fill (carry last known value forward)
                df[col] = df[col].ffill()
                # Fill any remaining NaNs with 0 (or neutral value for ratios)
                if 'RATIO' in col:
                    df[col] = df[col].fillna(1.0)  # Neutral ratio
                else:
                    df[col] = df[col].fillna(0.0)  # Zero for other indicators
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over period
        
        Args:
            prices: Series of closing prices
            period: RSI period (default: 14)
        
        Returns:
            Series of RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal EMA period (default: 9)
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bb_width(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Calculate Bollinger Band Width (normalized).
        
        BBW = (Upper Band - Lower Band) / Middle Band
        where:
        - Middle Band = SMA(period)
        - Upper Band = Middle + (std_dev * std)
        - Lower Band = Middle - (std_dev * std)
        
        Args:
            prices: Series of closing prices
            period: SMA period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
        
        Returns:
            Series of normalized Bollinger Band Width
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        # Normalized width
        bb_width = (upper_band - lower_band) / sma
        
        return bb_width
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        ATR = SMA(True Range, period)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ATR period (default: 14)
        
        Returns:
            Series of ATR values
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            prices: Series of prices
            period: SMA period
        
        Returns:
            Series of SMA values
        """
        return prices.rolling(window=period).mean()
    
    def _get_feature_columns(self) -> list:
        """
        Get the list of feature columns added by this module.
        
        Returns:
            List of feature column names
        """
        return [
            'RSI_14',
            'MACD',
            'MACD_signal',
            'MACD_hist',
            'BB_WIDTH',
            'ATR_NORM',
            'SMA_200_RATIO'
        ]
    
    def add_correlation(
        self,
        df: pd.DataFrame,
        benchmark_ticker: str = 'BTC',
        window: int = 96
    ) -> pd.DataFrame:
        """
        Add rolling correlation feature (correlation to benchmark asset).
        
        Phase 5.5 Sprint 2: The Correlation Eye
        This feature helps the AI identify hedging assets (like PAXG) that
        decouple from the broad market (BTC).
        
        Logic:
        - High Correlation (+0.9): High Beta. Good for Bull Runs.
        - Low/Negative Correlation (< 0.0): Hedge. Good for Bear Markets.
        
        The Brain's Lesson: "When Volatility is High AND Market Trend is Down
        -> Shift weight to assets with Low Correlation (PAXG)."
        
        Args:
            df: DataFrame in "Long" format with columns: 'timestamp', 'ticker', 'close'
            benchmark_ticker: Ticker to use as benchmark (default: 'BTC' for Bitcoin)
            window: Rolling window for correlation calculation (default: 96 periods)
                   For 15-minute bars, 96 = 24 hours (1 day)
        
        Returns:
            DataFrame with added 'corr_btc' column (rolling correlation to benchmark)
        """
        # Make a copy to avoid modifying the original
        df_enriched = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_enriched['timestamp']):
            df_enriched['timestamp'] = pd.to_datetime(df_enriched['timestamp'])
        
        # Ensure data is sorted by timestamp and ticker
        df_enriched = df_enriched.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
        
        print(f"  Calculating rolling correlation to {benchmark_ticker} (window={window})...")
        
        # Pivot to Wide format: timestamp x ticker (close prices)
        # This allows us to calculate correlations across assets
        pivoted = df_enriched.pivot(index='timestamp', columns='ticker', values='close')
        
        # Calculate percentage returns (needed for correlation)
        returns = pivoted.pct_change(fill_method=None).dropna()
        
        # Check if benchmark exists
        if benchmark_ticker not in returns.columns:
            # Try alternative formats (BTC/USD -> BTC, etc.)
            benchmark_variants = [
                benchmark_ticker,
                benchmark_ticker.replace('/USD', ''),
                f"{benchmark_ticker}/USD",
                'BTC',
                'BTC/USD'
            ]
            benchmark_returns = None
            for variant in benchmark_variants:
                if variant in returns.columns:
                    benchmark_returns = returns[variant]
                    benchmark_ticker = variant  # Use the found variant
                    break
            
            if benchmark_returns is None:
                print(f"  ⚠ Warning: Benchmark ticker '{benchmark_ticker}' not found in data.")
                print(f"  Available tickers: {list(returns.columns)[:10]}...")
                print(f"  Setting correlation to 1.0 (assume correlated)")
                df_enriched['corr_btc'] = 1.0
                return df_enriched
        else:
            benchmark_returns = returns[benchmark_ticker]
        
        # Calculate rolling correlation for each asset against benchmark
        # We need to calculate correlation for each asset separately
        corr_dict = {}
        
        for ticker in returns.columns:
            if ticker == benchmark_ticker:
                # Benchmark is always perfectly correlated with itself
                corr_dict[ticker] = pd.Series(1.0, index=returns.index)
            else:
                # Calculate rolling correlation for this asset against benchmark
                asset_returns = returns[ticker]
                
                # Calculate rolling correlation manually (pandas rolling.corr() doesn't work with two series)
                # We need to compute correlation for each rolling window
                corr_values = []
                for i in range(len(returns)):
                    if i < window - 1:
                        corr_values.append(np.nan)
                    else:
                        # Get window of data
                        asset_window = asset_returns.iloc[i - window + 1:i + 1]
                        benchmark_window = benchmark_returns.iloc[i - window + 1:i + 1]
                        
                        # Calculate correlation
                        if len(asset_window) < 2 or asset_window.std() == 0 or benchmark_window.std() == 0:
                            corr_values.append(np.nan)
                        else:
                            corr = asset_window.corr(benchmark_window)
                            corr_values.append(corr if not pd.isna(corr) else np.nan)
                
                corr_dict[ticker] = pd.Series(corr_values, index=returns.index)
        
        # Convert to DataFrame and melt back to Long format
        corr_df = pd.DataFrame(corr_dict, index=returns.index)
        corr_df.index.name = 'timestamp'
        corr_df = corr_df.reset_index()
        
        # Melt to Long format: timestamp, ticker, corr_btc
        corr_long = corr_df.melt(
            id_vars=['timestamp'],
            var_name='ticker',
            value_name='corr_btc'
        )
        
        # Merge back into original DataFrame
        df_enriched = df_enriched.merge(
            corr_long,
            on=['timestamp', 'ticker'],
            how='left'
        )
        
        # Ensure corr_btc column exists (in case merge didn't create it)
        if 'corr_btc' not in df_enriched.columns:
            df_enriched['corr_btc'] = 1.0  # Default: assume correlated
        else:
            # Fill NaNs: Use 1.0 as conservative default (assume correlated if unknown)
            # This is safe because:
            # - Early history (before window periods): Assume correlated
            # - Missing data: Assume correlated (conservative)
            df_enriched['corr_btc'] = df_enriched['corr_btc'].fillna(1.0)
        
        # Clip to valid correlation range [-1, 1] (safety check)
        df_enriched['corr_btc'] = df_enriched['corr_btc'].clip(-1.0, 1.0)
        
        print(f"  ✓ Added corr_btc column (correlation to {benchmark_ticker})")
        print(f"    Correlation range: [{df_enriched['corr_btc'].min():.3f}, {df_enriched['corr_btc'].max():.3f}]")
        
        return df_enriched
    
    @staticmethod
    def get_training_feature_columns() -> list:
        """
        Get the list of TA feature columns used in training (normalized versions).
        
        This is the SINGLE SOURCE OF TRUTH for feature selection.
        Both training and inference must use this exact list to prevent "Brain Scramble".
        
        Returns:
            List of normalized feature column names in training order
        """
        return [
            'RSI_14_norm',      # RSI normalized to 0-1
            'MACD_norm',        # MACD normalized per asset
            'BB_WIDTH_norm',    # Bollinger Band Width normalized per asset
            'ATR_NORM',         # Already normalized
            'SMA_200_RATIO'     # Already a ratio
        ]
