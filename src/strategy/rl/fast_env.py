"""
Fast Crypto RL Environment - Optimized Numpy Version

This is a high-performance version of CryptoDeepEnv that uses Numpy arrays
instead of Pandas DataFrames for blazingly fast stepping. Designed for
parallel training with SubprocVecEnv.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class FastCryptoEnv(gym.Env):
    """
    Highly optimized crypto trading environment using Numpy arrays.
    
    This version pre-processes all data into float32 arrays for maximum
    performance during training. Designed for parallel execution.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        data_config: Optional[Dict] = None,
        lookback_window: int = 30,
        initial_cash: float = 100000.0,
        fee_rate: float = 0.001,  # Phase 5.5 Sprint 1: Transaction fees (0.1%)
        benchmark_returns: Optional[np.ndarray] = None  # Phase 6 Sprint 2: DIR benchmark
    ):
        """
        Initialize the fast crypto trading environment.
        
        Args:
            data_path: Path to crypto_training_data.parquet (used if data_config is None)
            data_config: Preprocessed data configuration with keys:
                - 'returns_path': Path to returns_array.npy (memory-mapped)
                - 'volatility_path': Path to volatility_array.npy (memory-mapped)
                - 'commit_path': Path to commit_array.npy (memory-mapped)
                - 'utility_path': Path to utility_array.npy (memory-mapped)
                - 'price_path': Path to price_array.npy (memory-mapped)
                - 'asset_list': List of asset symbols
                - 'timestamps': List of timestamps
                - 'n_steps': Number of timesteps
                - 'n_assets': Number of assets
            lookback_window: Number of days for Sortino calculation
            initial_cash: Starting portfolio value
        """
        super().__init__()
        
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate  # Phase 5.5 Sprint 1: Transaction fees
        
        # Phase 6 Sprint 2: Differential Information Ratio (DIR) - Benchmark-aware reward
        self.benchmark_returns = benchmark_returns
        if benchmark_returns is not None:
            self.benchmark_returns = np.array(benchmark_returns, dtype=np.float32)
        else:
            # Fallback: zero benchmark (no DIR penalty)
            self.benchmark_returns = None
        
        # Phase 6 Sprint 2: DIR State Variables (Exponential Moving Averages)
        self.A_t = 0.0  # Running mean of active return (First Moment)
        self.B_t = 0.0  # Running second moment of active return (Variance)
        self.eta = 0.01  # Learning/decay rate for EMA (small = slow adaptation)

        # Phase 5.6 Sprint 1: Fortress Brain - Composite Reward Coefficients
        self.sortino_coeff = 0.5  # Penalty for downside volatility
        self.calmar_coeff = 2.0  # Penalty for drawdown (quadratic)

        # Phase 5.6 Sprint 1: Track peak equity (high water mark) for Calmar calculation
        self.peak_equity = 0.0

        # v27: bars per trading day — set here as default, overridden from data_config below.
        # Used to scale borrow rate and cooldown steps to actual calendar time regardless
        # of bar frequency (15-min=96, 4h=6, 1h=24).
        self.bars_per_day = 96

        # Use memory-mapped files if provided, otherwise load from parquet
        if data_config is not None:
            # Lightweight initialization: load from memory-mapped files (zero-copy)
            # mmap_mode='r' means read-only, shared memory access
            self.returns_array = np.load(data_config['returns_path'], mmap_mode='r')
            self.volatility_array = np.load(data_config['volatility_path'], mmap_mode='r')
            self.commit_array = np.load(data_config['commit_path'], mmap_mode='r')
            self.utility_array = np.load(data_config['utility_path'], mmap_mode='r')
            self.price_array = np.load(data_config['price_path'], mmap_mode='r')
            # Phase 5.5 Sprint 2: Load correlation array (only if present)
            self.has_correlation = data_config.get('has_correlation', False)
            if self.has_correlation:
                self.corr_array = np.load(data_config['corr_path'], mmap_mode='r')
            
            # Phase 6 Sprint 1: Load GARCH volatility array (only if present)
            self.has_garch = data_config.get('has_garch', False)
            if self.has_garch:
                self.garch_array = np.load(data_config['garch_path'], mmap_mode='r')

            # Macro features: VIX, DXY, TNX, macro_trend_score (global context)
            self.has_macro = data_config.get('has_macro', False)
            if self.has_macro and 'macro_path' in data_config:
                self.macro_array = np.load(data_config['macro_path'], mmap_mode='r')
            else:
                self.has_macro = False

            # v7: Multi-timeframe returns per asset (4h / 1d / 7d)
            self.has_mtf = data_config.get('has_mtf', False)
            if self.has_mtf:
                self.ret_4h_array = np.load(data_config['ret_4h_path'], mmap_mode='r')
                self.ret_1d_array = np.load(data_config['ret_1d_path'], mmap_mode='r')
                self.ret_7d_array = np.load(data_config['ret_7d_path'], mmap_mode='r')

            # v7: Crypto regime features (BTC 1d/7d return, market 7d return, BTC vs SMA200)
            self.has_crypto_regime = 'crypto_regime_path' in data_config
            if self.has_crypto_regime:
                self.crypto_regime_array = np.load(data_config['crypto_regime_path'], mmap_mode='r')

            # Phase 6 Sprint 2: Load benchmark returns for DIR calculation
            if 'benchmark_path' in data_config:
                benchmark_array = np.load(data_config['benchmark_path'], mmap_mode='r')
                self.benchmark_returns = benchmark_array
            elif benchmark_returns is not None:
                # Use provided benchmark_returns (for backward compatibility)
                self.benchmark_returns = np.array(benchmark_returns, dtype=np.float32)
            else:
                # Fallback: zero benchmark (no DIR)
                self.benchmark_returns = None
            
            self.asset_list = data_config['asset_list']
            self.timestamps = data_config['timestamps']
            self.n_steps = data_config['n_steps']
            self.has_ta_features = data_config.get('has_ta_features', False)
            # v26: Gate 3 is inference-only. During training (default True) the gate
            # is SKIPPED so bear/bull windows receive full P&L gradient. Set False
            # in backtest data_config to activate the gate at inference time.
            self.training_mode = data_config.get('training_mode', True)
            # v27: bar frequency — 96 for 15-min (default), 6 for 4h
            self.bars_per_day  = data_config.get('bars_per_day', 96)
            
            # Load TA arrays if present
            if self.has_ta_features:
                self.rsi_array = np.load(data_config['rsi_path'], mmap_mode='r')
                self.macd_array = np.load(data_config['macd_path'], mmap_mode='r')
                self.bbw_array = np.load(data_config['bbw_path'], mmap_mode='r')
                self.atr_array = np.load(data_config['atr_path'], mmap_mode='r')
                self.sma_ratio_array = np.load(data_config['sma_ratio_path'], mmap_mode='r')
            else:
                # Create zero arrays for backward compatibility
                self.rsi_array = np.zeros((self.n_steps, len(self.asset_list)), dtype=np.float32)
                self.macd_array = np.zeros((self.n_steps, len(self.asset_list)), dtype=np.float32)
                self.bbw_array = np.zeros((self.n_steps, len(self.asset_list)), dtype=np.float32)
                self.atr_array = np.zeros((self.n_steps, len(self.asset_list)), dtype=np.float32)
                self.sma_ratio_array = np.ones((self.n_steps, len(self.asset_list)), dtype=np.float32)
        else:
            # Legacy mode: load from file (for backward compatibility)
            if data_path is None:
                project_root = Path(__file__).parent.parent.parent.parent
                data_path = project_root / "data" / "processed" / "crypto_training_data.parquet"
            
            self.data_path = data_path
            print("[*] Loading and optimizing crypto training data...")
            self._load_and_optimize()
        
        # Get unique assets
        self.assets = sorted(self.asset_list)
        self.n_assets = len(self.assets)

        # v38: Cyclical time features — hour/day-of-week/month as sin+cos pairs.
        # Lets the LSTM learn: weekend liquidity dips, monthly options expiry,
        # 8h funding rate cycles (every 2 bars at 4h), quarterly rebalancing.
        ts_index = pd.to_datetime(self.timestamps)
        _h = ts_index.hour.values.astype(np.float32)
        _d = ts_index.dayofweek.values.astype(np.float32)
        _m = (ts_index.month.values - 1).astype(np.float32)  # 0-indexed so Jan=0
        self.time_array = np.stack([
            np.sin(2 * np.pi * _h / 24).astype(np.float32),
            np.cos(2 * np.pi * _h / 24).astype(np.float32),
            np.sin(2 * np.pi * _d / 7).astype(np.float32),
            np.cos(2 * np.pi * _d / 7).astype(np.float32),
            np.sin(2 * np.pi * _m / 12).astype(np.float32),
            np.cos(2 * np.pi * _m / 12).astype(np.float32),
        ], axis=1)

        print(f"[OK] Optimized {self.n_steps} steps for {self.n_assets} assets: {self.assets}")
        
        # v9 BICAMERAL — Brain 1 outputs Alpha Scores per asset in [-1, +1]
        #   +1 = maximum long conviction (expect strong positive return)
        #   -1 = maximum short conviction (expect strong negative return)
        #    0 = neutral / no edge detected
        #   Cash is NOT part of the action — Brain 2 (Sortino Solver) computes it.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # v10: Hard risk gate thresholds
        self.calmar_threshold          = 0.15   # Gate 1 fires  at 15% drawdown
        self.calmar_recovery_threshold = 0.07   # Gate 1 resets at  7% drawdown (hysteresis)
        # v27: cooldown = 1 trading day regardless of bar frequency
        self.calmar_cooldown_steps     = self.bars_per_day
        # v16: recalibrated from 0.30 → 0.50. macro_trend_score actual range is
        # [0.37, 0.73] — the old threshold of 0.30 is BELOW the minimum possible
        # value so the gate has never fired once. 0.50 fires ~6.5% of steps.
        self.macro_panic_threshold     = 0.50   # Gate 2: macro_score < 0.50 → scale down

        # Gate 1 runtime state (reset on each episode)
        self.calmar_gate_active  = False
        self.calmar_cooldown_rem = 0

        # v38: 50d/200d SMA crossover regime labeling — diagnostic only, not used for sampling.
        #
        # Sampling is uniform random (reset() below). Labels are computed here for
        # diagnostic display only so we can verify the data distribution looks right.
        #
        # Why crossover instead of single sma_200:
        #   sma_200 alone lags regime transitions by 4-7 months (missed all of 2022-Q1
        #   bear and wrongly labeled 2023 recovery as bear). The 50d/200d crossover
        #   (death cross / golden cross) catches regime changes within 4-6 weeks and
        #   correctly identified 2022-Q1 as BEAR and 2023-Q1 recovery as neutral.
        #
        # Why NOT two-stage equal-weight sampling:
        #   Neutral is only 9.5% of bars (brief crossover transition windows). Equal
        #   33/33/33 sampling would oversample neutral 3.5x and distort training.
        #   Uniform random gives the natural 55% bull / 35% bear / 9% neutral — which
        #   reflects real market conditions and gives Brain 1 ~3,100 genuine bear bars
        #   (full 2022 crash + 2025 decline) without any artificial weighting.
        self._regime_indices = {}
        if hasattr(self, 'price_array') and hasattr(self, 'asset_list'):
            try:
                WARMUP = 200
                MIN_EP = 1000
                valid  = np.arange(WARMUP, self.n_steps - MIN_EP)

                btc_idx   = list(self.asset_list).index('BTC')
                btc_price = pd.Series(self.price_array[:, btc_idx].astype(np.float64))
                sma_50d   = btc_price.rolling(300,  min_periods=50).mean()
                sma_200d  = btc_price.rolling(1200, min_periods=100).mean()
                ratio     = (sma_50d / sma_200d).fillna(1.0).values

                bull_mask    = ratio > 1.02
                bear_mask    = ratio < 0.98
                neutral_mask = ~bull_mask & ~bear_mask

                for name, mask in [('bull', bull_mask), ('bear', bear_mask), ('neutral', neutral_mask)]:
                    idx = valid[mask[valid]]
                    self._regime_indices[name] = idx
                    print(f"  v38 crossover regime [{name:7s}]: {len(idx):,} valid start indices")
            except (ValueError, Exception):
                pass  # BTC not in asset list or price_array unavailable — skip

        # v39: SMA10 signal (C4) and SMA50 signal (C6)
        # SMA10 = 60-bar (10-day at 4h) — catches regime shifts in ~1-2 weeks → C4 direct alpha gradient
        # SMA50 = 300-bar (50-day at 4h) — catches regime shifts in ~3-4 weeks → C6 weight penalty
        # Stored as price/SMA - 1.0, clipped to ±0.30 to prevent extreme values in volatile periods.
        self.sma10_signal = np.zeros(self.n_steps, dtype=np.float32)
        self.sma50_signal = np.zeros(self.n_steps, dtype=np.float32)
        if hasattr(self, 'price_array') and hasattr(self, 'asset_list'):
            try:
                _btc_idx   = list(self.asset_list).index('BTC')
                _btc_price = pd.Series(self.price_array[:, _btc_idx].astype(np.float64))
                _sma10     = _btc_price.rolling(60,  min_periods=10).mean()
                _sma50     = _btc_price.rolling(300, min_periods=50).mean()
                self.sma10_signal = np.nan_to_num(np.clip(
                    (_btc_price.values / _sma10.values - 1.0).astype(np.float32), -0.30, 0.30), nan=0.0)
                self.sma50_signal = np.nan_to_num(np.clip(
                    (_btc_price.values / _sma50.values - 1.0).astype(np.float32), -0.30, 0.30), nan=0.0)
                print(f"  v39 SMA10 signal: mean={float(np.nanmean(self.sma10_signal)):.3f}  std={float(np.nanstd(self.sma10_signal)):.3f}")
                print(f"  v39 SMA50 signal: mean={float(np.nanmean(self.sma50_signal)):.3f}  std={float(np.nanstd(self.sma50_signal)):.3f}")
            except (ValueError, Exception):
                pass

        # Short position borrowing cost: 3% per year → per bar (scales with bar frequency)
        # Models crypto perpetual futures funding rate (typical for liquid coins)
        self.short_borrow_rate = 0.03 / (365 * self.bars_per_day)
        
        # Define observation space DYNAMICALLY
        # Base features per asset: [return, volatility, commit_count_norm, utility_score] = 4
        # Phase 5.5 Sprint 2: Added corr_btc_norm (rolling correlation to BTC) = +1 (optional)
        # TA features per asset (if present): [RSI, MACD, BBW, ATR, SMA_RATIO] = 5
        # Plus: portfolio_value_norm, cash_weight = 2
        
        # Backward compatibility: Dynamically calculate base features
        # Base features: returns, volatility, commit_count, utility_score
        # Optional: corr_btc (Phase 5.5), garch_volatility (Phase 6)
        has_corr         = hasattr(self, 'has_correlation')   and self.has_correlation
        has_garch        = hasattr(self, 'has_garch')          and self.has_garch
        has_macro        = hasattr(self, 'has_macro')          and self.has_macro
        has_mtf          = hasattr(self, 'has_mtf')            and self.has_mtf
        has_crypto_regime= hasattr(self, 'has_crypto_regime')  and self.has_crypto_regime

        # Per-asset feature count
        # base: returns(1) + commit(1) + utility(1) = 3
        base_features_per_asset = 3
        if has_corr:  base_features_per_asset += 1   # corr_btc
        if has_garch: base_features_per_asset += 1   # GARCH volatility
        ta_features_per_asset  = 5 if (hasattr(self, 'has_ta_features') and self.has_ta_features) else 0
        mtf_features_per_asset = 3 if has_mtf else 0  # v7: ret_4h, ret_1d, ret_7d
        n_features_per_asset   = base_features_per_asset + ta_features_per_asset + mtf_features_per_asset

        # Global feature count
        n_macro_features        = 4 if has_macro         else 0   # VIX, DXY, TNX, macro_score
        n_crypto_regime_features= 4 if has_crypto_regime else 0   # v7: BTC1d, BTC7d, Mkt7d, SMA200
        # v20: current per-asset weights added to observation so Brain 1 can see
        # its own position book directly (previously only saw total cash + PV).
        n_position_features     = self.n_assets                    # one weight per asset

        n_time_features = 6  # hour_sin/cos, dow_sin/cos, month_sin/cos
        n_obs_features = (self.n_assets * n_features_per_asset) + 2 + n_macro_features + n_crypto_regime_features + n_position_features + n_time_features

        self.n_features_per_asset = n_features_per_asset  # stored for obs construction

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_obs_features,),
            dtype=np.float32
        )

        # Build feature info string
        feature_parts = []
        if has_corr:   feature_parts.append("corr")
        if has_garch:  feature_parts.append("garch")
        if has_mtf:    feature_parts.append("4h/1d/7d")
        feature_info   = f" ({', '.join(feature_parts)})" if feature_parts else ""
        global_info    = []
        if has_macro:          global_info.append("macro×4")
        if has_crypto_regime:  global_info.append("crypto_regime*4")
        global_info.append(f"positions*{self.n_assets}")
        global_info.append("time*6")
        global_str = " + " + " + ".join(global_info) if global_info else ""
        print(f"  [OK] Observation space: {n_obs_features} features "
              f"({base_features_per_asset} base{feature_info} + {ta_features_per_asset} TA per asset{global_str})"
              f"  [v38: +cyclical time features]")
        
        # Initialize episode state
        self.current_step = 0
        self.portfolio_value = initial_cash
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0  # Start with 100% cash

        # Peak equity for drawdown calculation
        self.peak_equity = float(initial_cash)

    
    def _load_and_optimize(self):
        """
        Load data and convert to optimized numpy arrays.
        
        This pre-processes everything into float32 arrays for maximum speed.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}\n"
                "Please run scripts/09_generate_sentiment.py first."
            )
        
        df = pd.read_parquet(self.data_path)
        df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
        
        # Get unique assets and timestamps
        self.asset_list = sorted(df['ticker'].unique().tolist())
        self.timestamps = sorted(df['timestamp'].unique())
        self.n_steps = len(self.timestamps)
        n_assets = len(self.asset_list)
        
        # Pre-allocate numpy arrays for maximum speed
        # Shape: [n_steps, n_assets, n_features]
        # Features: [returns, volatility, commit_count_norm, utility_score, corr_btc_norm]
        # Phase 5.5 Sprint 2: Added correlation feature
        n_features = 5  # Now 5 base features (was 4)
        
        # Data arrays
        self.returns_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        self.volatility_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        self.commit_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        self.utility_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        # Phase 5.5 Sprint 2: Correlation array (default to neutral 0.5 = corr=0)
        self.corr_array = np.full((self.n_steps, n_assets), 0.5, dtype=np.float32)
        
        # Price arrays for portfolio return calculation
        self.price_array = np.zeros((self.n_steps, n_assets), dtype=np.float32)
        
        # Process each asset
        for asset_idx, asset in enumerate(self.asset_list):
            asset_data = df[df['ticker'] == asset].sort_values('timestamp').reset_index(drop=True)
            
            # Calculate returns
            asset_data['returns'] = asset_data['close'].pct_change().fillna(0.0)
            
            # Calculate volatility (rolling std)
            asset_data['volatility'] = asset_data['returns'].rolling(window=7, min_periods=1).std().fillna(0.0)
            
            # Normalize commit_count
            commit_min = asset_data['commit_count_4_weeks'].min()
            commit_max = asset_data['commit_count_4_weeks'].max()
            if commit_max > commit_min:
                asset_data['commit_count_norm'] = (
                    (asset_data['commit_count_4_weeks'] - commit_min) / (commit_max - commit_min)
                )
            else:
                asset_data['commit_count_norm'] = 0.5
            
            # Ensure utility_score is valid
            asset_data['utility_score'] = asset_data['utility_score'].fillna(0.5).clip(0.0, 1.0)
            
            # Map to timestamps
            for ts_idx, timestamp in enumerate(self.timestamps):
                ts_data = asset_data[asset_data['timestamp'] == timestamp]
                if len(ts_data) > 0:
                    row = ts_data.iloc[0]
                    self.returns_array[ts_idx, asset_idx] = float(row['returns'])
                    self.volatility_array[ts_idx, asset_idx] = float(row['volatility'])
                    self.commit_array[ts_idx, asset_idx] = float(row['commit_count_norm'])
                    self.utility_array[ts_idx, asset_idx] = float(row['utility_score'])
                    # Phase 5.5 Sprint 2: Load correlation feature (default to 0.5 if not present)
                    self.corr_array[ts_idx, asset_idx] = float(row.get('corr_btc_norm', 0.5))
                    self.price_array[ts_idx, asset_idx] = float(row['close'])
    
    def _construct_portfolio(self, alpha_scores: np.ndarray) -> np.ndarray:
        """
        v16 Brain 2 — Sortino Solver (The Risk Manager).

        Converts Brain 1's alpha scores into portfolio weights by maximising
        the Sortino Ratio using a semicovariance matrix.

        v12 lesson: regime-aware limits on Brain 2 caused 14× more portfolio
        turnover (btc_7d is noisy → MAX_LONG oscillated wildly → 0.1% fee ×
        high turnover = 72% costs).
        v13 lesson: SMA200 gate also caused oscillation near its threshold
        → ~155% fee drag.  Regime awareness now handled solely by Brain 1 via
        the alpha direction accuracy reward, which teaches directional skill
        without any hard-coded threshold that can oscillate.
        v14 lesson: alpha accuracy scale 0.001 was too weak to flip BTC/ETH/AVAX
        negative. Tripled to 0.003. MAX_PER_ASSET halved from 0.20 → 0.10 to
        prevent single-asset concentration blowing up the portfolio.
        v15 lesson: 0.003 was too strong — Brain 1 saturated at -1.0 in all
        regimes ("short everything"). Fix: scale→0.0015, entropy bonus to
        penalise saturation, loss aversion 2.5→1.75 to allow neutral stance.
        v16 lesson: binary accuracy reward has base-rate exploit (75% non-bull
        training steps → "always short" wins). Fix: alpha-weighted return signal,
        bull oversampling 3×, macro thresholds recalibrated, drawdown drag halved,
        entropy bonus 4×.

        Constraints enforced
        --------------------
        - Max total long  exposure : 80 %
        - Max total short exposure : 40 %
        - Max per-asset position   : 20 %
        - Cash                     : 1 − gross_exposure  (always ≥ 0)
        """
        alpha = np.clip(alpha_scores, -1.0, 1.0).astype(np.float64)

        # ── Step 1: Semicovariance matrix from recent history ─────────
        # v29: use 20 trading days (not 1). At 4h bars, bars_per_day=6 → W was only
        # 6 observations for a 14×14 matrix (196 params) — completely underdetermined.
        # 20 days = 120 bars gives a statistically valid semicovariance estimate.
        W = getattr(self, 'bars_per_day', 96) * 20
        t     = self.current_step
        start = max(0, t - W)
        hist  = self.returns_array[start:t].astype(np.float64)  # (W, n_assets)

        if len(hist) >= 10:
            # Downside returns only: set positive returns to 0
            downside  = np.minimum(hist, 0.0)                   # (W, n_assets)
            semi_cov  = (downside.T @ downside) / len(hist)     # (n, n)
            # Tikhonov regularisation — ensures invertibility.
            # Use 10% of the mean diagonal as shrinkage (Ledoit-Wolf inspired)
            # to prevent matrix inverse from blowing up on noisy early signals.
            shrinkage = max(1e-4, 0.10 * float(np.diag(semi_cov).mean()))
            semi_cov += shrinkage * np.eye(self.n_assets)

            # ── Step 2: Sortino-optimal weights ──────────────────────────────
            try:
                semi_cov_inv = np.linalg.inv(semi_cov)
                raw_w = semi_cov_inv @ alpha
            except np.linalg.LinAlgError:
                # Fallback: diagonal Sortino (per-asset downside vol scaling)
                semi_vols = np.sqrt(np.maximum(np.diag(semi_cov), 1e-8))
                raw_w     = alpha / semi_vols
        else:
            # Not enough history yet — fall back to per-asset downside vol
            vol   = self.volatility_array[t].astype(np.float64)
            raw_w = alpha / np.clip(vol, 1e-8, None)

        # ── Step 2.5: Direction-lock REMOVED (v38) ──────────────────────────────
        # v33 direction-lock was blocking Brain 1 from learning negative BTC alpha.
        # Mechanism: lock zeroed any long in bear mode → BTC position = 0 → zero
        # P&L gradient for BTC alpha → Brain 1 never penalised for bullish BTC in
        # a crash → BTC alpha stuck at +0.66 throughout the entire 2025-2026 bear.
        # The SMA regime signal is still in Brain 1's observation (crypto_regime[:,3])
        # so it can learn the direction itself and get punished when wrong.

        # ── Step 3: Apply exposure constraints (v19: conviction-scaled) ────────
        # v13–v18 bug: normalisation always rescaled to exactly MAX_LONG / MAX_SHORT
        # regardless of alpha magnitude.  alpha=-1.0 and alpha=-0.1 produced
        # identical 0.40 short books — Brain 1 only needed to get the SIGN right
        # so it saturated at ±1.0 with zero incentive to vary magnitude.
        #
        # Fix (v19): scale total gross exposure by mean |alpha| conviction.
        #   alpha = ±1.0 for all assets → full MAX_LONG / MAX_SHORT exposure
        #   alpha = ±0.5 for all assets → half exposure
        #   alpha ≈  0.0              → near-zero exposure (model chooses flat)
        #
        # Brain 1's output magnitude now directly affects portfolio P&L, so
        # gradients flow correctly: wrong-direction at high conviction costs
        # more than wrong-direction at low conviction.  The model learns to
        # go high-conviction in the correct direction and near-zero when unsure.
        MAX_LONG      = 0.80
        MAX_SHORT     = 0.60   # v29: raised from 0.40 → enables 10%+ returns in bear
        MAX_PER_ASSET = 0.10

        # Mean absolute conviction ∈ [0, 1]
        mean_conviction = float(np.mean(np.abs(alpha)))

        longs  = np.clip(raw_w,  0.0, None)
        shorts = np.clip(raw_w, None, 0.0)

        long_sum  = float(longs.sum())
        short_sum = float(np.abs(shorts).sum())

        # Scale aggregate exposure by conviction — magnitude now matters
        if long_sum  > 1e-8: longs  = longs  / long_sum  * (MAX_LONG  * mean_conviction)
        if short_sum > 1e-8: shorts = shorts / short_sum * (MAX_SHORT * mean_conviction)

        longs  = np.clip(longs,   0.0,           MAX_PER_ASSET)
        shorts = np.clip(shorts, -MAX_PER_ASSET,  0.0)

        weights = (longs + shorts).astype(np.float32)
        gross   = float(np.abs(weights).sum())
        cash    = float(np.clip(1.0 - gross, 0.0, 1.0))

        return np.concatenate([weights, [cash]]).astype(np.float32)

    def _apply_risk_gates(self, weights: np.ndarray) -> np.ndarray:
        """
        Hard risk gates — override Brain 2 weights when market conditions
        are extreme.  These run AFTER the Sortino Solver and cannot be
        overridden by the RL model.

        Gate 1 — Calmar Circuit Breaker (with hysteresis + recovery)
        -------------------------------------------------------------
        FIRES  : drawdown > 15% → force 100% cash + start 96-step cooldown
        RESETS : cooldown expired AND drawdown < 7% → resume normal trading
        Hysteresis (different trigger vs recovery thresholds) prevents rapid
        oscillation in and out of cash when near the 15% boundary.

        Gate 2 — Macro Panic Gate
        --------------------------
        If the macro_trend_score falls below 0.30 (extreme global fear),
        all positions are scaled down proportionally.  The lower the score,
        the more defensive the portfolio becomes (min 10% of normal size).

        Gate 3 — Regime Uncertainty Gate (v25)
        ----------------------------------------
        When BTC 7-day momentum is in neutral (|btc_7d| ≤ 0.3), the regime
        is ambiguous — no reliable bull or bear trend to exploit. Positions
        are tapered linearly to zero as btc_7d approaches zero.

        This is a HARD ARCHITECTURAL CONSTRAINT, not a reward signal.
        Previous attempts (v22–v24) tried to teach neutral=cash via reward
        shaping (cash bonuses, position penalties). All failed because:
          - Cash bonus → created "cash mode" that bled into bull periods (v22)
          - Position penalty → C4 pushed model LONG in neutral instead (v23)
          - Weak cash bonus → LSTM bear/bull memory overwhelmed it (v24)
        The reward signal cannot reliably force a specific portfolio state
        because the LSTM carries regime memory across steps. A hard gate in
        Brain 2 enforces the constraint architecturally without touching the
        reward function or creating any training bias.

        Scale: |btc_7d| / 0.3
          btc_7d =  0.00 → scale = 0.00 → 100% cash (full neutral)
          btc_7d = ±0.15 → scale = 0.50 → 50% of normal positions
          btc_7d = ±0.30 → scale = 1.00 → full positions restored
          btc_7d > ±0.30 → gate inactive → Brain 2 runs unconstrained
        """
        weights = weights.copy()

        # ── Gate 1: Calmar Circuit Breaker ───────────────────────────────────
        if self.peak_equity > 1e-8:
            drawdown = (self.peak_equity - float(self.portfolio_value)) / self.peak_equity

            if self.calmar_gate_active:
                # Gate is ON — count down cooldown and check recovery condition
                self.calmar_cooldown_rem = max(0, self.calmar_cooldown_rem - 1)
                recovered = (self.calmar_cooldown_rem == 0 and
                             drawdown < self.calmar_recovery_threshold)
                if recovered:
                    self.calmar_gate_active = False   # resume trading
                else:
                    weights[:]  = 0.0
                    weights[-1] = 1.0                 # stay 100% cash
                    return weights

            if not self.calmar_gate_active and drawdown > self.calmar_threshold:
                # Gate fires: new drawdown breach
                # Reset peak to CURRENT value so recovery is measured from here,
                # not from all-time high (otherwise cash earns 0% and gate never releases)
                self.calmar_gate_active  = True
                self.calmar_cooldown_rem = self.calmar_cooldown_steps
                self.peak_equity         = float(self.portfolio_value)
                weights[:]  = 0.0
                weights[-1] = 1.0
                return weights

        # ── Gate 2: Macro Panic Gate ──────────────────────────────────────────
        # v13: SMA200 gate removed — it caused rapid oscillation near the 0.5
        # threshold (price/SMA200 bounces above/below each 15-min bar) which
        # created ~155% transaction costs in v12. Brain 1's alpha accuracy
        # reward now teaches directional skill directly instead.
        if hasattr(self, 'has_macro') and self.has_macro and hasattr(self, 'macro_array'):
            macro_score = float(self.macro_array[self.current_step, 3])
            if macro_score < self.macro_panic_threshold:
                # Linear scale: macro=0.0 → 10% of positions, macro=0.30 → 100%
                scale = max(0.10, macro_score / self.macro_panic_threshold)
                weights[:-1] *= scale
                gross         = float(np.abs(weights[:-1]).sum())
                weights[-1]   = float(np.clip(1.0 - gross, 0.0, 1.0))

        # ── Gate 3: Regime Uncertainty Gate (v26 — SMA200, hard zero, inference-only) ──
        # SKIPPED during training (self.training_mode=True, the default).
        # v25 lesson: gate during training silenced neutral sub-steps inside bear/bull
        # windows → model lost bear-SHORT learning (BTC_alpha -0.974→+0.270) → -57%.
        # Fix: gate fires ONLY at inference (training_mode=False in backtest data_config).
        #
        # Signal change: btc_7d → btc_sma200_norm (col 3).
        # v26 preview showed btc_7d gate still lost even in "bear" weeks (-9.12% contrib)
        # because btc_7d is lagging: gate fires AFTER the crash, SHORT enters at bottom
        # as market rebounds. SMA200 is structural: when BTC is below 200-day MA, the
        # TREND is bearish for weeks — gate stays OFF, model stays SHORT throughout.
        #
        # Gate logic (hard zero, not linear taper):
        #   sma200_norm ∈ (0.475, 0.525): BTC within ±5% of SMA200 → transitioning → CASH
        #   sma200_norm outside that band: clear bull/bear trend → model trades freely
        #
        # Linear taper was insufficient: at sma200=0.49 (just inside neutral), taper
        # scale = (0.49-0.475)/(0.525-0.475) = 0.3 → still 30% positions → still loses.
        # Hard zero guarantees zero exposure during regime transitions.
        if not getattr(self, 'training_mode', True):
            if hasattr(self, 'crypto_regime_array') and self.current_step < len(self.crypto_regime_array):
                sma200_norm = float(self.crypto_regime_array[self.current_step, 3])
                if abs(sma200_norm - 0.5) <= 0.025:   # within ±5% of SMA200
                    weights[:-1] = 0.0
                    weights[-1]  = 1.0
                    return weights

        return weights

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector using fast numpy operations.
        
        Phase 5.1: Now includes TA features if available.
        Feature order (per asset): [returns, volatility, commit_count, utility, RSI, MACD, BBW, ATR, SMA_RATIO]
        
        Returns:
            Observation vector as float32 numpy array
        """
        if self.current_step >= self.n_steps:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        idx = 0
        
        # For each asset, extract features (vectorized)
        has_ta = hasattr(self, 'has_ta_features') and self.has_ta_features
        
        for asset_idx in range(self.n_assets):
            # Base features: returns, commit_count, utility_score (3 features)
            # NOTE: 7-bar rolling volatility dropped — GARCH below is a better signal
            obs[idx]     = self.returns_array[self.current_step, asset_idx]
            obs[idx + 1] = self.commit_array[self.current_step, asset_idx]
            obs[idx + 2] = self.utility_array[self.current_step, asset_idx]
            idx += 3
            
            # Phase 5.5 Sprint 2: Correlation feature (optional, for backward compatibility)
            has_corr = hasattr(self, 'has_correlation') and self.has_correlation
            if has_corr:
                obs[idx] = self.corr_array[self.current_step, asset_idx] if hasattr(self, 'corr_array') else 0.5
                idx += 1  # Add correlation feature
            
            # Phase 6 Sprint 1: GARCH volatility feature (optional, for backward compatibility)
            has_garch = hasattr(self, 'has_garch') and self.has_garch
            if has_garch:
                obs[idx] = self.garch_array[self.current_step, asset_idx] if hasattr(self, 'garch_array') else 0.5
                idx += 1  # Add GARCH volatility feature
            
            # TA features (if present)
            if has_ta:
                obs[idx]     = self.rsi_array[self.current_step, asset_idx]
                obs[idx + 1] = self.macd_array[self.current_step, asset_idx]
                obs[idx + 2] = self.bbw_array[self.current_step, asset_idx]
                obs[idx + 3] = self.atr_array[self.current_step, asset_idx]
                obs[idx + 4] = self.sma_ratio_array[self.current_step, asset_idx]
                idx += 5

            # v7: Multi-timeframe returns (4h, 1d, 7d) — explicit trend signal
            if hasattr(self, 'has_mtf') and self.has_mtf:
                obs[idx]     = self.ret_4h_array[self.current_step, asset_idx]
                obs[idx + 1] = self.ret_1d_array[self.current_step, asset_idx]
                obs[idx + 2] = self.ret_7d_array[self.current_step, asset_idx]
                idx += 3

        # Add portfolio value (normalized) + cash weight
        obs[idx]     = self.portfolio_value / self.initial_cash
        obs[idx + 1] = self.current_weights[-1]
        idx += 2

        # v20: per-asset position weights — Brain 1 can now see its own book.
        # Without this, the LSTM had to infer its own positions from memory
        # across random episode starts, making it harder to learn to flip direction.
        # Weights are in [-MAX_PER_ASSET, +MAX_PER_ASSET] = [-0.10, +0.10].
        obs[idx:idx + self.n_assets] = self.current_weights[:-1]
        idx += self.n_assets

        # Macro features: VIX, DXY, TNX, macro_trend_score (traditional market context)
        if hasattr(self, 'has_macro') and self.has_macro and hasattr(self, 'macro_array'):
            obs[idx:idx + 4] = self.macro_array[self.current_step]
            idx += 4

        # v7: Crypto regime features (BTC 1d/7d, market 7d, BTC vs SMA200)
        # These directly signal crypto-specific bear/bull regimes the LSTM can act on
        if hasattr(self, 'has_crypto_regime') and self.has_crypto_regime:
            obs[idx:idx + 4] = self.crypto_regime_array[self.current_step]
            idx += 4

        # v38: Cyclical time features (hour_sin/cos, dow_sin/cos, month_sin/cos)
        obs[idx:idx + 6] = self.time_array[self.current_step]
        idx += 6

        return obs
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # v38: Uniform random sampling — no regime weighting.
        # Two-stage equal-weight sampling (v17) oversampled neutral (9.5% of bars
        # inflated to 33%) and undersampled bull/bear. Uniform random reflects actual
        # market distribution: ~55% bull, ~35% bear, ~9% neutral. Data has enough
        # bear bars (full 2022 crash + 2025 decline) without artificial oversampling.
        # _regime_indices are still computed in __init__ for diagnostic display only.
        WARMUP = 200
        MIN_EP  = 1000
        if self.n_steps > WARMUP + MIN_EP:
            self.current_step = np.random.randint(WARMUP, self.n_steps - MIN_EP)
        else:
            self.current_step = WARMUP
        
        # Reset episode state
        self.portfolio_value = self.initial_cash
        self.current_weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.current_weights[-1] = 1.0
        # Phase 5.5: Track previous weights for turnover calculation
        self.prev_weights = self.current_weights.copy()

        # Phase 5.6 Sprint 1: Reset peak equity to initial capital
        self.peak_equity = float(self.initial_cash)

        # v10: Reset Calmar Gate state for new episode
        self.calmar_gate_active  = False
        self.calmar_cooldown_rem = 0

        # Reset DIR state — must restart per episode so EMA baseline matches new start point
        self.A_t = 0.0
        self.B_t = 0.0

        observation = self._get_observation()
        info = {
            'portfolio_value': float(self.portfolio_value),
            'step': int(self.current_step)
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step using optimized numpy operations.

        v9 BICAMERAL:
            action  — Brain 1 output: shape (n_assets,), values in [-1, +1]
                      Each value is a directional signal, NOT a portfolio weight.
            weights — Brain 2 output: _construct_portfolio() converts signals
                      to risk-parity portfolio weights + cash residual.
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        action = action.flatten()[:self.n_assets]   # Brain 1: n_assets signals

        # Brain 2: Sortino Solver — converts alpha scores → portfolio weights
        weights = self._construct_portfolio(action)
        # Hard risk gates: Calmar circuit breaker + Macro panic gate
        weights = self._apply_risk_gates(weights)
        self.current_weights = weights
        
        # Check termination
        if self.current_step >= self.n_steps - 1:
            observation = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                'portfolio_value': float(self.portfolio_value),
                'step': int(self.current_step),
                'final_return': float((self.portfolio_value / self.initial_cash) - 1.0)
            }
            return observation, reward, terminated, truncated, info
        
        # Calculate portfolio return (vectorized numpy)
        asset_returns = np.zeros(self.n_assets, dtype=np.float32)  # default: no returns
        if self.current_step < self.n_steps - 1:
            current_prices = self.price_array[self.current_step, :]
            next_prices = self.price_array[self.current_step + 1, :]

            # Avoid division by zero
            mask = current_prices > 1e-8
            asset_returns[mask] = (next_prices[mask] / current_prices[mask]) - 1.0

            # Portfolio return = dot product of weights and returns
            portfolio_return = np.dot(weights[:-1], asset_returns)  # Exclude cash weight
        else:
            portfolio_return = 0.0
        
        # Transaction cost: fee on every weight change (long or short)
        turnover = np.sum(np.abs(weights - self.prev_weights))
        transaction_cost = turnover * self.fee_rate

        # Short borrow cost: ongoing per-step cost for all short positions
        short_exposure = float(np.sum(np.abs(np.minimum(weights[:-1], 0.0))))
        short_borrow_cost = short_exposure * self.short_borrow_rate

        # Net portfolio return after all costs
        net_portfolio_return = portfolio_return - transaction_cost - short_borrow_cost
        
        # Update portfolio value (using net return after fees)
        self.portfolio_value *= (1.0 + net_portfolio_return)
        
        # Phase 5.6 Sprint 1: Update peak equity (high water mark)
        # Peak equity can only increase, never decrease
        self.peak_equity = max(self.peak_equity, float(self.portfolio_value))
        
        # Store current weights for next step's turnover calculation
        self.prev_weights = weights.copy()
        
        # ── Reward: Asymmetric Risk-Adjusted Return (v7) ────────────────────
        #
        # Three components, each scaled to ~0.001/step (typical 15-min return):
        #
        #  1. Asymmetric portfolio return
        #     Gains feel normal; losses sting 2.5× harder.
        #     → Strong pressure to avoid any losing step.
        #
        #  2. Proactive drawdown drag (linear)
        #     Every step underwater costs 1% of current drawdown.
        #     → Punishes staying invested during sustained bear moves.
        #
        #  3. VIX-aware cash bonus
        #     Bonus for holding cash during macro risk-off.

        r = float(net_portfolio_return)

        # Compute current drawdown first (used by Components 1 and 2)
        if self.peak_equity > 1e-8:
            current_drawdown = max(0.0,
                (self.peak_equity - float(self.portfolio_value)) / self.peak_equity)
        else:
            current_drawdown = 0.0

        # Component 1: Drawdown-triggered loss aversion (v20)
        #
        # v19 bug: applying 1.75× to EVERY negative bar created short bias.
        # Math: in a 52/48 bull market, long positions (56% exposure) lose on
        # 48% of bars while short positions (28% exposure) lose on 52% of bars.
        # The smaller short size nearly cancels the higher loss frequency, making
        # short ≈ as cheap as long in moderate bull markets. Brain 1 correctly
        # learned to stay short because the reward said so.
        #
        # Fix: only fire the 1.75× when ALREADY in a sustained drawdown.
        # → Long in bull: portfolio trends up, drawdown ≈ 0 → losses treated normally
        # → Short in bull: portfolio trends down, drawdown builds → all losses hit 1.75×
        # This preserves loss aversion for sustained wrong-direction positions while
        # not penalising individual noise bars in an otherwise winning position.
        if r < 0.0 and current_drawdown > 0.005:
            reward = r * 1.35   # in drawdown AND losing: amplified penalty (v38: 1.75->1.35, less long bias)
        else:
            reward = r          # gaining or not in drawdown: symmetric

        # Component 2: Proactive drawdown drag (v16: halved 0.01 → 0.005)
        # v15 diagnosis: this dominated the reward (-12.97 vs -6.71 for returns).
        # Brain 1 learned to short to MINIMISE DRAWDOWN rather than maximise returns.
        # Halving rebalances so return signal is the primary driver.
        reward -= current_drawdown * 0.005

        # Component 3: VIX-aware cash bonus (v32: REMOVED — was competing with C6)
        # v31 diagnosis: C3 rewarded cash during bear/risk-off periods (cash×risk_off×0.001).
        # This directly competed with C6's goal to get the model SHORT in bear, reinforcing
        # the "cash = safe" local optimum. Combined with C5 saturation, it locked Brain 1
        # at alpha=+0.99 in cash mode. Removing C3 eliminates the bleed signal.
        # (Code kept for reference — coefficient zeroed out)
        if hasattr(self, 'has_macro') and self.has_macro and hasattr(self, 'macro_array'):
            macro_score = float(self.macro_array[self.current_step, 3])
            risk_off_signal = max(0.0, 0.52 - macro_score) / 0.52
            reward += float(self.current_weights[-1]) * risk_off_signal * 0.0  # v32: disabled

        # Component 4: Alpha regime-alignment reward (v34 — replaces bar-return version)
        #
        # v18–v33 used mean(alpha × sign(next_4h_return)) × 0.003.
        # Diagnosis: signal was too noisy to teach regime sensitivity.
        #   Even in a strong bear, 40–45% of 4h bars go UP (dead-cat bounces).
        #   Net C4 signal in bear for wrong-direction alpha ≈ −0.0006/step — nearly invisible.
        #   Brain 1 was learning to predict 4h candle direction, not structural regime.
        #
        # v34 fix: measure alpha direction against SMA200 structural regime instead.
        #   • Fires on ALPHAS directly — no credit-assignment gap through Brain 2
        #   • Direction-lock cannot shield Brain 1 from this signal (lock runs after Brain 2)
        #   • Same SMA200 signal as C6 and direction-lock — fully consistent
        #   • Scaled by |sma_signal| so stronger regimes give stronger gradient
        #
        #   bear (sma=-0.15), BTC_alpha=+0.98 (wrong):
        #     mean(+0.98 × -1) × 0.15 × 0.005 = -0.00074/step  ← direct penalty to Brain 1
        #   bear (sma=-0.15), BTC_alpha=-0.98 (correct):
        #     mean(-0.98 × -1) × 0.15 × 0.005 = +0.00074/step  ← direct reward to Brain 1
        #
        # Coefficient 0.005 (slightly above old 0.003) to compensate for C6's signal
        # being partially decoupled from Brain 1 by the direction-lock (v33).
        # Component 4: Per-asset alpha regime-alignment reward (v38)
        # v34-v37 used mean(alpha * regime_dir) — Brain 1 gamed it by making 13 alts
        # strongly negative while keeping BTC positive, averaging out to a "correct" mean.
        # BTC alpha was +0.66 in a crash and C4 never penalized it because the mean
        # across all 14 assets was still negative (dominated by the alts).
        #
        # Fix (mirrors C6 structure): reward each aligned asset, penalize each wrong one.
        # Now BTC alpha=+0.66 in bear gets: +0.66 * (-1) = -0.66 → wrong bucket → direct penalty.
        # Brain 1 can no longer hide BTC's wrong direction behind the other 13 assets.
        # v40: C4 reverts to SMA200 (crypto_regime col 3) — stays clearly negative for months
        # in genuine bear markets, giving sustained directional gradient. SMA10/SMA50 were too
        # noisy (flipped positive on every 1-2 week bounce, causing bull bias in training).
        # mean/mean fix kept from v39 (removes 14x asymmetry between correct and wrong).
        action_signals = action[:self.n_assets]
        if hasattr(self, 'crypto_regime_array') and self.current_step < len(self.crypto_regime_array):
            _sma_norm_c4 = float(self.crypto_regime_array[self.current_step, 3])
            _sma_sig_c4  = _sma_norm_c4 - 0.5
            if abs(_sma_sig_c4) > 0.025:
                _regime_dir_c4 = np.sign(_sma_sig_c4)
                _per_asset_c4  = action_signals * _regime_dir_c4   # [14]: +aligned, -wrong
                _correct_c4    = np.maximum(_per_asset_c4, 0.0)
                _wrong_c4      = np.minimum(_per_asset_c4, 0.0)
                reward += abs(_sma_sig_c4) * float(np.mean(_correct_c4)) * 0.005
                reward += abs(_sma_sig_c4) * float(np.mean(_wrong_c4))   * 0.005

        # Component 5: Alpha entropy regulariser — DROPPED in v35
        # Conviction scaling in Brain 2 already creates the right gradient: wrong
        # direction at high conviction costs more, so Brain 1 learns selectivity
        # without an explicit suppression penalty. C5 was collapsing mean_conviction
        # to ~0.23 (gross exposure ~14%). Removing it restores natural alpha
        # magnitudes and ~50% exposure, matching v19's working configuration.
        # alpha_entropy = -float(np.mean(action_signals ** 2))
        # reward += alpha_entropy * 0.0005

        # Component 6: Per-asset regime-alignment reward (v31)
        #
        # v27 bug: fired on Brain 1 alpha signals (range ±1), not on actual portfolio weights.
        #   Brain 1 learned to output 5L+9S mixed alphas → earn partial C6 reward on alphas
        #   → Brain 2 Sortino solver cancels mixed signals → near-zero weights → cash.
        #   Model gamed the reward: correct INTENTION (alphas) rewarded, zero EXECUTION (weights).
        #
        # Fix: fire C6 on self.current_weights[:n_assets] — the actual Brain 2 portfolio
        # weights — so the reward measures what the model IS DOING, not what Brain 1 signals.
        # Weights are in [-MAX_PER_ASSET, +MAX_PER_ASSET] = [-0.10, +0.10] per asset, vs
        # alphas in [-1, +1], so coefficients are scaled up ~10× to preserve reward magnitude.
        #
        #   asset_w_i = self.current_weights[i]   (Brain 2 output, after risk gates)
        #   per_asset = asset_w × regime_dir       (+positive = aligned, -negative = wrong)
        #   correct   = max(per_asset, 0)
        #   wrong     = min(per_asset, 0)
        #
        #   C6 = |sma| × mean(correct) × 2.0    ← reward actual aligned exposure
        #      + |sma| × sum(wrong)    × 2.0    ← penalise each wrong-direction asset (v32: 0.20→2.0)
        #
        # v32: wrong penalty strengthened from ×0.20 to ×2.0 (10×).
        # Diagnosis: with ×0.20, the wrong penalty was too weak relative to C1 returns
        # earned during bull training windows (33% of curriculum). The LSTM learned
        # "cash is safe" because going to cash removed the C6 penalty without needing
        # to flip direction. With ×2.0, being wrong in a clear regime costs:
        #   bear, all-long: 0.15 × sum(14 × 0.057) × 2.0 = -0.239/step (severe)
        # vs cash at 0 — the asymmetry now strongly favours going SHORT over staying in cash.
        #
        # In bear (sma=0.15), ALL SHORT (each asset weight ≈ -0.043 = 0.60/14):
        #   correct: 0.15 × mean(0.043) × 2.0 = +0.0129/step
        # In bear (sma=0.15), ALL LONG (each asset weight ≈ +0.057 = 0.80/14):
        #   wrong:   0.15 × sum(-14×0.057) × 2.0 = -0.239/step  ← severe, not cashable
        # In cash (all weights=0): C6 = 0 — cash no longer the comfortable middle ground
        # v40: C6 reverts to SMA200 (crypto_regime col 3) — same reason as C4 above.
        # SMA200 stays deeply negative for months in real bear markets (2022 crash: ~-0.30),
        # giving strong sustained penalty for wrong-direction positions.
        # mean/mean fix kept from v39 (was np.sum for wrong — 14x asymmetry that caused cash-hiding).
        if hasattr(self, 'crypto_regime_array') and self.current_step < len(self.crypto_regime_array):
            sma200_norm = float(self.crypto_regime_array[self.current_step, 3])
            sma_signal  = sma200_norm - 0.5
            if abs(sma_signal) > 0.025:
                regime_dir = np.sign(sma_signal)
                asset_w    = self.current_weights[:self.n_assets]   # Brain 2 weights
                per_asset  = asset_w * regime_dir    # +aligned, -wrong
                correct    = np.maximum(per_asset, 0.0)
                wrong      = np.minimum(per_asset, 0.0)
                reward += abs(sma_signal) * float(np.mean(correct)) * 2.0
                reward += abs(sma_signal) * float(np.mean(wrong))   * 2.0

        # Keep benchmark return for logging (no longer used in reward)
        if self.benchmark_returns is not None and self.current_step < len(self.benchmark_returns):
            r_benchmark = float(self.benchmark_returns[self.current_step])
        else:
            r_benchmark = 0.0
        r_active = r - r_benchmark
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Check termination
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        long_exposure  = float(np.sum(np.maximum(weights[:-1], 0.0)))
        info = {
            'portfolio_value'  : float(self.portfolio_value),
            'peak_equity'      : float(self.peak_equity),
            'current_drawdown' : float(current_drawdown),
            'portfolio_return' : float(net_portfolio_return),
            'benchmark_return' : float(r_benchmark),
            'active_return'    : float(r_active),
            'gross_return'     : float(portfolio_return),
            'transaction_cost' : float(transaction_cost),
            'short_borrow_cost': float(short_borrow_cost),
            'turnover'         : float(turnover),
            'long_exposure'    : long_exposure,
            'short_exposure'   : short_exposure,
            'cash_weight'      : float(weights[-1]),
            'step'             : int(self.current_step),
            'reward_type'        : 'asymmetric_bicameral_v16',
            'reward'             : float(reward),
            # Risk gate status (diagnostic)
            'calmar_gate_active' : self.calmar_gate_active,
            'calmar_cooldown_rem': self.calmar_cooldown_rem,
            'macro_gate_active'  : (
                hasattr(self, 'has_macro') and self.has_macro and
                hasattr(self, 'macro_array') and
                float(self.macro_array[self.current_step, 3]) < self.macro_panic_threshold
            ),
        }
        
        return observation, reward, terminated, truncated, info

