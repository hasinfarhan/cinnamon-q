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
            print("📊 Loading and optimizing crypto training data...")
            self._load_and_optimize()
        
        # Get unique assets
        self.assets = sorted(self.asset_list)
        self.n_assets = len(self.assets)
        
        print(f"✓ Optimized {self.n_steps} steps for {self.n_assets} assets: {self.assets}")
        
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
        self.calmar_cooldown_steps     = 96     # Minimum 1 day (96×15min) before checking recovery
        self.macro_panic_threshold     = 0.30   # Gate 2: macro_score < 0.30 → scale down

        # Gate 1 runtime state (reset on each episode)
        self.calmar_gate_active  = False
        self.calmar_cooldown_rem = 0

        # v9: Episode diversity — pre-compute bear/bull/neutral start indices
        # Regime label based on BTC 7d return (crypto_regime column 1)
        # Used in reset() to sample uniformly across market regimes
        self._regime_indices = {}
        if hasattr(self, 'has_crypto_regime') and self.has_crypto_regime:
            WARMUP  = 200
            MIN_EP  = 1000
            btc_7d  = self.crypto_regime_array[:, 1]
            for name, mask in [
                ('bear',    btc_7d < -0.3),
                ('bull',    btc_7d >  0.3),
                ('neutral', (btc_7d >= -0.3) & (btc_7d <= 0.3)),
            ]:
                idx = np.where(mask)[0]
                idx = idx[(idx >= WARMUP) & (idx < self.n_steps - MIN_EP)]
                self._regime_indices[name] = idx

        # Short position borrowing cost: 3% per year → per 15-min bar
        # Models crypto perpetual futures funding rate (typical for liquid coins)
        self.short_borrow_rate = 0.03 / (365 * 24 * 4)
        
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

        n_obs_features = (self.n_assets * n_features_per_asset) + 2 + n_macro_features + n_crypto_regime_features

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
        if has_crypto_regime:  global_info.append("crypto_regime×4")
        global_str = " + " + " + ".join(global_info) if global_info else ""
        print(f"  ✓ Observation space: {n_obs_features} features "
              f"({base_features_per_asset} base{feature_info} + {ta_features_per_asset} TA per asset{global_str})"
              f"  [v7: MTF returns + crypto regime]")
        
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
        v14 Brain 2 — Sortino Solver (The Risk Manager).

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

        Constraints enforced
        --------------------
        - Max total long  exposure : 80 %
        - Max total short exposure : 40 %
        - Max per-asset position   : 20 %
        - Cash                     : 1 − gross_exposure  (always ≥ 0)
        """
        alpha = np.clip(alpha_scores, -1.0, 1.0).astype(np.float64)

        # ── Step 1: Semicovariance matrix from recent 96-bar history ─────────
        W     = 96   # 1 trading day (96 × 15 min)
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

        # ── Step 3: Apply exposure constraints (v13: constant limits = low turnover) ─
        # v11 used btc_7d-linked limits here → limits changed every 15 min →
        # massive portfolio churn → ~72% of portfolio consumed by 0.1% fees.
        # Regime awareness lives in _apply_risk_gates() (SMA200, changes monthly).
        MAX_LONG      = 0.80
        MAX_SHORT     = 0.40
        MAX_PER_ASSET = 0.10   # v14: halved from 0.20 — AVAX at 20% cost -18% in v13

        longs  = np.clip(raw_w,  0.0, None)
        shorts = np.clip(raw_w, None, 0.0)

        long_sum  = float(longs.sum())
        short_sum = float(np.abs(shorts).sum())

        if long_sum  > 1e-8: longs  = longs  / long_sum  * MAX_LONG
        if short_sum > 1e-8: shorts = shorts / short_sum * MAX_SHORT

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

        # Macro features: VIX, DXY, TNX, macro_trend_score (traditional market context)
        if hasattr(self, 'has_macro') and self.has_macro and hasattr(self, 'macro_array'):
            obs[idx:idx + 4] = self.macro_array[self.current_step]
            idx += 4

        # v7: Crypto regime features (BTC 1d/7d, market 7d, BTC vs SMA200)
        # These directly signal crypto-specific bear/bull regimes the LSTM can act on
        if hasattr(self, 'has_crypto_regime') and self.has_crypto_regime:
            obs[idx:idx + 4] = self.crypto_regime_array[self.current_step]
            idx += 4

        return obs
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # v9 Episode Diversity — regime-stratified start sampling
        # Pick a regime uniformly at random (bear / bull / neutral each ~33%),
        # then pick a random start step within that regime.
        # This counteracts the natural imbalance in the data (~40% bull, ~25% bear)
        # so the LSTM sees equal bear/bull/neutral exposure during training.
        WARMUP = 200
        available = {k: v for k, v in self._regime_indices.items() if len(v) > 0}
        if available and self.n_steps > 2000:
            regime = np.random.choice(list(available.keys()))
            self.current_step = int(np.random.choice(available[regime]))
        elif self.n_steps > 2000:
            self.current_step = np.random.randint(WARMUP, self.n_steps - 1000)
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

        # Component 1: Asymmetric return (losses ×2.5)
        reward = r if r >= 0.0 else r * 2.5

        # Component 2: Proactive drawdown drag
        if self.peak_equity > 1e-8:
            current_drawdown = max(0.0,
                (self.peak_equity - float(self.portfolio_value)) / self.peak_equity)
        else:
            current_drawdown = 0.0
        reward -= current_drawdown * 0.01

        # Component 3: VIX-aware cash bonus
        if hasattr(self, 'has_macro') and self.has_macro and hasattr(self, 'macro_array'):
            macro_score = float(self.macro_array[self.current_step, 3])
            risk_off_signal = max(0.0, 0.4 - macro_score) / 0.4
            reward += float(self.current_weights[-1]) * risk_off_signal * 0.001

        # Component 4: Alpha direction accuracy bonus (v13)
        #
        # Core problem fixed: Brain 1 had a persistent long bias (alpha ≈ +0.4→+0.9
        # regardless of regime) because P&L outcome alone gives no directional signal.
        # This component directly rewards correct directional prediction:
        #   - Positive alpha on a rising asset   → +bonus
        #   - Positive alpha on a falling asset  → -penalty
        #   - Negative alpha on a falling asset  → +bonus  (teaches shorting!)
        #   - Negative alpha on a rising asset   → -penalty
        #
        # Weighted by |asset_return| so large moves contribute more than noise.
        # Scale 0.003 ≈ 30% of a typical step return — strong enough to override long bias.
        # v13 used 0.001 (too weak: shorts improved but BTC/ETH/AVAX longs persisted).
        # v14 triples the signal so the penalty for longing a falling major is felt clearly.
        if isinstance(asset_returns, np.ndarray) and asset_returns.sum() != 0:
            action_signals = action[:self.n_assets]  # Brain 1 alphas, shape (n_assets,)
            direction_correct = (np.sign(action_signals) == np.sign(asset_returns)).astype(np.float32)
            abs_returns = np.abs(asset_returns)
            total_abs = abs_returns.sum()
            if total_abs > 1e-10:
                weighted_accuracy = float(np.dot(direction_correct - 0.5, abs_returns) / total_abs)
            else:
                weighted_accuracy = 0.0
            reward += weighted_accuracy * 0.003

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
            'reward_type'        : 'asymmetric_bicameral_v14',
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

