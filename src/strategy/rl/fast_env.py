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
        
        # Action space: long/short weights for each asset + cash weight
        # Negative weight = short position, positive = long, last element = cash [0,1]
        # Constraint enforced in step(): sum(|asset_weights|) + cash = 1.0  (no leverage)
        self.action_space = spaces.Box(
            low=np.array([-1.0] * self.n_assets + [0.0], dtype=np.float32),
            high=np.array([ 1.0] * self.n_assets + [1.0], dtype=np.float32),
            dtype=np.float32
        )

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

        # v8: Rolling Sharpe buffer — stores last 96 step returns (= 1 trading day)
        # Used to compute rolling Sharpe ratio reward component.
        # 96 bars × 15 min = 24 hours: short enough to be responsive to crashes,
        # long enough to smooth out single-step noise.
        self._sharpe_window = 96
        self.return_buffer: list = []
    
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
        
        # Random start to prevent overfitting (skip first 200 bars for TA/GARCH warmup)
        WARMUP = 200
        if self.n_steps > 2000:
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

        # Reset DIR state — must restart per episode so EMA baseline matches new start point
        self.A_t = 0.0
        self.B_t = 0.0

        # Reset rolling Sharpe buffer
        self.return_buffer = []
        
        observation = self._get_observation()
        info = {
            'portfolio_value': float(self.portfolio_value),
            'step': int(self.current_step)
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step using optimized numpy operations.
        
        Args:
            action: Portfolio weights (shape: (n_assets + 1,))
        
        Returns:
            Standard gymnasium 5-tuple
        """
        # Ensure action is a numpy array and has correct shape
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # Flatten action if it's multi-dimensional
        action = action.flatten()
        
        # Ensure action has correct length
        expected_len = self.n_assets + 1
        if len(action) != expected_len:
            raise ValueError(
                f"Action shape mismatch: expected {expected_len} (n_assets={self.n_assets} + 1), "
                f"got {len(action)}"
            )
        
        # Normalise long/short action into a valid portfolio
        # Cash must be ≥ 0; asset weights can be negative (short)
        # Constraint: sum(|asset_weights|) + cash_weight = 1.0  (fully collateralised, no leverage)
        cash_raw   = float(np.clip(action[-1], 0.0, 1.0))
        assets_raw = action[:-1].astype(np.float32)          # may contain negatives

        gross = float(np.sum(np.abs(assets_raw)))
        total = gross + cash_raw

        if total > 1e-8:
            scale        = 1.0 / total
            asset_w      = (assets_raw * scale).astype(np.float32)
            cash_w       = float(cash_raw * scale)
        else:
            asset_w      = np.zeros(self.n_assets, dtype=np.float32)
            cash_w       = 1.0

        weights = np.concatenate([asset_w, [cash_w]]).astype(np.float32)
        
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
        if self.current_step < self.n_steps - 1:
            current_prices = self.price_array[self.current_step, :]
            next_prices = self.price_array[self.current_step + 1, :]
            
            # Avoid division by zero
            mask = current_prices > 1e-8
            asset_returns = np.zeros(self.n_assets, dtype=np.float32)
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
        
        # ── Reward: Asymmetric Risk-Adjusted Return + Rolling Sharpe (v8) ────
        #
        # Four components, each scaled to ~0.001/step (typical 15-min return):
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
        #
        #  4. Rolling Sharpe bonus  ← KEY NEW COMPONENT (v8)
        #     Reward = rolling Sharpe over the last 96 steps (1 trading day).
        #     A static portfolio has a FIXED Sharpe — it cannot improve it by
        #     doing the same thing every step.  A dynamic portfolio that goes
        #     short during crashes raises its rolling Sharpe → higher reward.
        #     This is the incentive the model was missing to trade dynamically.
        #     Scale: typical Sharpe ≈ ±0.1–0.4  →  reward contribution ≈ ±0.0002–0.0008/step
        #            (same order as the asymmetric return component)

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

        # Component 4: Rolling Sharpe bonus
        # Update the return buffer with this step's net return
        self.return_buffer.append(r)
        if len(self.return_buffer) > self._sharpe_window:
            self.return_buffer.pop(0)   # drop oldest entry (O(n) but n=96, negligible)

        if len(self.return_buffer) >= 20:  # need at least 20 steps for a stable estimate
            ret_arr   = np.array(self.return_buffer, dtype=np.float32)
            mu        = float(ret_arr.mean())
            sigma     = float(ret_arr.std()) + 1e-8
            rolling_sharpe = mu / sigma   # dimensionless: positive = profitable & consistent
            # Scale so Sharpe≈0.1 → +0.0002/step, Sharpe≈0.4 → +0.0008/step
            # This makes it competitive with but not dominant over the step-return component
            reward += rolling_sharpe * 0.002
        else:
            rolling_sharpe = 0.0

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
            'reward_type'      : 'asymmetric_rolling_sharpe',
            'reward'           : float(reward),
            'rolling_sharpe'   : float(rolling_sharpe) if 'rolling_sharpe' in dir() else 0.0,
        }
        
        return observation, reward, terminated, truncated, info

