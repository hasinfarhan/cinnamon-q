"""
PPO Strategy - RL Inference Engine

This module provides the PPOStrategy class that loads a trained PPO model
and generates target portfolios from live market data.

Critical: This class handles normalization drift by loading and applying
the VecNormalize statistics from training to ensure the agent sees data
in the same format it was trained on.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO  # LSTM support
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from execution.models import TargetPortfolio
from features.indicators import TechnicalAnalysis
from strategy.optimizer.sortino_solver import SortinoRatioOptimizer
from strategy.risk.semicovariance import SemicovarianceCalculator
from strategy.rl.fast_env import FastCryptoEnv
from strategy.validator.risk import CalmarRiskGate


def load_asset_universe(universe_csv_path: Optional[Path] = None) -> list:
    """
    Load asset universe from crypto_universe.csv.
    
    This ensures the asset list matches the training data structure.
    Assets are sorted alphabetically to match the training environment.
    
    Args:
        universe_csv_path: Path to crypto_universe.csv (default: auto-detect)
    
    Returns:
        List of asset symbols sorted alphabetically (matching training order)
    """
    if universe_csv_path is None:
        # Auto-detect path relative to project root
        project_root = Path(__file__).parent.parent.parent
        universe_csv_path = project_root / "data" / "raw" / "crypto_universe.csv"
    
    if not universe_csv_path.exists():
        raise FileNotFoundError(
            f"Universe CSV not found: {universe_csv_path}\n"
            "Please ensure data/raw/crypto_universe.csv exists."
        )
    
    # Load CSV and extract alpaca_symbol column
    df = pd.read_csv(universe_csv_path)
    
    if 'alpaca_symbol' not in df.columns:
        raise ValueError(
            f"Universe CSV missing 'alpaca_symbol' column: {universe_csv_path}"
        )
    
    # Extract and sort alphabetically (CRITICAL: matches training data order)
    # Training data uses sorted(df['ticker'].unique().tolist())
    asset_list = sorted(df['alpaca_symbol'].dropna().unique().tolist())
    
    return asset_list


class PPOStrategy:
    """
    PPO-based strategy that uses a trained reinforcement learning agent
    to generate target portfolio allocations.
    
    This class handles the critical normalization step: the agent was trained
    on normalized data (via VecNormalize), so live data must be normalized
    using the same statistics before prediction.
    """
    
    def __init__(
        self,
        model_path: Path,
        vec_norm_path: Path,
        data_path: Path,
        asset_list: Optional[list] = None,
        provider: Optional[any] = None,
        optimizer_config: Optional[Dict] = None,
        trading_engine: Optional[any] = None  # Phase 5.6: For Calmar Gate
    ):
        """
        Initialize the PPO strategy.
        
        Args:
            model_path: Path to trained PPO model (.zip file)
            vec_norm_path: Path to VecNormalize statistics (.pkl file)
            data_path: Path to training data (parquet file) - used for structure/fallback
            asset_list: List of asset tickers (default: auto-detect from data)
            provider: Optional ExecutionProvider for fetching live data (if None, uses static data_path)
        
        Raises:
            FileNotFoundError: If model or normalization files don't exist
        """
        self.data_path = data_path
        self.model_path = model_path
        self.vec_norm_path = vec_norm_path
        self.provider = provider  # For live data fetching
        
        # Verify files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not vec_norm_path.exists():
            raise FileNotFoundError(f"Normalization stats not found: {vec_norm_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"📦 Loading RecurrentPPO (LSTM) model from: {model_path}")
        
        # Load the trained RecurrentPPO model (supports both PPO and RecurrentPPO)
        # RecurrentPPO.load() can load both types, but we'll try RecurrentPPO first
        try:
            self.model = RecurrentPPO.load(str(model_path))
            self.is_lstm = True
            print(f"✓ RecurrentPPO (LSTM) model loaded - Brain has MEMORY!")
        except Exception:
            # Fallback: Try loading as regular PPO (for backward compatibility)
            from stable_baselines3 import PPO
            self.model = PPO.load(str(model_path))
            self.is_lstm = False
            print(f"✓ PPO (MLP) model loaded - No memory (backward compatibility)")
        
        # Initialize LSTM states for stateful inference (if LSTM model)
        if self.is_lstm:
            # LSTM states: (hidden_state, cell_state) for each LSTM layer
            # Shape depends on policy_kwargs['lstm_hidden_size'] and 'n_lstm_layers'
            # For now, initialize to None - will be set on first prediction
            self.lstm_states = None
            self.episode_starts = np.array([True])  # Start of episode flag
            print(f"  LSTM state management initialized")
        
        # Load asset universe - CRITICAL: Must match training data exactly
        # The model was trained on a specific set of assets, so we must use the same set
        # Priority: 1) Provided asset_list, 2) Training data assets, 3) Universe CSV (filtered)
        if asset_list is None:
            # First, try to get assets from the training data (most accurate)
            # This ensures we use exactly the assets the model was trained on
            try:
                # Load a sample of the training data to get the asset list
                df_sample = pd.read_parquet(data_path, columns=['ticker'])
                training_assets = sorted(df_sample['ticker'].unique().tolist())
                
                # Convert to Alpaca format (add /USD if needed)
                # Training data uses base symbols like "BTC", but we need "BTC/USD" for Alpaca
                self.asset_list = []
                for asset in training_assets:
                    if '/' not in asset and not asset.endswith('USD'):
                        self.asset_list.append(f"{asset}/USD")
                    else:
                        self.asset_list.append(asset)
                
                # Ensure sorted (matching training order)
                self.asset_list = sorted(self.asset_list)
                
                print(f"✓ Loaded {len(self.asset_list)} assets from training data")
                print(f"  Assets: {self.asset_list}")
            except Exception as e:
                # Fallback: Try universe CSV, but this may have more assets than training
                print(f"⚠ Warning: Could not load assets from training data: {e}")
                try:
                    project_root = Path(__file__).parent.parent.parent
                    universe_csv = project_root / "data" / "raw" / "crypto_universe.csv"
                    all_assets = load_asset_universe(universe_csv)
                    
                    # Filter to only assets that exist in training data
                    df_sample = pd.read_parquet(data_path, columns=['ticker'])
                    training_assets = set(df_sample['ticker'].unique())
                    
                    # Filter universe assets to match training data
                    self.asset_list = [
                        asset for asset in all_assets
                        if asset.replace('/USD', '') in training_assets or asset in training_assets
                    ]
                    self.asset_list = sorted(self.asset_list)
                    
                    print(f"✓ Loaded {len(self.asset_list)} assets from universe CSV (filtered to training data)")
                except Exception as e2:
                    # Last resort: Use environment (for backward compatibility)
                    print(f"⚠ Warning: Could not load from CSV: {e2}")
                    print("  Falling back to environment asset list...")
                    fallback_env = FastCryptoEnv(data_path=data_path)
                    self.asset_list = fallback_env.assets
        else:
            # Use provided asset list (but ensure it's sorted)
            self.asset_list = sorted(asset_list)
        
        self.n_assets = len(self.asset_list)
        
        # Load normalization statistics (CRITICAL)
        print(f"📦 Loading normalization statistics from: {vec_norm_path}")
        
        # To load VecNormalize, we need a dummy environment that matches training structure
        # Create a temporary FastCryptoEnv to get the correct structure
        # This must be created before the closure to ensure proper scope access
        temp_env = FastCryptoEnv(data_path=data_path)
        
        # Create a dummy vectorized environment for VecNormalize
        # We need to pass a factory function, not the env itself
        def make_dummy_env():
            return temp_env
        
        dummy_vec_env = DummyVecEnv([make_dummy_env])
        
        # Load the normalization statistics
        self.norm_env = VecNormalize.load(str(vec_norm_path), dummy_vec_env)
        
        # CRITICAL: Freeze the normalization stats
        # We don't want live trading to update the moving averages
        self.norm_env.training = False
        self.norm_env.norm_reward = False
        
        # Store reference to norm_env for direct access (used in evaluation scripts)
        # This allows evaluation scripts to normalize observations directly
        self.norm_env = self.norm_env
        
        print(f"✓ Normalization statistics loaded and frozen")
        print(f"  Assets: {self.asset_list}")
        print(f"  Observation space: {self.norm_env.observation_space.shape}")
        
        # Detect if model was trained with TA features
        # Observation space size tells us: (n_assets * n_features_per_asset) + 2
        # Base features: 4 per asset (returns, volatility, commit_count, utility_score)
        # TA features: 5 per asset (RSI, MACD, BBW, ATR, SMA_RATIO)
        # Portfolio features: 2 (portfolio_value, cash_weight)
        obs_size = self.norm_env.observation_space.shape[0]
        base_features_per_asset = 4
        ta_features_per_asset = 5
        portfolio_features = 2
        expected_base_size = (self.n_assets * base_features_per_asset) + portfolio_features
        expected_with_ta_size = (self.n_assets * (base_features_per_asset + ta_features_per_asset)) + portfolio_features
        
        if obs_size == expected_base_size:
            self.model_has_ta_features = False
            print(f"  Model type: Base features only (no TA)")
        elif obs_size == expected_with_ta_size:
            self.model_has_ta_features = True
            print(f"  Model type: Base + TA features")
        else:
            # Fallback: assume no TA if size doesn't match exactly
            self.model_has_ta_features = False
            print(f"  ⚠ Warning: Observation size {obs_size} doesn't match expected sizes")
            print(f"    Expected base: {expected_base_size}, with TA: {expected_with_ta_size}")
            print(f"    Assuming no TA features")
        
        # Initialize Bicameral Brain components (v2.1 - Sortino Edition)
        # Optimizer config defaults (can be overridden)
        opt_config = optimizer_config or {}
        lookback_days = opt_config.get("lookback_days", 90)
        benchmark_rate = opt_config.get("benchmark_rate", 0.0)
        risk_free_rate = opt_config.get("risk_free_rate", 0.0)
        
        # Initialize Downside Risk Manager (Semicovariance Calculator)
        # This calculates downside risk only (ignores "good" volatility)
        self.risk_calc = SemicovarianceCalculator(
            lookback_days=lookback_days,
            benchmark_rate=benchmark_rate
        )
        print(f"✓ Semicovariance Calculator initialized (lookback: {lookback_days} days, benchmark: {benchmark_rate})")
        
        # Initialize Sortino Ratio Optimizer (The Negotiator)
        # This maximizes Sortino Ratio (return per unit of downside risk)
        self.solver = SortinoRatioOptimizer(risk_free_rate=risk_free_rate)
        print(f"✓ Sortino Ratio Optimizer initialized (risk-free rate: {risk_free_rate})")
        
        # Phase 5.6 Sprint 2: Calmar Risk Gate (Circuit Breaker)
        # Initialize with default limits (15% max drawdown, 10% unlock threshold)
        self.risk_gate = CalmarRiskGate(
            max_drawdown_limit=0.15,  # 15% drawdown triggers lock
            unlock_threshold=0.10  # 10% drawdown required to unlock
        )
        print("✓ Calmar Risk Gate initialized (Circuit Breaker)")
    
    def reset_lstm_state(self):
        """
        Reset LSTM states (for new episodes or when restarting inference).
        
        This should be called when:
        - Starting a new trading session
        - After a significant market event (e.g., market crash)
        - When you want the model to "forget" previous context
        """
        if self.is_lstm:
            self.lstm_states = None
            self.episode_starts = np.array([True])
            print("🔄 LSTM states reset - Brain memory cleared")
    
    def _prepare_observation(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare observation vector from live data.
        
        Phase 5.1: Now includes Technical Analysis features.
        This method replicates the feature engineering from FastCryptoEnv:
        - Base features: Returns, Volatility, Commit Count (norm), Utility Score
        - TA features: RSI, MACD, BBW, ATR, SMA_RATIO (if available)
        
        CRITICAL: Feature order must match training exactly to prevent "Brain Scramble".
        
        Args:
            df: DataFrame with market data (must have columns matching training data)
        
        Returns:
            Observation array with shape (1, n_features) ready for normalization
        """
        # Step 1: Add TA features ONLY if model was trained with them
        # This prevents shape mismatch errors when model expects different feature count
        if self.model_has_ta_features:
            ta = TechnicalAnalysis()
            df = ta.add_all_features(df)
        
        # Ensure data is sorted by timestamp and ticker
        df = df.sort_values(['timestamp', 'ticker']).reset_index(drop=True)
        
        # Check if TA features are present (only if model expects them)
        ta_columns = ['RSI_14', 'MACD', 'BB_WIDTH', 'ATR_NORM', 'SMA_200_RATIO']
        has_ta_features = self.model_has_ta_features and all(col in df.columns for col in ta_columns)
        
        # Get the most recent timestamp
        latest_timestamp = df['timestamp'].max()
        latest_data = df[df['timestamp'] == latest_timestamp]
        
        # Initialize observation array
        # Base features per asset: 4 (returns, volatility, commit_count_norm, utility_score)
        # TA features per asset: 5 (RSI, MACD, BBW, ATR, SMA_RATIO) if present
        base_features_per_asset = 4
        ta_features_per_asset = 5 if has_ta_features else 0
        n_features_per_asset = base_features_per_asset + ta_features_per_asset
        n_obs_features = (self.n_assets * n_features_per_asset) + 2
        obs = np.zeros(n_obs_features, dtype=np.float32)
        
        idx = 0
        
        # For each asset, extract features (matching FastCryptoEnv structure)
        # Note: asset_list may have symbols without /USD, but data file uses base symbols
        for asset in self.asset_list:
            # Strip /USD if present for data lookup (data file uses base symbols like "BTC")
            base_asset = asset.replace('/USD', '') if '/USD' in asset else asset
            asset_data = latest_data[latest_data['ticker'] == base_asset]
            
            if len(asset_data) > 0:
                row = asset_data.iloc[0]
                
                # Calculate returns (pct_change)
                # Get previous row for this asset to calculate return
                # Use base_asset for data lookup (data uses "BTC", not "BTC/USD")
                asset_history = df[df['ticker'] == base_asset].sort_values('timestamp')
                if len(asset_history) > 1:
                    current_price = asset_history.iloc[-1]['close']
                    previous_price = asset_history.iloc[-2]['close']
                    returns = (current_price / previous_price) - 1.0 if previous_price > 0 else 0.0
                else:
                    returns = 0.0
                
                # Calculate volatility (rolling std of returns)
                if len(asset_history) > 7:
                    asset_history['returns'] = asset_history['close'].pct_change()
                    volatility = asset_history['returns'].rolling(window=7, min_periods=1).std().iloc[-1]
                    volatility = volatility if not pd.isna(volatility) else 0.0
                else:
                    volatility = 0.0
                
                # Normalize commit_count (MinMax scaling 0-1)
                commit_count = row.get('commit_count_4_weeks', None)
                # Handle None/NaN values - use placeholder if missing
                if commit_count is None or pd.isna(commit_count):
                    commit_count = 100.0  # Placeholder for missing fundamentals
                
                # Use base_asset for data lookup (data uses "BTC", not "BTC/USD")
                asset_all_data = df[df['ticker'] == base_asset]
                commit_values = asset_all_data['commit_count_4_weeks'].dropna()
                
                if len(commit_values) > 0:
                    commit_min = commit_values.min()
                    commit_max = commit_values.max()
                    if commit_max > commit_min:
                        commit_count_norm = (commit_count - commit_min) / (commit_max - commit_min)
                        commit_count_norm = max(0.0, min(1.0, commit_count_norm))  # Clip to [0, 1]
                    else:
                        commit_count_norm = 0.5
                else:
                    # No commit_count data available, use neutral value
                    commit_count_norm = 0.5
                
                # Utility score (already 0-1, but ensure valid)
                utility_score = row.get('utility_score', None)
                # Handle None/NaN values - default to neutral (0.5)
                if utility_score is None or pd.isna(utility_score):
                    utility_score = 0.5
                else:
                    utility_score = float(utility_score)
                utility_score = max(0.0, min(1.0, utility_score))  # Clip to [0, 1]
                
                # Base features (always present)
                obs[idx] = returns
                obs[idx + 1] = volatility
                obs[idx + 2] = commit_count_norm
                obs[idx + 3] = utility_score
                idx += 4
                
                # TA features (if present, normalized to match training)
                if has_ta_features:
                    # RSI: Normalize 0-100 to 0-1
                    rsi = row.get('RSI_14', 50.0)
                    if pd.isna(rsi):
                        rsi = 50.0
                    rsi_norm = float(rsi) / 100.0
                    obs[idx] = max(0.0, min(1.0, rsi_norm))
                    
                    # MACD: Normalize per asset (min-max)
                    macd = row.get('MACD', 0.0)
                    if pd.isna(macd):
                        macd = 0.0
                    asset_all_data = df[df['ticker'] == base_asset]
                    macd_values = asset_all_data['MACD'].dropna()
                    if len(macd_values) > 0:
                        macd_min = macd_values.min()
                        macd_max = macd_values.max()
                        if macd_max > macd_min:
                            macd_norm = (float(macd) - macd_min) / (macd_max - macd_min)
                        else:
                            macd_norm = 0.0
                    else:
                        macd_norm = 0.0
                    obs[idx + 1] = max(0.0, min(1.0, macd_norm))
                    
                    # BBW: Normalize per asset (min-max)
                    bbw = row.get('BB_WIDTH', 0.0)
                    if pd.isna(bbw):
                        bbw = 0.0
                    bbw_values = asset_all_data['BB_WIDTH'].dropna()
                    if len(bbw_values) > 0:
                        bbw_min = bbw_values.min()
                        bbw_max = bbw_values.max()
                        if bbw_max > bbw_min:
                            bbw_norm = (float(bbw) - bbw_min) / (bbw_max - bbw_min)
                        else:
                            bbw_norm = 0.0
                    else:
                        bbw_norm = 0.0
                    obs[idx + 2] = max(0.0, min(1.0, bbw_norm))
                    
                    # ATR_NORM: Already normalized, just ensure valid
                    atr_norm = row.get('ATR_NORM', 0.0)
                    if pd.isna(atr_norm):
                        atr_norm = 0.0
                    obs[idx + 3] = max(0.0, min(1.0, float(atr_norm)))
                    
                    # SMA_200_RATIO: Already a ratio, just ensure valid
                    sma_ratio = row.get('SMA_200_RATIO', 1.0)
                    if pd.isna(sma_ratio):
                        sma_ratio = 1.0
                    obs[idx + 4] = max(0.0, min(2.0, float(sma_ratio)))  # Allow up to 2x
                    
                    idx += 5
            else:
                # Asset not available, use zeros (or neutral for TA)
                obs[idx:idx+4] = 0.0
                idx += 4
                if has_ta_features:
                    obs[idx:idx+4] = 0.0  # RSI, MACD, BBW, ATR = 0
                    obs[idx + 4] = 1.0    # SMA_RATIO = 1.0 (neutral)
                    idx += 5
        
        # Add portfolio value (normalized to initial cash)
        # For live trading, we'll use 1.0 (current portfolio value / initial)
        # This should be updated from actual portfolio state in production
        obs[idx] = 1.0  # portfolio_value_norm
        
        # Add cash weight (will be set by agent, but initialize to 1.0 for safety)
        obs[idx + 1] = 1.0  # cash_weight
        
        # Reshape to (1, n_features) for batch prediction
        return obs.reshape(1, -1)
    
    def _fetch_live_data(self) -> pd.DataFrame:
        """
        Fetch live market data from Alpaca API.
        
        This is the "Live Data Bridge" that ensures the PPO agent sees current
        market prices instead of stale static data.
        
        Returns:
            pd.DataFrame: DataFrame with same structure as training data
                Columns: timestamp, ticker, open, high, low, close, volume, 
                         commit_count_4_weeks, utility_score
        
        Raises:
            RuntimeError: If provider is not set
            ValueError: If data fetching fails
        """
        if self.provider is None:
            # Fallback to static data if provider not available
            print("⚠ Warning: No provider set, using static data from parquet file")
            return pd.read_parquet(self.data_path)
        
        # Convert asset list to Alpaca symbols (add /USD if needed)
        alpaca_symbols = []
        for asset in self.asset_list:
            # Strip /USD if present, then add it back for Alpaca
            base_asset = asset.replace('/USD', '') if '/USD' in asset else asset
            alpaca_symbols.append(f"{base_asset}/USD")
        
        print(f"📡 Fetching live market data for {len(alpaca_symbols)} assets...")
        
        # Fetch enough history for feature calculation
        # For 15Min bars: 500 hours = 2000 bars (~83 days)
        # This ensures we can calculate rolling volatility and returns correctly
        df = self.provider.fetch_crypto_history(
            symbols=alpaca_symbols,
            timeframe="15Min",  # v1.2: High-frequency 15-minute bars
            hours=500,  # Enough for rolling features
            include_fundamentals=False  # We'll use synthetic/neutral values
        )
        
        # Ensure we have the required columns
        required_cols = ['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in live data: {missing_cols}")
        
        # Fill missing fundamental columns with neutral values (for feature engineering)
        if 'commit_count_4_weeks' not in df.columns:
            df['commit_count_4_weeks'] = 100.0  # Neutral placeholder
        if 'utility_score' not in df.columns:
            df['utility_score'] = 0.5  # Neutral sentiment
        
        print(f"✓ Live data fetched: {len(df)} bars across {df['ticker'].nunique()} assets")
        return df
    
    def get_target_portfolio(self, portfolio_value: float = 100000.0) -> TargetPortfolio:
        """
        Generate target portfolio using the Bicameral Brain architecture (v2.1).
        
        This method implements the "Analyst vs. Risk Manager" model:
        1. ML Analyst (PPO): Outputs raw alpha scores (expected returns)
        2. Quant Risk Manager: Calculates covariance matrix
        3. Optimizer: Solves mean-variance optimization with Halal constraints
        
        Args:
            portfolio_value: Current portfolio value (for normalization)
                            Default: 100000.0 (initial cash)
        
        Returns:
            TargetPortfolio with optimized weights
        
        Raises:
            ValueError: If prediction fails or optimization fails
        """
        # Fetch live data (or fallback to static if provider not set)
        df = self._fetch_live_data()
        
        # Prepare raw observation
        raw_obs = self._prepare_observation(df)
        
        # CRITICAL: Normalize observation using training statistics
        # This ensures the agent sees data in the same format it was trained on
        norm_obs = self.norm_env.normalize_obs(raw_obs)
        
        # ============================================================
        # STEP A: ML Analyst (PPO/LSTM) - Get Raw Alpha Scores
        # ============================================================
        print("🧠 PPO Analyst: Generating expected returns...")
        
        # Predict action (raw output from PPO/LSTM - DO NOT normalize/clip here!)
        # For LSTM models, we need to pass states and episode_starts
        if self.is_lstm:
            # LSTM models are stateful - maintain memory across predictions
            # This allows the brain to remember market trends from previous 15-min ticks
            action, self.lstm_states = self.model.predict(
                norm_obs,
                state=self.lstm_states,  # Previous LSTM hidden/cell states
                episode_start=self.episode_starts,  # True if new episode, False if continuation
                deterministic=True
            )
            # After prediction, this is a continuation (not a new episode)
            self.episode_starts = np.array([False])
        else:
            # Regular PPO (MLP) - stateless prediction
            action, _ = self.model.predict(norm_obs, deterministic=True)
        
        # Flatten action array
        action = action.flatten()
        
        # Convert raw PPO action to expected returns (alpha scores)
        # CRITICAL: Do NOT normalize/clip the actions here.
        # We treat the raw output as "expected returns" (μ).
        available_assets = set(df['ticker'].unique()) if 'ticker' in df.columns else set()
        expected_returns_dict = {}
        
        # Action array: [Asset1, Asset2, ..., AssetN, Cash]
        # self.asset_list contains only assets (no cash)
        for i, asset in enumerate(self.asset_list):
            # Strip /USD for comparison (data uses base symbols like "BTC")
            base_asset = asset.replace('/USD', '') if '/USD' in asset else asset
            
            # Only include assets that have data
            if base_asset not in available_assets:
                continue
            
            # Get raw score from PPO (this is the "expected return")
            raw_score = float(action[i])
            
            # Store as expected return (use base symbol for now, will convert later)
            expected_returns_dict[base_asset] = raw_score
        
        # Convert to Series for optimizer
        expected_returns = pd.Series(expected_returns_dict)
        
        if expected_returns.empty:
            print("  ⚠ Warning: No assets with data, returning empty portfolio")
            return TargetPortfolio(weights={})
        
        print(f"  ✓ PPO Raw Scores: {dict(expected_returns.head(5))}...")
        
        # ============================================================
        # STEP B: Downside Risk Manager - Calculate Semicovariance Matrix
        # ============================================================
        print("📉 Downside Risk Manager: Calculating semicovariance matrix...")
        
        # Prepare price DataFrame for semicovariance calculation
        # Need format: Index=Date, Columns=Tickers, Values=Close prices
        price_pivot = df.pivot_table(
            index='timestamp',
            columns='ticker',
            values='close',
            aggfunc='first'
        )
        
        # Sort by timestamp (ascending)
        price_pivot = price_pivot.sort_index()
        
        # Filter to only assets we have expected returns for
        common_assets = price_pivot.columns.intersection(expected_returns.index)
        if len(common_assets) == 0:
            raise ValueError("No common assets between expected returns and price data")
        
        price_pivot = price_pivot[common_assets]
        expected_returns = expected_returns[common_assets]
        
        # Calculate semicovariance matrix (downside risk only)
        # This ignores "good" volatility (upside moves) and only considers "bad" volatility (downside moves)
        try:
            downside_risk_matrix = self.risk_calc.get_risk_matrix(price_pivot)
            print(f"  ✓ Semicovariance matrix calculated: {downside_risk_matrix.shape}")
        except Exception as e:
            print(f"  ⚠ Warning: Semicovariance calculation failed ({e}), using equal weights")
            # Fallback to equal weights if semicovariance fails
            n_assets = len(common_assets)
            weights_dict = {f"{asset}/USD": 1.0 / n_assets for asset in common_assets}
            return TargetPortfolio(weights=weights_dict)
        
        # ============================================================
        # STEP C: Sortino Optimizer (The Negotiator) - Maximize Sortino Ratio
        # ============================================================
        print("⚖️ Sortino Optimizer: Solving for maximum Sortino Ratio...")
        
        try:
            # Solve the optimization problem
            # max_sharpe() with semicovariance matrix = max_sortino()
            raw_target = self.solver.optimize(expected_returns, downside_risk_matrix)
            print(f"  ✓ Optimization complete: {len(raw_target.weights)} assets selected (maximizing Sortino Ratio)")
            
            # Phase 5.6 Sprint 2: Apply Calmar Risk Gate (Circuit Breaker)
            # If drawdown exceeds limit, override to 100% cash
            if self.trading_engine is not None:
                try:
                    current_state = self.trading_engine.get_portfolio_state()
                    final_target = self.risk_gate.validate(raw_target, current_state)
                    if final_target != raw_target:
                        print(f"  🛡️ Calmar Gate: Override to 100% cash (drawdown protection)")
                except Exception as e:
                    # If gate validation fails, log warning but return raw target
                    import logging
                    logging.warning(f"⚠ Calmar Gate validation failed: {e}. Using raw target.")
                    final_target = raw_target
            else:
                # No trading engine available, skip gate validation
                final_target = raw_target
            
            return final_target
        except Exception as e:
            print(f"  ⚠ Warning: Optimization failed ({e}), using equal weights")
            # Fallback to equal weights if optimization fails
            n_assets = len(common_assets)
            weights_dict = {f"{asset}/USD": 1.0 / n_assets for asset in common_assets}
            return TargetPortfolio(weights=weights_dict)

