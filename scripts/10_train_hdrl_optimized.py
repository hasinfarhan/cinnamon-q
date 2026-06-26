"""
HDRL Training Script - Optimized for High-End Hardware

Phase 3: The Crypto Expert Alpha Master - Step 3 (Optimized)

This script trains a PPO agent with maximum hardware utilization:
- RTX 4070 Ti Super: Large batch sizes and deep networks
- Ryzen 3700X (16 threads): 16 parallel environments
- Numpy arrays: Blazing fast data access

Based on the optimization strategy for saturating GPU and CPU.
"""

import os
import sys
import time
import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# Add src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
from sb3_contrib import RecurrentPPO  # LSTM support
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.monitor import Monitor

# Import environment and data loader directly to avoid dependency chain
sys.path.insert(0, str(project_root / "src" / "strategy" / "rl"))
from fast_env import FastCryptoEnv
from vector_env import CryptoVectorEnv
from data_loader import prepare_training_data

# Import multimodal policy for Research Track
sys.path.insert(0, str(project_root / "src" / "strategy" / "rl"))
from policy import MultimodalExtractor


def _check_tensorboard():
    """Check if tensorboard is available."""
    try:
        import tensorboard
        return True
    except ImportError:
        return False


def make_env_factory(data_config: dict, fee_rate: float = 0.001, use_embeddings: bool = False, embeddings_path: Optional[Path] = None):
    """
    Factory function for creating environments with memory-mapped data.
    
    This is required for SubprocVecEnv. Each process will call this
    to create its own environment instance. The data_config contains
    file paths to memory-mapped .npy files (zero-copy access).
    
    Args:
        data_config: Data configuration dictionary from prepare_training_data()
                    Contains paths to .npy files and metadata
                    Phase 6: Now includes 'benchmark_path' for DIR calculation
        fee_rate: Transaction fee rate (default: 0.001 = 0.1% = 10 bps)
                 Phase 5.5: Real World Environment - Fees prevent over-trading
        use_embeddings: Whether to use multimodal environment with embeddings (Research Track)
        embeddings_path: Path to news_embeddings.npy file
    
    Returns:
        Factory function that creates FastCryptoEnv or CryptoVectorEnv instances
    """
    def _init():
        # Research Track: Use multimodal environment if embeddings are available
        if use_embeddings and embeddings_path and embeddings_path.exists():
            env = CryptoVectorEnv(
                data_config=data_config,
                embeddings_path=embeddings_path,
                fee_rate=fee_rate,
                benchmark_returns=data_config.get('benchmark_returns')
            )
        else:
            # Standard environment (backward compatible)
            env = FastCryptoEnv(data_config=data_config, fee_rate=fee_rate)
        return env
    return _init


def main():
    """Main training execution with hardware optimization."""
    print("=" * 80)
    print("[*] HDRL Brain Training - OPTIMIZED for High-End Hardware")
    print("Phase 3: The Crypto Expert Alpha Master - Step 3")
    print("=" * 80)
    print()
    
    # --- HARDWARE DETECTION ---
    print("[?] Hardware Detection:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU available: {gpu_name} ({gpu_memory:.1f} GB) - using CPU for stability")
    # Force CPU: laptop GPU (RTX 3060) causes silent CUDA crashes on long training runs
    device = "cpu"
    print(f"  Device: CPU (stable mode)")
    
    cpu_cores = os.cpu_count()
    print(f"  [OK] CPU Cores: {cpu_cores}")
    print()
    
    # --- PATHS ---
    # v5: 5-year dataset (Jan 2021 -> now), all features inline, no multi-step pipeline
    v5_path             = project_root / "data" / "processed" / "crypto_deep_v5.parquet"
    v4_sentiment_path   = project_root / "data" / "processed" / "crypto_deep_v4_sentiment.parquet"
    sentiment_data_path = project_root / "data" / "processed" / "crypto_deep_v3_sentiment.parquet"
    v3_data_path = project_root / "data" / "processed" / "crypto_deep_v3.parquet"
    v2_data_path = project_root / "data" / "processed" / "crypto_deep_v2.parquet"
    v1_data_path = project_root / "data" / "processed" / "crypto_training_data.parquet"
    master_data_path = project_root / "data" / "processed" / "crypto_deep_master.parquet"

    if v5_path.exists():
        data_path = v5_path
        print("[OK] Using crypto_deep_v5.parquet (5-year dataset Jan 2021->now, all features) - V35 MODE!")
    elif v4_sentiment_path.exists():
        data_path = v4_sentiment_path
        print("[OK] Using crypto_deep_v4_sentiment.parquet (14 clean assets, all features fixed) - CLEAN THESIS MODE!")
    elif sentiment_data_path.exists():
        data_path = sentiment_data_path
        print("[OK] Using crypto_deep_v3_sentiment.parquet (real Gemini LLM scores) - FULL THESIS MODE!")
    elif v3_data_path.exists():
        data_path = v3_data_path
        print("[OK] Using crypto_deep_v3.parquet (with GARCH + TA features) - World-Class Brain mode!")
    elif v2_data_path.exists():
        data_path = v2_data_path
        print("[!!] Using crypto_deep_v2.parquet (with TA features, no GARCH) - Super Brain mode!")
    elif v1_data_path.exists():
        data_path = v1_data_path
        print("[!!] Using crypto_training_data.parquet (no TA features)")
        print("  For best results, run: python scripts/12_enrich_data.py")
    elif master_data_path.exists():
        data_path = master_data_path
        print("[!!] Using crypto_deep_master.parquet (no TA features, no utility_score)")
    else:
        raise FileNotFoundError(
            f"Dataset not found. Please run scripts/08_fetch_crypto_deep.py and scripts/12_enrich_data.py first."
        )
    
    # Check for embeddings (Research Track)
    embeddings_path = project_root / "data" / "processed" / "news_embeddings.npy"
    use_embeddings = embeddings_path.exists()
    
    # Model directory selection based on data version and embeddings
    # Research Track: Multimodal (embeddings) -> hdrl_multimodal_v1
    # Phase 6: World-Class Brain (v3) -> hdrl_worldclass_v1
    # Phase 5.6: Fortress Brain (v2) -> hdrl_fortress_v1
    # Phase 5.2: Super Brain (v2) -> hdrl_super_brain (legacy)
    # Phase 5.1: Regular LSTM (v1) -> hdrl_lstm
    if use_embeddings:
        model_dir = project_root / "models" / "hdrl_multimodal_v1"
        log_dir = project_root / "logs" / "hdrl_training_multimodal_v1"
        print("  Research Track: Multimodal Brain mode (Embeddings + Market Features)!")
        print("  Model directory: models/hdrl_multimodal_v1")
    elif v5_path.exists() and data_path == v5_path:
        model_dir = project_root / "models" / "hdrl_worldclass_v40"
        log_dir = project_root / "logs" / "hdrl_training_worldclass_v40"
        print("  THESIS MODEL v40: C4+C6 revert to SMA200 + mean/mean fix, ent_coef=0.015")
        print("  Model directory: models/hdrl_worldclass_v40")
    elif v4_sentiment_path.exists() and data_path == v4_sentiment_path:
        model_dir = project_root / "models" / "hdrl_worldclass_v35"
        log_dir = project_root / "logs" / "hdrl_training_worldclass_v35"
        print("  THESIS MODEL v35: C5 dropped - conviction scaling alone handles alpha selectivity (restores v19 exposure)")
        print("  Model directory: models/hdrl_worldclass_v35")
    elif sentiment_data_path.exists() and data_path == sentiment_data_path:
        model_dir = project_root / "models" / "hdrl_worldclass_v2"
        log_dir = project_root / "logs" / "hdrl_training_worldclass_v2"
        print("  THESIS MODEL v2: Real Gemini LLM Sentiment + GARCH + TA!")
        print("  Model directory: models/hdrl_worldclass_v2")
    elif v3_data_path.exists() and data_path == v3_data_path:
        model_dir = project_root / "models" / "hdrl_worldclass_v1"
        log_dir = project_root / "logs" / "hdrl_training_worldclass_v1"
        print("  World-Class Brain mode (Phase 6): GARCH + DIR + Calmar!")
        print("  Model directory: models/hdrl_worldclass_v1")
    elif v2_data_path.exists() and data_path == v2_data_path:
        model_dir = project_root / "models" / "hdrl_fortress_v1"
        log_dir = project_root / "logs" / "hdrl_training_fortress_v1"
        print("  [*] Fortress Brain mode: Training with TA features + Composite Reward!")
        print("  [DIR] Model directory: models/hdrl_fortress_v1")
    else:
        model_dir = project_root / "models" / "hdrl_lstm"
        log_dir = project_root / "logs" / "hdrl_training_lstm"
        print("  [!!] Regular LSTM mode: No TA features available")
    tensorboard_dir = log_dir / "tensorboard"
    
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # --- LOAD DATA ONCE (Main Process) ---
    print("Step 1: Loading and preprocessing data (Main Process)...")
    start_data_load = time.time()
    
    # Create temporary directory for memory-mapped files
    temp_dir = Path(tempfile.mkdtemp(prefix="hdrl_data_"))
    
    # Register cleanup function to remove temp directory on exit
    def cleanup_temp_dir():
        """Clean up temporary directory on exit."""
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    atexit.register(cleanup_temp_dir)
    print(f"  Temporary directory: {temp_dir}")
    
    try:
        data_config = prepare_training_data(data_path, temp_dir, resample_freq='4h')
        data_load_time = time.time() - start_data_load
        
        # Validation: Count features per asset
        has_ta = data_config.get('has_ta_features', False)
        has_corr = data_config.get('has_correlation', False)
        base_features = 5  # Phase 5.5 Sprint 2: Now includes corr_btc_norm (was 4)
        # Base: returns, volatility, commit_count_norm, utility_score, corr_btc_norm
        ta_features = 5 if has_ta else 0  # RSI, MACD, BBW, ATR, SMA_RATIO
        portfolio_features = 2  # portfolio_value, cash_weight
        features_per_asset = base_features + ta_features
        total_features = (data_config['n_assets'] * features_per_asset) + portfolio_features
        
        print(f"[DONE] Data loaded and saved in {data_load_time:.2f}s")
        print(f"   Memory-mapped files ready for subprocesses")
        corr_info = " (includes corr_btc)" if has_corr else ""
        print(f"   Training with {features_per_asset} features per asset ({base_features} base{corr_info} + {ta_features} TA)")
        print(f"   Total observation space: {total_features} features")
        if has_ta:
            print(f"   [*] Super Brain mode: LSTM + TA Features = Maximum Intelligence!")
        if has_corr:
            print(f"   [*] Fortress Brain mode: Correlation Eye enabled (can identify hedges like PAXG)!")
        print()
    except Exception as e:
        print(f"[ERR] Error loading data: {e}")
        import traceback
        traceback.print_exc()
        cleanup_temp_dir()
        sys.exit(1)
    
    # Verify data exists
    if not data_path.exists():
        print(f"[ERR] Error: Dataset not found: {data_path}")
        print("Please run scripts/09_generate_sentiment.py first.")
        sys.exit(1)
    
    # --- PARALLELIZATION CONFIG ---
    # Use DummyVecEnv with 4 envs - faster iteration during v10 development
    n_envs = 4
    
    print("[CFG]  Configuration:")
    print(f"  Parallel Environments: {n_envs}")
    print(f"  Device: {device}")
    print()
    
    # Embeddings path already set above in model directory selection
    if use_embeddings:
        print("[TXT] Research Track: Embeddings detected! Using multimodal environment.")
        print(f"   Embeddings path: {embeddings_path}")
    else:
        print("[!!] Research Track: No embeddings found. Using standard environment.")
        print("   To enable multimodal training, run: python scripts/16_generate_embeddings.py")
    print()
    
    # --- CREATE VECTORIZED ENVIRONMENT ---
    print("Step 2: Creating vectorized environment...")
    print(f"  Spawning {n_envs} parallel environments...")
    
    start_init = time.time()
    
    # Create factory function with memory-mapped data config
    # Subprocesses will load from memory-mapped files (instant, zero-copy)
    # Phase 5.5: Pass fee_rate to enforce transaction costs (prevents over-trading)
    # Research Track: Pass embeddings for multimodal training
    fee_rate = 0.001  # 0.1% = 10 bps (standard crypto exchange fee)
    make_env_fn = make_env_factory(
        data_config, 
        fee_rate=fee_rate,
        use_embeddings=use_embeddings,
        embeddings_path=embeddings_path if use_embeddings else None
    )
    
    # DummyVecEnv = single-process (no subprocess crashes on Windows)
    # Slightly slower than SubprocVecEnv but fully stable
    env = make_vec_env(
        make_env_fn,
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )
    
    # VecNormalize = Essential for PPO convergence
    # Normalizes observations and rewards automatically
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=True
    )
    
    init_time = time.time() - start_init
    print(f"[DONE] Environments ready in {init_time:.2f}s")
    print(f"   (Total startup time: {data_load_time + init_time:.2f}s)")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()
    
    # --- HYPERPARAMETERS FOR RTX 4070 Ti SUPER (LSTM EDITION) ---
    print("Step 3: Configuring RecurrentPPO (LSTM) with GPU-optimized hyperparameters...")
    
    # LSTM Configuration
    # n_steps: Number of steps per environment per update
    # For LSTM, this becomes the BPTT sequence length (temporal window)
    # v6: 512 steps x 15 min = 128 hours ~ 5 trading days
    #     Captures weekly / multi-day regime patterns (bull/bear rotation).
    #     v5 used only 128 steps = 32 h - too short to see week-long regimes.
    n_steps = 512  # n_steps * n_envs = 512 * 4 = 2048 buffer per update

    # batch_size must divide n_steps * n_envs = 2048
    batch_size = 512
    
    # Research Track: Use multimodal policy if embeddings are available
    if use_embeddings:
        # For multimodal observations (Dict space), we need MultiInputPolicy
        # However, RecurrentPPO doesn't support MultiInputPolicy directly
        # We'll use standard PPO with MultiInputPolicy for now
        # TODO: Create custom RecurrentMultiInputPolicy for LSTM + multimodal
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import MultiInputPolicy
        
        # Get market feature dimension from observation space
        market_dim = env.observation_space['market'].shape[0] if hasattr(env.observation_space, 'spaces') else 150
        
        policy_kwargs = dict(
            features_extractor_class=MultimodalExtractor,
            features_extractor_kwargs=dict(
                market_dim=market_dim,
                text_dim=768,
                market_hidden=128,
                text_hidden=256,
                output_dim=64,
                fusion_dim=128
            ),
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256]
            ),
            activation_fn=torch.nn.ReLU
        )
        
        print(f"  Policy: MultiInputPolicy with MultimodalExtractor (Research Track)")
        print(f"  Market features: {market_dim} dim")
        print(f"  Text embeddings: 768 dim")
        print(f"  Fusion: 128 dim")
        print(f"  [!!] Note: Using standard PPO (not RecurrentPPO) for multimodal support")
        print(f"  n_steps (sequence length): {n_steps}")
        print(f"  batch_size: {batch_size}")
        print(f"  MLP layers: 256x256")
        print(f"  Buffer size per update: {n_steps * n_envs:,} steps")
        print()
        
        model = PPO(
            MultiInputPolicy,  # Multimodal policy with text embeddings
            env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_dir) if _check_tensorboard() else None
        )
    else:
        # Standard LSTM Policy: MlpLstmPolicy combines MLP layers with LSTM memory
        # lstm_hidden_size: Size of LSTM hidden state (memory capacity)
        # n_lstm_layers: Number of LSTM layers (stacked for deeper memory)
        policy_kwargs = dict(
            net_arch=dict(
                pi=[128, 128],  # Smaller network for CPU training
                vf=[128, 128]
            ),
            activation_fn=torch.nn.ReLU,
            lstm_hidden_size=128,  # v6: doubled from 64 - matches 146-feature obs space
            n_lstm_layers=1,
            enable_critic_lstm=False  # Disable critic LSTM to save compute
        )
        
        print(f"  Policy: MlpLstmPolicy (LSTM with Memory)")
        print(f"  n_steps (BPTT window): {n_steps} steps = {n_steps * 4:.0f} hours ~ {n_steps * 4 / 24:.0f} days  (4h bars)")
        print(f"  batch_size: {batch_size}")
        print(f"  MLP layers: 128x128")
        print(f"  LSTM hidden size: {policy_kwargs['lstm_hidden_size']}")
        print(f"  LSTM layers: {policy_kwargs['n_lstm_layers']}")
        print(f"  Buffer size per update: {n_steps * n_envs:,} steps")
        print()
        
        # v7: Higher entropy coefficient - prevents the policy-collapse-to-static-allocation
        # bug seen in v6. sb3-contrib RecurrentPPO does not support callable schedules for
        # ent_coef (only float), so we use a fixed 0.05 (5x v6's 0.01).
        # This keeps the policy exploring short/cash strategies throughout training.
        # Resume from checkpoint if one exists (e.g. after a crash)
        import glob as _glob, re as _re
        def _ckpt_steps(p):
            m = _re.search(r'_(\d+)_steps\.zip$', p)
            return int(m.group(1)) if m else 0
        _ckpts = sorted(_glob.glob(str(model_dir / "hdrl_crypto_lstm_*_steps.zip")), key=_ckpt_steps)
        if _ckpts:
            _latest_ckpt = _ckpts[-1]
            # Safety check: verify checkpoint obs space matches current env before resuming.
            # A mismatch (e.g. stale checkpoint from before feature additions) causes the
            # model to train with the wrong obs size, silently producing a broken model.
            try:
                _ckpt_model = RecurrentPPO.load(_latest_ckpt, device=device)
                _ckpt_obs_shape = _ckpt_model.observation_space.shape
                _env_obs_shape  = env.observation_space.shape
                del _ckpt_model
                if _ckpt_obs_shape != _env_obs_shape:
                    print(f"  [!!] Checkpoint obs shape {_ckpt_obs_shape} != env obs shape {_env_obs_shape}")
                    print(f"  [!!] Discarding stale checkpoint — starting fresh to avoid corrupted model.")
                    _ckpts = []
                else:
                    print(f"  [<-] Resuming from checkpoint: {Path(_latest_ckpt).name}")
            except Exception as _e:
                print(f"  [!!] Could not verify checkpoint obs shape ({_e}) — starting fresh.")
                _ckpts = []

        if _ckpts:
            _latest_ckpt = _ckpts[-1]
            model = RecurrentPPO.load(
                _latest_ckpt,
                env=env,
                device=device,
                verbose=1,
                tensorboard_log=str(tensorboard_dir) if _check_tensorboard() else None
            )
        else:
            model = RecurrentPPO(
                "MlpLstmPolicy",  # LSTM policy with memory
                env,
                verbose=1,
                device=device,
                learning_rate=3e-5,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.015,  # v39: raised from 0.005 — prevents premature convergence to low-conviction cash policy
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(tensorboard_dir) if _check_tensorboard() else None
            )
    
    print("[OK] RecurrentPPO (LSTM) agent initialized")
    print("  The brain now has MEMORY! It can learn temporal patterns.")
    print()
    
    # --- CALLBACKS ---
    print("Step 4: Setting up training callbacks...")
    
    # Checkpoint only - EvalCallback disabled (causes crashes on Windows laptops)
    # save_freq must be a multiple of (n_steps * n_envs) = 256 to actually trigger
    # 512 = every 2 rollouts (~every 2 minutes), ensures we never lose much progress
    checkpoint_callback = CheckpointCallback(
        save_freq=512,
        save_path=str(model_dir),
        name_prefix="hdrl_crypto_lstm",
        verbose=1
    )

    print("Callbacks configured (checkpoint-only mode)")
    print()
    
    # --- TRAINING ---
    print("Step 5: Starting LSTM training...")
    print("  Watch for increasing 'ep_rew_mean' in logs")
    print("  Training will use maximum hardware resources")
    print("  LSTM training may take longer than MLP (sequence processing)")
    print()
    
    # Total timesteps: 10M for thorough training
    # With 16 parallel envs and LSTM sequences, this may take longer than MLP
    # LSTM processes sequences, so effective training is deeper
    total_timesteps = 2_000_000  # v16: alpha-weighted reward + bull 3x oversample + macro recalib (2M ~ 80 min)
    
    start_time = time.time()
    
    try:
        # Check if rich is available for progress bar
        try:
            import rich
            use_progress_bar = True
        except ImportError:
            use_progress_bar = False
            print("  [!!] Progress bar disabled (rich not installed)")
        
        resuming = bool(_ckpts)
        model.learn(
            total_timesteps=total_timesteps if not resuming else (total_timesteps - _ckpt_steps(_ckpts[-1])),
            callback=[checkpoint_callback],
            progress_bar=use_progress_bar,
            reset_num_timesteps=not resuming
        )
        print()
        print("[OK] Training completed successfully!")
        print()
    except KeyboardInterrupt:
        print()
        print("[!!] Training interrupted by user")
        print("  Model checkpoints saved in:", model_dir)
        print()
    finally:
        # Always save final model and normalization stats
        print("Step 6: Saving final LSTM model and normalization stats...")
        
        # Save final model state
        final_model_path = model_dir / "final_model"
        model.save(str(final_model_path))
        print(f"[OK] Final LSTM model saved to: {final_model_path}.zip")
        print(f"  This model has MEMORY and can learn temporal patterns!")
        
        # Save normalization stats (critical for inference)
        vec_normalize_path = model_dir / "vec_normalize.pkl"
        env.save(str(vec_normalize_path))
        print(f"[OK] Normalization stats saved to: {vec_normalize_path}")
        
        training_time = (time.time() - start_time) / 60
        print(f"[OK] Training time: {training_time:.2f} minutes")
        print()
    
    # --- TEST PREDICTIONS ---
    print("Step 7: Testing model predictions...")

    # Reuse the existing training VecNormalize env (already has correct 212-feature stats).
    # Previously this created a separate FastCryptoEnv which loaded a stale pyc (206 features),
    # causing VecNormalize.load() to reinitialize obs_rms to (206,) shape, corrupting the saved pkl.
    test_env = env
    test_env.training = False
    test_env.norm_reward = False

    obs = test_env.reset()
    
    print("  Sample predictions (first 5 steps):")
    for i in range(5):
        # Predict action (model expects normalized observations)
        action, _states = model.predict(obs, deterministic=True)
        
        # Action from VecEnv is shape (1, n_assets+1), need to ensure correct format
        # VecEnv will handle the batching, but we need to ensure it's the right shape
        if isinstance(action, np.ndarray):
            action_flat = action.flatten()
        else:
            action_flat = np.array(action).flatten()
        
        # Normalize for display
        action_sum = action_flat.sum()
        if action_sum > 0:
            normalized_weights = action_flat / action_sum
        else:
            normalized_weights = action_flat
        
        # Convert to list for safe formatting
        weights_list = normalized_weights.tolist()
        
        print(f"    Step {i+1}:")
        print(f"      Portfolio weights: {weights_list}")
        print(f"      Cash weight: {weights_list[-1]:.3f}")
        
        # Step environment (VecEnv handles action batching)
        obs, reward, done, info = test_env.step(action)
        
        if done[0]:  # DummyVecEnv returns done as array
            print("      Episode ended, resetting...")
            obs = test_env.reset()
    
    print()
    
    # --- FINAL SUMMARY ---
    print("=" * 80)
    print("[DONE] HDRL Training Complete (Optimized)")
    print("=" * 80)
    print()
    print("Model Summary:")
    print(f"  Model file: {final_model_path}.zip")
    print(f"  Normalization: {vec_normalize_path}")
    print(f"  Training timesteps: {total_timesteps:,}")
    print(f"  Training time: {training_time:.2f} minutes")
    print(f"  Throughput: {total_timesteps / (training_time * 60):.0f} steps/sec")
    print()
    print("Logs & Monitoring:")
    print(f"  TensorBoard: tensorboard --logdir {tensorboard_dir}")
    print(f"  Logs directory: {log_dir}")
    print()
    print("Next Steps:")
    print("  - Review training logs and TensorBoard")
    print("  - Integrate model into Execution Engine (Step 4)")
    print()


if __name__ == "__main__":
    main()

