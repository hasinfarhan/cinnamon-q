"""
Model Training Stage - Train PPO Agent

This module wraps the training logic from scripts/10_train_hdrl_optimized.py
into a callable function for the MLOps pipeline.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from strategy.rl.data_loader import prepare_training_data
from strategy.rl.fast_env import FastCryptoEnv

logger = logging.getLogger(__name__)


def _check_tensorboard():
    """Check if tensorboard is available."""
    try:
        import tensorboard
        return True
    except ImportError:
        return False


def _check_rich():
    """Check if rich and tqdm are available for progress bar."""
    try:
        import rich
        import tqdm
        return True
    except ImportError:
        return False


def make_env_factory(data_config: dict):
    """Factory function for creating environments with memory-mapped data."""
    def _init():
        env = FastCryptoEnv(data_config=data_config)
        return env
    return _init


def run_training(
    data_path: Path,
    model_output_dir: Path,
    total_timesteps: int = 10_000_000,
    n_envs: int = 16,
    batch_size: int = 4096,
    learning_rate: float = 3e-4,
    device: str = "cuda"
) -> tuple[Path, Path]:
    """
    Run the model training stage.
    
    Trains a PPO agent on the enriched training data with GPU acceleration
    and parallel environments for maximum hardware utilization.
    
    Args:
        data_path: Path to enriched training data parquet file
        model_output_dir: Directory to save trained model and normalization stats
        total_timesteps: Total training timesteps (default: 10M)
        n_envs: Number of parallel environments (default: 16)
        batch_size: Training batch size (default: 4096)
        learning_rate: Learning rate (default: 3e-4)
        device: Device to use ("cuda" or "cpu", default: "cuda")
    
    Returns:
        Tuple of (model_path, vec_norm_path)
    """
    logger.info("=" * 80)
    logger.info("Stage 3: Model Training")
    logger.info("=" * 80)
    
    # Hardware detection
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("No GPU detected, using CPU")
        device = "cpu"
    
    cpu_cores = os.cpu_count()
    logger.info(f"CPU Cores: {cpu_cores}")
    
    # Create output directory
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare training data (memory-mapped)
    logger.info("Preparing training data (memory-mapped)...")
    logger.info(f"  Using data file: {data_path}")
    
    # Phase 5.1: Prefer crypto_deep_v2.parquet (with TA features) if available
    if not data_path.exists():
        # Try v2 first (with TA features)
        v2_path = data_path.parent / "crypto_deep_v2.parquet"
        if v2_path.exists():
            logger.info(f"  Falling back to: {v2_path} (with TA features)")
            data_path = v2_path
        else:
            # Try v1 (without TA features, backward compatible)
            v1_path = data_path.parent / "crypto_training_data.parquet"
            if v1_path.exists():
                logger.warning(f"  Falling back to: {v1_path} (no TA features)")
                data_path = v1_path
    
    temp_dir = Path(tempfile.mkdtemp(prefix="hdrl_data_"))
    
    try:
        data_arrays = prepare_training_data(data_path, temp_dir)
        
        # Create environment factory
        make_env_fn = make_env_factory(data_arrays)
        
        # Create vectorized environment
        logger.info(f"Creating {n_envs} parallel environments...")
        env = make_vec_env(make_env_fn, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        # Create evaluation environment
        logger.info("Creating evaluation environment...")
        eval_env_raw = FastCryptoEnv(data_config=data_arrays)
        eval_env = DummyVecEnv([lambda: eval_env_raw])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
        eval_env.training = False
        
        # Setup callbacks
        logger.info("Setting up training callbacks...")
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=str(model_output_dir / "checkpoints"),
            name_prefix="model"
        )
        
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=10,
            min_evals=20
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_output_dir),
            log_path=str(model_output_dir / "logs"),
            eval_freq=50_000,
            deterministic=True,
            render=False,
            callback_on_new_best=stop_callback
        )
        
        # Initialize PPO model
        logger.info("Initializing PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=10,
            policy_kwargs={
                "net_arch": [512, 512, 512]  # Deep network for GPU saturation
            },
            tensorboard_log=str(model_output_dir / "tensorboard") if _check_tensorboard() else None
        )
        
        # Train model
        logger.info(f"Starting training ({total_timesteps:,} timesteps)...")
        use_progress_bar = _check_rich()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=use_progress_bar
        )
        
        # Save final model
        logger.info("Saving final model...")
        model_path = model_output_dir / "final_model.zip"
        vec_norm_path = model_output_dir / "vec_normalize.pkl"
        
        model.save(str(model_path))
        env.save(str(vec_norm_path))
        
        logger.info(f"✓ Training complete: {model_path}, {vec_norm_path}")
        
        return model_path, vec_norm_path
        
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

