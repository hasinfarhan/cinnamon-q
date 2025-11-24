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

# Add src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
from stable_baselines3 import PPO
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
from data_loader import prepare_training_data


def _check_tensorboard():
    """Check if tensorboard is available."""
    try:
        import tensorboard
        return True
    except ImportError:
        return False


def make_env_factory(data_config: dict):
    """
    Factory function for creating environments with memory-mapped data.
    
    This is required for SubprocVecEnv. Each process will call this
    to create its own environment instance. The data_config contains
    file paths to memory-mapped .npy files (zero-copy access).
    
    Args:
        data_config: Data configuration dictionary from prepare_training_data()
                    Contains paths to .npy files and metadata
    
    Returns:
        Factory function that creates FastCryptoEnv instances
    """
    def _init():
        # Create environment with memory-mapped data (instant load, zero-copy)
        env = FastCryptoEnv(data_config=data_config)
        return env
    return _init


def main():
    """Main training execution with hardware optimization."""
    print("=" * 80)
    print("🚀 HDRL Brain Training - OPTIMIZED for High-End Hardware")
    print("Phase 3: The Crypto Expert Alpha Master - Step 3")
    print("=" * 80)
    print()
    
    # --- HARDWARE DETECTION ---
    print("🔍 Hardware Detection:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  ✓ GPU: {gpu_name}")
        print(f"  ✓ GPU Memory: {gpu_memory:.1f} GB")
        device = "cuda"
    else:
        print("  ⚠ No GPU detected, using CPU")
        device = "cpu"
    
    cpu_cores = os.cpu_count()
    print(f"  ✓ CPU Cores: {cpu_cores}")
    print()
    
    # --- PATHS ---
    data_path = project_root / "data" / "processed" / "crypto_training_data.parquet"
    model_dir = project_root / "models" / "hdrl_optimized"
    log_dir = project_root / "logs" / "hdrl_training_optimized"
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
        data_config = prepare_training_data(data_path, temp_dir)
        data_load_time = time.time() - start_data_load
        print(f"✅ Data loaded and saved in {data_load_time:.2f}s")
        print(f"   Memory-mapped files ready for subprocesses")
        print()
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        cleanup_temp_dir()
        sys.exit(1)
    
    # Verify data exists
    if not data_path.exists():
        print(f"✗ Error: Dataset not found: {data_path}")
        print("Please run scripts/09_generate_sentiment.py first.")
        sys.exit(1)
    
    # --- PARALLELIZATION CONFIG ---
    # Use 16 environments for 16 CPU threads (Ryzen 3700X)
    # This allows collecting 16 steps of data in parallel
    n_envs = min(16, cpu_cores)  # Don't exceed available cores
    
    print("⚙️  Configuration:")
    print(f"  Parallel Environments: {n_envs}")
    print(f"  Device: {device}")
    print()
    
    # --- CREATE VECTORIZED ENVIRONMENT ---
    print("Step 2: Creating vectorized environment...")
    print(f"  Spawning {n_envs} parallel environments...")
    
    start_init = time.time()
    
    # Create factory function with memory-mapped data config
    # Subprocesses will load from memory-mapped files (instant, zero-copy)
    make_env_fn = make_env_factory(data_config)
    
    # SubprocVecEnv = True parallelism (separate processes)
    # Each subprocess loads from memory-mapped files (zero-copy, instant access)
    env = make_vec_env(
        make_env_fn,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
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
    print(f"✅ Environments ready in {init_time:.2f}s")
    print(f"   (Total startup time: {data_load_time + init_time:.2f}s)")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()
    
    # --- HYPERPARAMETERS FOR RTX 4070 Ti SUPER ---
    print("Step 3: Configuring PPO with GPU-optimized hyperparameters...")
    
    # Buffer Size = n_steps * n_envs
    # Target: ~32,000 steps per update
    # 2048 * 16 = 32,768 steps per update
    n_steps = 2048
    
    # Batch Size: Massive for GPU saturation
    # 4070 Ti Super has 16GB VRAM, can handle large batches
    # Powers of 2 are GPU-friendly
    batch_size = 4096 if device == "cuda" else 256
    
    # Policy: Deep & Wide Network
    # Default is [64, 64] - too small for complex patterns
    # [512, 512, 512] = 10x larger, GPU handles it easily
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 512],  # Policy network (actor)
            vf=[512, 512, 512]   # Value network (critic)
        ),
        activation_fn=torch.nn.ReLU
    )
    
    print(f"  n_steps: {n_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  Network architecture: 512x512x512")
    print(f"  Buffer size per update: {n_steps * n_envs:,} steps")
    print()
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,  # Reuse data 10 times (GPU is fast enough)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Exploration bonus
        vf_coef=0.5,    # Value function loss weight
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_dir) if _check_tensorboard() else None
    )
    
    print("✓ PPO agent initialized")
    print()
    
    # --- CALLBACKS ---
    print("Step 4: Setting up training callbacks...")
    
    # Checkpoint callback: Save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save every 100k steps
        save_path=str(model_dir),
        name_prefix="hdrl_crypto_optimized"
    )
    
    # Evaluation callback: Track best model
    # Create a separate eval environment (single process for consistency)
    # Wrap in DummyVecEnv first (VecNormalize requires a VecEnv)
    def make_eval_env():
        env = FastCryptoEnv(data_path=data_path)
        env = Monitor(env, str(log_dir / "eval"))
        return env
    
    eval_env = DummyVecEnv([make_eval_env])
    
    # Load normalization stats if they exist, otherwise create new
    if (log_dir / "vec_normalize.pkl").exists():
        eval_env = VecNormalize.load(str(log_dir / "vec_normalize.pkl"), eval_env)
        eval_env.training = False  # Disable training mode for evaluation
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=True,
            training=False
        )
    
    # Early stopping callback: Stop if no improvement
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,  # Stop if no new record after 10 checks
        min_evals=20,  # Warmup period (evaluate at least 20 times first)
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(log_dir / "evaluations"),
        eval_freq=50000,  # Evaluate every 50k steps
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback  # Trigger early stopping on new best
    )
    
    print("✓ Callbacks configured")
    print()
    
    # --- TRAINING ---
    print("Step 4: Starting training...")
    print("  Watch for increasing 'ep_rew_mean' in logs")
    print("  Training will use maximum hardware resources")
    print()
    
    # Total timesteps: 10M for thorough training
    # With 16 parallel envs, this is ~625k environment steps
    # Should complete in <1 hour on RTX 4070 Ti Super
    total_timesteps = 10_000_000
    
    start_time = time.time()
    
    try:
        # Check if rich is available for progress bar
        try:
            import rich
            use_progress_bar = True
        except ImportError:
            use_progress_bar = False
            print("  ⚠ Progress bar disabled (rich not installed)")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=use_progress_bar
        )
        print()
        print("✓ Training completed successfully!")
        print()
    except KeyboardInterrupt:
        print()
        print("⚠ Training interrupted by user")
        print("  Model checkpoints saved in:", model_dir)
        print()
    finally:
        # Always save final model and normalization stats
        print("Step 6: Saving final model and normalization stats...")
        
        # Save final model state
        final_model_path = model_dir / "final_model"
        model.save(str(final_model_path))
        print(f"✓ Final model saved to: {final_model_path}.zip")
        
        # Save normalization stats (critical for inference)
        vec_normalize_path = model_dir / "vec_normalize.pkl"
        env.save(str(vec_normalize_path))
        print(f"✓ Normalization stats saved to: {vec_normalize_path}")
        
        training_time = (time.time() - start_time) / 60
        print(f"✓ Training time: {training_time:.2f} minutes")
        print()
    
    # --- TEST PREDICTIONS ---
    print("Step 7: Testing model predictions...")
    
    # Load normalization for consistent testing
    test_env = FastCryptoEnv(data_path=data_path)
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecNormalize.load(str(vec_normalize_path), test_env)
    
    obs = test_env.reset()
    
    print("  Sample predictions (first 5 steps):")
    for i in range(5):
        action, _states = model.predict(obs, deterministic=True)
        
        # Normalize for display
        action = action.flatten()  # Ensure 1D array
        action_sum = action.sum()
        if action_sum > 0:
            normalized_weights = action / action_sum
        else:
            normalized_weights = action
        
        # Convert to list for safe formatting
        weights_list = normalized_weights.tolist()
        
        print(f"    Step {i+1}:")
        print(f"      Portfolio weights: {weights_list}")
        print(f"      Cash weight: {weights_list[-1]:.3f}")
        
        obs, reward, done, info = test_env.step(action)
        
        if done[0]:  # DummyVecEnv returns done as array
            break
    
    print()
    
    # --- FINAL SUMMARY ---
    print("=" * 80)
    print("✅ HDRL Training Complete (Optimized)")
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

