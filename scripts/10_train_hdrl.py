"""
HDRL Training Script - Train the Master Brain

Phase 3: The Crypto Expert Alpha Master - Step 3

This script trains a PPO agent on the multi-modal crypto dataset to learn
optimal portfolio allocation strategies that maximize the Sortino Ratio.
"""

import sys
from pathlib import Path

# Add src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Import environment directly to avoid dependency chain issues
sys.path.insert(0, str(project_root / "src" / "strategy" / "rl"))
from env import CryptoDeepEnv


def main():
    """Main training execution."""
    print("=" * 80)
    print("🧠 HDRL Brain Training - Master Agent")
    print("Phase 3: The Crypto Expert Alpha Master - Step 3")
    print("=" * 80)
    print()
    
    # Step 1: Initialize environment
    print("Step 1: Initializing CryptoDeepEnv...")
    env = CryptoDeepEnv()
    
    # Wrap with Monitor for logging
    log_dir = project_root / "logs" / "hdrl_training"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(env, str(log_dir))
    
    print(f"✓ Environment initialized")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")
    print()
    
    # Step 2: Initialize PPO agent
    print("Step 2: Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(log_dir / "tensorboard")
    )
    print("✓ PPO agent initialized")
    print()
    
    # Step 3: Setup callbacks
    print("Step 3: Setting up training callbacks...")
    
    # Evaluation callback
    eval_env = CryptoDeepEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "evaluations"),
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="hdrl_crypto"
    )
    
    print("✓ Callbacks configured")
    print()
    
    # Step 4: Train the agent
    print("Step 4: Training PPO agent...")
    print("  This may take a while. Watch for increasing 'ep_rew_mean' in logs.")
    print()
    
    total_timesteps = 20000
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        print()
        print("✓ Training completed successfully!")
        print()
    except KeyboardInterrupt:
        print()
        print("⚠ Training interrupted by user")
        print()
    
    # Step 5: Save final model
    print("Step 5: Saving final model...")
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "hdrl_crypto_v1"
    model.save(str(model_path))
    
    print(f"✓ Model saved to: {model_path}.zip")
    print()
    
    # Step 6: Test prediction
    print("Step 6: Testing model predictions...")
    obs, info = env.reset()
    
    print("  Sample predictions (first 5 steps):")
    for i in range(5):
        action, _states = model.predict(obs, deterministic=True)
        
        # Normalize action for display
        action_sum = action.sum()
        if action_sum > 0:
            normalized_weights = action / action_sum
        else:
            normalized_weights = action
        
        print(f"    Step {i+1}:")
        print(f"      Portfolio weights: {normalized_weights}")
        print(f"      Cash weight: {normalized_weights[-1]:.3f}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    print()
    
    # Final summary
    print("=" * 80)
    print("✅ HDRL Training Complete!")
    print("=" * 80)
    print()
    print("Model Summary:")
    print(f"  Model file: {model_path}.zip")
    print(f"  Training timesteps: {total_timesteps}")
    print(f"  Logs directory: {log_dir}")
    print()
    print("Next Steps:")
    print("  - Review training logs in logs/hdrl_training/")
    print("  - Check TensorBoard: tensorboard --logdir logs/hdrl_training/tensorboard")
    print("  - Integrate model into Execution Engine (Step 4)")
    print()


if __name__ == "__main__":
    main()

