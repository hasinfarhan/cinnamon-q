"""
Stats Recovery Script - Regenerate VecNormalize Statistics

This script regenerates the vec_normalize.pkl file by replaying training data
through a dummy environment. This is useful when normalization stats are missing
or need to be regenerated.

The script runs through the entire training dataset to calculate running mean
and variance statistics for the observation space.
"""

import sys
from pathlib import Path

# Add src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import environment directly to avoid dependency chain
sys.path.insert(0, str(project_root / "src" / "strategy" / "rl"))
from fast_env import FastCryptoEnv


def main():
    """Main execution to recover normalization statistics."""
    print("=" * 80)
    print("📊 VecNormalize Statistics Recovery")
    print("Regenerating normalization statistics from training data")
    print("=" * 80)
    print()
    
    # Paths
    data_path = project_root / "data" / "processed" / "crypto_training_data.parquet"
    output_path = project_root / "models" / "hdrl_optimized" / "vec_normalize.pkl"
    
    # Verify data exists
    if not data_path.exists():
        print(f"✗ Error: Dataset not found: {data_path}")
        print("Please run scripts/09_generate_sentiment.py first.")
        sys.exit(1)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Loading training data...")
    print(f"  Data path: {data_path}")
    
    # Initialize environment
    print("Step 2: Initializing FastCryptoEnv...")
    env = FastCryptoEnv(data_path=data_path)
    
    print(f"✓ Environment initialized")
    print(f"  Steps: {env.n_steps}")
    print(f"  Assets: {env.assets}")
    print()
    
    # Wrap in DummyVecEnv and VecNormalize
    print("Step 3: Creating vectorized normalized environment...")
    
    def make_env():
        return FastCryptoEnv(data_path=data_path)
    
    vec_env = DummyVecEnv([make_env])
    
    # Create VecNormalize with training=True to calculate stats
    norm_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        training=True  # Enable training mode to calculate stats
    )
    
    print("✓ Normalized environment created")
    print()
    
    # Step through entire dataset to calculate statistics
    print("Step 4: Replaying training data to calculate statistics...")
    print("  This may take a few minutes...")
    print()
    
    # Reset environment
    obs = norm_env.reset()
    
    # Step through all data
    total_steps = env.n_steps
    for step in range(total_steps):
        # Use random actions (we just need to step through to collect stats)
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, info = norm_env.step([action])
        
        # Reset if episode ended
        if done[0]:
            obs = norm_env.reset()
        
        # Progress indicator
        if (step + 1) % 1000 == 0:
            progress = ((step + 1) / total_steps) * 100
            print(f"  Progress: {step + 1}/{total_steps} steps ({progress:.1f}%)")
    
    print()
    print("✓ Statistics calculated")
    print()
    
    # Save normalization statistics
    print("Step 5: Saving normalization statistics...")
    print(f"  Output path: {output_path}")
    
    norm_env.save(str(output_path))
    
    print("✓ Statistics saved successfully")
    print()
    
    # Display summary
    print("=" * 80)
    print("✅ Statistics Recovery Complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Input data: {data_path}")
    print(f"  Output stats: {output_path}")
    print(f"  Steps processed: {total_steps:,}")
    print()
    print("Next Steps:")
    print("  - Use this file with PPOStrategy for inference")
    print("  - Path: models/hdrl_optimized/vec_normalize.pkl")
    print()


if __name__ == "__main__":
    main()

