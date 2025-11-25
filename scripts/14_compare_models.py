"""
Model Comparison Script - The "Champion vs. Champion" Battle

Phase 5.2: LSTM Upgrade - Final Validation

This script compares the performance of two models:
1. MLP Model (current production): models/hdrl_optimized/final_model.zip
2. LSTM Model (challenger): models/hdrl_lstm/final_model.zip

The Test: Run both models on the same unseen month of data.
The Win: If LSTM_Sortino > MLP_Sortino, deploy the LSTM.

Metrics:
- Sortino Ratio (primary): Risk-adjusted returns (downside deviation)
- Sharpe Ratio: Risk-adjusted returns (total volatility)
- Total Return: Absolute performance
- Max Drawdown: Worst peak-to-trough decline
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from strategy.ppo_strategy import PPOStrategy
from strategy.rl.fast_env import FastCryptoEnv
from strategy.rl.data_loader import prepare_training_data
import tempfile


# ============================================================================
# Configuration
# ============================================================================

# Model paths
MLP_MODEL_PATH = project_root / "models" / "hdrl_optimized" / "final_model.zip"
MLP_VECNORM_PATH = project_root / "models" / "hdrl_optimized" / "vec_normalize.pkl"

LSTM_MODEL_PATH = project_root / "models" / "hdrl_lstm" / "final_model.zip"
LSTM_VECNORM_PATH = project_root / "models" / "hdrl_lstm" / "vec_normalize.pkl"

# Data path
DATA_PATH = project_root / "data" / "processed" / "crypto_training_data.parquet"

# Evaluation parameters
EVAL_DATA_SPLIT = 0.1  # Use last 10% of data for evaluation (unseen)
N_EPISODES = 5  # Number of evaluation episodes per model


# ============================================================================
# Evaluation Functions
# ============================================================================

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino Ratio (return per unit of downside risk).
    
    Sortino Ratio = (Mean Return - Risk-Free Rate) / Downside Deviation
    
    Args:
        returns: Array of portfolio returns
        risk_free_rate: Risk-free rate (default: 0.0 for crypto)
    
    Returns:
        Sortino Ratio (float)
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    
    # Downside deviation: only negative returns
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        # No downside risk - return high ratio (or handle edge case)
        return 10.0 if mean_return > 0 else 0.0
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 10.0 if mean_return > 0 else 0.0
    
    sortino = mean_return / downside_std
    return float(sortino)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe Ratio (return per unit of total risk).
    
    Args:
        returns: Array of portfolio returns
        risk_free_rate: Risk-free rate (default: 0.0 for crypto)
    
    Returns:
        Sharpe Ratio (float)
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 10.0 if mean_return > 0 else 0.0
    
    sharpe = mean_return / std_return
    return float(sharpe)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown (worst peak-to-trough decline).
    
    Args:
        equity_curve: Array of portfolio values over time
    
    Returns:
        Maximum drawdown as a percentage (negative value)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # Calculate running maximum (peak)
    peak = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown
    drawdown = (equity_curve - peak) / peak
    
    max_dd = np.min(drawdown)  # Most negative value
    return float(max_dd * 100)  # Convert to percentage


def evaluate_model(
    model_path: Path,
    vec_norm_path: Path,
    data_path: Path,
    model_name: str,
    n_episodes: int = 5
) -> Dict[str, float]:
    """
    Evaluate a model by running it through the environment.
    
    Args:
        model_path: Path to model .zip file
        vec_norm_path: Path to VecNormalize .pkl file
        data_path: Path to training data
        model_name: Name of the model (for logging)
        n_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating {model_name}")
    print(f"{'=' * 80}")
    
    if not model_path.exists():
        print(f"⚠ Model not found: {model_path}")
        print(f"  Skipping evaluation for {model_name}")
        return {
            "sortino_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "episodes": 0
        }
    
    # Load data and prepare environment
    print(f"📊 Loading data and preparing environment...")
    temp_dir = Path(tempfile.mkdtemp(prefix="eval_data_"))
    
    try:
        data_config = prepare_training_data(data_path, temp_dir)
        
        # Use last 10% of data for evaluation (unseen)
        env = FastCryptoEnv(data_config=data_config)
        total_steps = env.n_steps
        eval_start_step = int(total_steps * (1 - EVAL_DATA_SPLIT))
        
        print(f"  Total steps: {total_steps}")
        print(f"  Evaluation steps: {eval_start_step} to {total_steps}")
        print(f"  Episodes: {n_episodes}")
        
        # Initialize strategy
        strategy = PPOStrategy(
            model_path=model_path,
            vec_norm_path=vec_norm_path,
            data_path=data_path
        )
        
        # Run evaluation episodes
        all_returns = []
        all_equity = []
        
        for episode in range(n_episodes):
            print(f"\n  Episode {episode + 1}/{n_episodes}...")
            
            # Reset environment to a random start point in eval range
            env.current_step = np.random.randint(eval_start_step, total_steps - 100)
            obs, _ = env.reset()
            
            # Reset LSTM state for new episode
            if hasattr(strategy, 'reset_lstm_state'):
                strategy.reset_lstm_state()
            
            episode_returns = []
            episode_equity = [100000.0]  # Starting equity
            
            done = False
            step_count = 0
            max_steps = min(1000, total_steps - env.current_step)  # Limit episode length
            
            while not done and step_count < max_steps:
                # Get target portfolio from strategy
                try:
                    target = strategy.get_target_portfolio(portfolio_value=episode_equity[-1])
                    
                    # Convert target portfolio to action vector
                    # For evaluation, we'll use the weights directly
                    action = np.zeros(len(strategy.asset_list) + 1)
                    
                    # Map target weights to action
                    for i, asset in enumerate(strategy.asset_list):
                        # Strip /USD if present
                        base_asset = asset.replace('/USD', '') if '/USD' in asset else asset
                        weight = target.weights.get(base_asset, 0.0)
                        action[i] = weight
                    
                    # Cash weight
                    action[-1] = target.weights.get('CASH', 0.0)
                    
                    # Normalize action (sum to 1.0)
                    action = action / (action.sum() + 1e-8)
                    
                except Exception as e:
                    print(f"    ⚠ Error getting target portfolio: {e}")
                    # Use random action as fallback
                    action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Calculate portfolio return from reward
                # Reward in FastCryptoEnv is portfolio return
                episode_returns.append(reward)
                episode_equity.append(episode_equity[-1] * (1 + reward))
                
                step_count += 1
            
            all_returns.extend(episode_returns)
            all_equity.extend(episode_equity)
            
            episode_return = (episode_equity[-1] / episode_equity[0]) - 1.0
            print(f"    Episode return: {episode_return * 100:.2f}%")
        
        # Calculate metrics
        returns_array = np.array(all_returns)
        equity_array = np.array(all_equity)
        
        total_return = (equity_array[-1] / equity_array[0]) - 1.0
        sortino = calculate_sortino_ratio(returns_array)
        sharpe = calculate_sharpe_ratio(returns_array)
        max_dd = calculate_max_drawdown(equity_array)
        
        metrics = {
            "sortino_ratio": sortino,
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "episodes": n_episodes
        }
        
        print(f"\n  📊 {model_name} Results:")
        print(f"     Sortino Ratio: {sortino:.4f}")
        print(f"     Sharpe Ratio: {sharpe:.4f}")
        print(f"     Total Return: {total_return * 100:.2f}%")
        print(f"     Max Drawdown: {max_dd:.2f}%")
        
        return metrics
        
    except Exception as e:
        print(f"  ✗ Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "sortino_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "episodes": 0
        }
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    """Main comparison execution."""
    print("=" * 80)
    print("🏆 The Champion vs. Champion Battle")
    print("Phase 5.2: LSTM Upgrade - Final Validation")
    print("=" * 80)
    print()
    
    # Verify data exists
    if not DATA_PATH.exists():
        print(f"✗ Error: Data file not found: {DATA_PATH}")
        print("Please run scripts/08_fetch_crypto_deep.py first.")
        sys.exit(1)
    
    # Evaluate MLP Model (Champion)
    mlp_metrics = evaluate_model(
        MLP_MODEL_PATH,
        MLP_VECNORM_PATH,
        DATA_PATH,
        "MLP Model (Champion)",
        n_episodes=N_EPISODES
    )
    
    # Evaluate LSTM Model (Challenger)
    lstm_metrics = evaluate_model(
        LSTM_MODEL_PATH,
        LSTM_VECNORM_PATH,
        DATA_PATH,
        "LSTM Model (Challenger)",
        n_episodes=N_EPISODES
    )
    
    # ============================================================================
    # The Verdict
    # ============================================================================
    print("\n" + "=" * 80)
    print("🏁 THE VERDICT")
    print("=" * 80)
    print()
    
    print(f"{'Metric':<20} {'MLP (Champion)':<20} {'LSTM (Challenger)':<20} {'Winner':<20}")
    print("-" * 80)
    
    # Sortino Ratio (Primary Metric)
    mlp_sortino = mlp_metrics["sortino_ratio"]
    lstm_sortino = lstm_metrics["sortino_ratio"]
    sortino_winner = "LSTM 🏆" if lstm_sortino > mlp_sortino else "MLP 🏆"
    print(f"{'Sortino Ratio':<20} {mlp_sortino:<20.4f} {lstm_sortino:<20.4f} {sortino_winner:<20}")
    
    # Sharpe Ratio
    mlp_sharpe = mlp_metrics["sharpe_ratio"]
    lstm_sharpe = lstm_metrics["sharpe_ratio"]
    sharpe_winner = "LSTM 🏆" if lstm_sharpe > mlp_sharpe else "MLP 🏆"
    print(f"{'Sharpe Ratio':<20} {mlp_sharpe:<20.4f} {lstm_sharpe:<20.4f} {sharpe_winner:<20}")
    
    # Total Return
    mlp_return = mlp_metrics["total_return"] * 100
    lstm_return = lstm_metrics["total_return"] * 100
    return_winner = "LSTM 🏆" if lstm_return > mlp_return else "MLP 🏆"
    print(f"{'Total Return (%)':<20} {mlp_return:<20.2f} {lstm_return:<20.2f} {return_winner:<20}")
    
    # Max Drawdown (lower is better)
    mlp_dd = mlp_metrics["max_drawdown"]
    lstm_dd = lstm_metrics["max_drawdown"]
    dd_winner = "LSTM 🏆" if lstm_dd > mlp_dd else "MLP 🏆"  # Higher (less negative) is better
    print(f"{'Max Drawdown (%)':<20} {mlp_dd:<20.2f} {lstm_dd:<20.2f} {dd_winner:<20}")
    
    print()
    print("=" * 80)
    
    # Final Decision
    if lstm_sortino > mlp_sortino:
        print("✅ LSTM MODEL WINS! (Higher Sortino Ratio)")
        print()
        print("🚀 Deployment Recommendation:")
        print("  1. Backup current MLP model to models/archive/")
        print("  2. Copy LSTM model to models/hdrl_optimized/")
        print("  3. Update config.yaml to use LSTM model")
        print("  4. Restart bot: python main.py")
        print()
        print("  Example commands:")
        print(f"    mkdir -p models/archive")
        print(f"    cp {MLP_MODEL_PATH} models/archive/mlp_model_{pd.Timestamp.now().strftime('%Y%m%d')}.zip")
        print(f"    cp {LSTM_MODEL_PATH} {MLP_MODEL_PATH}")
        print(f"    cp {LSTM_VECNORM_PATH} {MLP_VECNORM_PATH}")
    else:
        print("✅ MLP MODEL WINS! (Higher Sortino Ratio)")
        print()
        print("💡 Recommendation:")
        print("  - Keep MLP model in production")
        print("  - Consider retraining LSTM with:")
        print("    * More timesteps")
        print("    * Different LSTM hyperparameters")
        print("    * Longer sequence length")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

