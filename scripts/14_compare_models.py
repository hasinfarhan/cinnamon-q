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

from strategy.benchmarks.buy_and_hold import BuyAndHoldStrategy
from strategy.benchmarks.mean_reversion import MeanReversionStrategy
from strategy.benchmarks.trend import TrendFollowingStrategy
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

# Fee rate configuration
# CRITICAL: Match the fee_rate used during training!
# - Old models (trained before Phase 5.5 Sprint 1, Nov 27, 2025): 0.0 (no fees)
# - New models (trained after Phase 5.5 Sprint 1): 0.001 (0.1% fees)
# 
# To check: Look at model file modification date or training logs
# After retraining with fees, change this to 0.001
FEE_RATE = 0.0  # Change to 0.001 after retraining models with fees


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
        # FEE_RATE must match training environment to avoid training/serving skew
        env = FastCryptoEnv(data_config=data_config, fee_rate=FEE_RATE)
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
            
            # For evaluation, we'll use the model's raw action prediction
            # This avoids the bicameral brain complexity (risk calculator + optimizer)
            # which requires more data history than we have in the environment
            while not done and step_count < max_steps:
                # Get raw action from model (bypassing get_target_portfolio)
                # This is simpler and faster for evaluation
                try:
                    # Fetch data from environment's current state
                    # The environment has the data, we just need to prepare observation
                    # Get current observation from environment
                    current_obs = obs.copy()
                    
                    # Normalize observation using strategy's normalization
                    norm_obs = strategy.norm_env.normalize_obs(current_obs.reshape(1, -1))
                    
                    # Predict action directly from model
                    if strategy.is_lstm:
                        action, strategy.lstm_states = strategy.model.predict(
                            norm_obs,
                            state=strategy.lstm_states,
                            episode_start=np.array([step_count == 0]),  # True only on first step
                            deterministic=True
                        )
                        strategy.episode_starts = np.array([False])
                    else:
                        action, _ = strategy.model.predict(norm_obs, deterministic=True)
                    
                    # Ensure action is valid (non-negative, sums to 1.0)
                    action = np.clip(action, 0, 1)
                    action = action / (action.sum() + 1e-8)
                    
                except Exception as e:
                    print(f"    ⚠ Error predicting action: {e}")
                    # Use random action as fallback
                    action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # CRITICAL: Reward is Sortino Ratio (not percentage return)!
                # We need to use the actual portfolio_return from info dict
                # Reward can be Sortino Ratio (e.g., 2.0) which is NOT a 200% return!
                portfolio_return = info.get('portfolio_return', reward)
                
                # If portfolio_return is not available, use reward but clamp it
                # to reasonable bounds (reward might be Sortino ratio, not return)
                if 'portfolio_return' not in info:
                    # If reward looks like a ratio (> 1.0), it's probably Sortino
                    # In that case, we can't use it for compounding
                    # Fallback: use a small return estimate or skip this step
                    if abs(reward) > 1.0:
                        # Reward is likely Sortino ratio, not return
                        # Use a conservative estimate or skip compounding
                        portfolio_return = 0.0  # Skip this step's return
                    else:
                        # Reward might be actual return (small value like 0.01)
                        portfolio_return = reward
                
                # Clamp portfolio return to reasonable bounds (-50% to +50% per step)
                # This prevents compounding explosions from bugs
                portfolio_return = np.clip(portfolio_return, -0.5, 0.5)
                
                episode_returns.append(portfolio_return)
                episode_equity.append(episode_equity[-1] * (1 + portfolio_return))
                
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


def evaluate_benchmark(
    strategy,
    data_path: Path,
    strategy_name: str,
    n_episodes: int = 5
) -> Dict[str, float]:
    """
    Evaluate a benchmark strategy by running it through the environment.
    
    Args:
        strategy: Benchmark strategy instance (BuyAndHold, MeanReversion, TrendFollowing)
        data_path: Path to training data
        strategy_name: Name of the strategy (for logging)
        n_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating {strategy_name}")
    print(f"{'=' * 80}")
    
    # Load data and prepare environment
    print(f"📊 Loading data and preparing environment...")
    temp_dir = Path(tempfile.mkdtemp(prefix="eval_benchmark_"))
    
    try:
        data_config = prepare_training_data(data_path, temp_dir)
        
        # Use last 10% of data for evaluation (unseen)
        # FEE_RATE must match training environment to avoid training/serving skew
        env = FastCryptoEnv(data_config=data_config, fee_rate=FEE_RATE)
        total_steps = env.n_steps
        eval_start_step = int(total_steps * (1 - EVAL_DATA_SPLIT))
        
        print(f"  Total steps: {total_steps}")
        print(f"  Evaluation steps: {eval_start_step} to {total_steps}")
        print(f"  Episodes: {n_episodes}")
        
        # Run evaluation episodes
        all_returns = []
        all_equity = []
        
        for episode in range(n_episodes):
            print(f"\n  Episode {episode + 1}/{n_episodes}...")
            
            # Reset environment to a random start point in eval range
            env.current_step = np.random.randint(eval_start_step, total_steps - 100)
            obs, _ = env.reset()
            
            episode_returns = []
            episode_equity = [100000.0]  # Starting equity
            
            done = False
            step_count = 0
            max_steps = min(1000, total_steps - env.current_step)  # Limit episode length
            
            while not done and step_count < max_steps:
                # Get target portfolio from benchmark strategy
                try:
                    target = strategy.get_target_portfolio(portfolio_value=episode_equity[-1])
                    
                    # Convert target portfolio to action vector
                    action = np.zeros(env.action_space.shape[0])
                    
                    # Map target weights to action
                    # Get asset list from environment
                    env_assets = env.assets
                    
                    for i, asset in enumerate(env_assets):
                        # Try to match asset (handle /USD differences)
                        weight = 0.0
                        for target_asset, target_weight in target.weights.items():
                            if target_asset == "CASH":
                                continue
                            # Match asset (handle /USD suffix)
                            if (asset == target_asset or 
                                asset.replace('/USD', '') == target_asset or
                                asset == target_asset.replace('/USD', '')):
                                weight = target_weight
                                break
                        
                        action[i] = weight
                    
                    # Cash weight
                    action[-1] = target.weights.get('CASH', 0.0)
                    
                    # Normalize action (sum to 1.0)
                    action = np.clip(action, 0, 1)
                    action = action / (action.sum() + 1e-8)
                    
                except Exception as e:
                    print(f"    ⚠ Error getting target portfolio: {e}")
                    # Use random action as fallback
                    action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Use portfolio_return from info (not reward, which might be Sortino ratio)
                portfolio_return = info.get('portfolio_return', reward)
                portfolio_return = np.clip(portfolio_return, -0.5, 0.5)  # Clamp to prevent explosions
                
                episode_returns.append(portfolio_return)
                episode_equity.append(episode_equity[-1] * (1 + portfolio_return))
                
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
        
        print(f"\n  📊 {strategy_name} Results:")
        print(f"     Sortino Ratio: {sortino:.4f}")
        print(f"     Sharpe Ratio: {sharpe:.4f}")
        print(f"     Total Return: {total_return * 100:.2f}%")
        print(f"     Max Drawdown: {max_dd:.2f}%")
        
        return metrics
        
    except Exception as e:
        print(f"  ✗ Error evaluating {strategy_name}: {e}")
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
    
    # Evaluate Benchmark Strategies (Control Groups)
    print("\n" + "=" * 80)
    print("Evaluating Benchmark Strategies (Control Groups)")
    print("=" * 80)
    
    # Buy and Hold
    buy_hold_metrics = evaluate_benchmark(
        BuyAndHoldStrategy(data_path=DATA_PATH),
        DATA_PATH,
        "Buy and Hold (Passive)",
        n_episodes=N_EPISODES
    )
    
    # Mean Reversion (Bollinger Bands)
    mean_rev_metrics = evaluate_benchmark(
        MeanReversionStrategy(data_path=DATA_PATH, window=20, num_std_dev=2.0),  # Updated to Bollinger Bands
        DATA_PATH,
        "Mean Reversion (Bollinger Bands)",
        n_episodes=N_EPISODES
    )
    
    # Trend Following
    trend_metrics = evaluate_benchmark(
        TrendFollowingStrategy(data_path=DATA_PATH, window=200),
        DATA_PATH,
        "Trend Following (Momentum)",
        n_episodes=N_EPISODES
    )
    
    # ============================================================================
    # The Tournament - Full Comparison
    # ============================================================================
    print("\n" + "=" * 80)
    print("🏁 THE TOURNAMENT - FULL RESULTS")
    print("=" * 80)
    print()
    
    # Collect all results
    all_results = {
        "MLP (AI)": mlp_metrics,
        "LSTM (AI)": lstm_metrics,
        "Buy & Hold": buy_hold_metrics,
        "Mean Reversion": mean_rev_metrics,
        "Trend Following": trend_metrics
    }
    
    # Sort by Sortino Ratio (primary metric)
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["sortino_ratio"],
        reverse=True
    )
    
    print(f"{'Rank':<6} {'Strategy':<25} {'Sortino':<12} {'Sharpe':<12} {'Return %':<12} {'Max DD %':<12}")
    print("-" * 80)
    
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal} {rank:<4} {name:<25} {metrics['sortino_ratio']:<12.4f} "
              f"{metrics['sharpe_ratio']:<12.4f} {metrics['total_return']*100:<12.2f} "
              f"{metrics['max_drawdown']:<12.2f}")
    
    print()
    print("=" * 80)
    
    # Analysis
    winner_name, winner_metrics = sorted_results[0]
    ai_models = [("MLP", mlp_metrics), ("LSTM", lstm_metrics)]
    benchmarks = [("Buy & Hold", buy_hold_metrics), ("Mean Reversion", mean_rev_metrics), ("Trend Following", trend_metrics)]
    
    best_ai = max(ai_models, key=lambda x: x[1]["sortino_ratio"])
    best_benchmark = max(benchmarks, key=lambda x: x[1]["sortino_ratio"])
    
    print("📊 ANALYSIS")
    print("=" * 80)
    print()
    
    if winner_name in ["MLP (AI)", "LSTM (AI)"]:
        print(f"✅ AI WINS! {winner_name} has the highest Sortino Ratio")
        print(f"   AI outperformed all benchmarks - complexity adds value!")
    elif winner_name in ["Buy & Hold", "Mean Reversion", "Trend Following"]:
        print(f"⚠️  BENCHMARK WINS! {winner_name} has the highest Sortino Ratio")
        print(f"   This suggests:")
        print(f"   - Market regime favors simple strategies")
        print(f"   - AI may be overfitting or over-trading")
        print(f"   - Consider: Reduce trading frequency, add regime detection")
    
    print()
    print(f"Best AI Model: {best_ai[0]} (Sortino: {best_ai[1]['sortino_ratio']:.4f})")
    print(f"Best Benchmark: {best_benchmark[0]} (Sortino: {best_benchmark[1]['sortino_ratio']:.4f})")
    
    if best_ai[1]["sortino_ratio"] > best_benchmark[1]["sortino_ratio"]:
        print()
        print("✅ AI beats benchmarks - The brain is working!")
        if best_ai[0] == "LSTM":
            print()
            print("🚀 Deployment Recommendation: Deploy LSTM Model")
            print("  1. Backup current MLP model to models/archive/")
            print("  2. Copy LSTM model to models/hdrl_optimized/")
            print("  3. Restart bot: python main.py")
    else:
        print()
        print("⚠️  Benchmarks beat AI - Consider:")
        print("  - Retraining with more data")
        print("  - Adjusting reward function")
        print("  - Adding regime detection (switch strategies based on market conditions)")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

