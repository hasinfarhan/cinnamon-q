"""
Model Evaluation Stage - Challenger vs Champion

This module wraps the evaluation logic from scripts/11_retrain_model.py
into a callable function for the MLOps pipeline.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from strategy.rl.data_loader import prepare_training_data
from strategy.rl.fast_env import FastCryptoEnv

logger = logging.getLogger(__name__)


def calculate_sortino_ratio(returns_array: np.ndarray) -> float:
    """
    Calculate Sortino Ratio from returns array.
    
    Sortino Ratio = Mean Return / Downside Deviation
    Downside Deviation = std of negative returns only
    
    Args:
        returns_array: Array of portfolio returns
    
    Returns:
        Sortino Ratio (float)
    """
    if len(returns_array) == 0:
        return 0.0
    
    mean_return = np.mean(returns_array)
    downside_returns = returns_array[returns_array < 0]
    
    if len(downside_returns) == 0:
        # No downside, perfect score (or divide by zero protection)
        return 100.0 if mean_return > 0 else 0.0
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 100.0 if mean_return > 0 else 0.0
    
    sortino = mean_return / downside_std
    return float(sortino)


def evaluate_model(
    model: PPO,
    vec_norm_path: Path,
    data_path: Path,
    n_episodes: int = 10
) -> float:
    """
    Evaluate a model on the training data.
    
    Args:
        model: Trained PPO model
        vec_norm_path: Path to VecNormalize statistics
        data_path: Path to training data
        n_episodes: Number of evaluation episodes
    
    Returns:
        Average Sortino Ratio across episodes
    """
    import tempfile
    
    temp_dir = Path(tempfile.mkdtemp(prefix="eval_data_"))
    
    try:
        # Prepare data
        data_arrays = prepare_training_data(data_path, temp_dir)
        
        # Create evaluation environment
        eval_env_raw = FastCryptoEnv(data_config=data_arrays)
        eval_env = DummyVecEnv([lambda: eval_env_raw])
        
        # Load normalization stats
        if vec_norm_path.exists():
            eval_env = VecNormalize.load(str(vec_norm_path), eval_env)
        else:
            logger.warning(f"VecNormalize stats not found at {vec_norm_path}, using unnormalized")
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
        
        eval_env.training = False
        eval_env.norm_reward = False
        
        # Run evaluation episodes
        sortino_scores = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            episode_returns = []
            done = [False]
            
            step_count = 0
            max_steps = 1000  # Prevent infinite loops
            
            while not done[0] and step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_returns.append(float(reward[0]))
                step_count += 1
            
            if len(episode_returns) > 0:
                sortino = calculate_sortino_ratio(np.array(episode_returns))
                sortino_scores.append(sortino)
        
        avg_sortino = np.mean(sortino_scores) if sortino_scores else 0.0
        return avg_sortino
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def run_evaluation(
    champion_model_path: Path,
    challenger_model_path: Path,
    champion_vecnorm_path: Path,
    challenger_vecnorm_path: Path,
    data_path: Path,
    n_episodes: int = 10
) -> Tuple[float, float]:
    """
    Run evaluation stage: Compare Challenger vs Champion.
    
    Args:
        champion_model_path: Path to production (champion) model
        challenger_model_path: Path to new (challenger) model
        champion_vecnorm_path: Path to champion normalization stats
        challenger_vecnorm_path: Path to challenger normalization stats
        data_path: Path to training data for evaluation
        n_episodes: Number of evaluation episodes per model
    
    Returns:
        Tuple of (champion_score, challenger_score)
    """
    logger.info("=" * 80)
    logger.info("Stage 4: Model Evaluation (Challenger vs Champion)")
    logger.info("=" * 80)
    
    # Load models
    logger.info("Loading models...")
    champion_model = PPO.load(str(champion_model_path))
    challenger_model = PPO.load(str(challenger_model_path))
    
    # Evaluate Champion
    logger.info("Evaluating Champion model...")
    champion_score = evaluate_model(champion_model, champion_vecnorm_path, data_path, n_episodes)
    logger.info(f"  Champion Sortino Ratio: {champion_score:.4f}")
    
    # Evaluate Challenger
    logger.info("Evaluating Challenger model...")
    challenger_score = evaluate_model(challenger_model, challenger_vecnorm_path, data_path, n_episodes)
    logger.info(f"  Challenger Sortino Ratio: {challenger_score:.4f}")
    
    # Compare
    logger.info("=" * 80)
    if challenger_score > champion_score:
        improvement = ((challenger_score - champion_score) / champion_score) * 100
        logger.info(f"✓ Challenger WINS! Improvement: {improvement:.2f}%")
    else:
        degradation = ((champion_score - challenger_score) / champion_score) * 100
        logger.info(f"✗ Challenger loses. Degradation: {degradation:.2f}%")
    logger.info("=" * 80)
    
    return champion_score, challenger_score

