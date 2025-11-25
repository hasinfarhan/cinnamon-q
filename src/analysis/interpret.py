"""
Model Interpretability - SHAP Analysis

This module provides SHAP (Shapley Additive Explanations) analysis for
understanding which features drive the PPO agent's decisions.

Phase 5.3: The "Glass Box" - Making the AI's reasoning transparent.

Uses KernelExplainer for robustness (works with any model structure).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from strategy.rl.data_loader import prepare_training_data
from strategy.rl.fast_env import FastCryptoEnv

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """
    Interpreter for PPO models using SHAP values (KernelExplainer).
    
    This class uses SHAP KernelExplainer to quantify the contribution
    of each feature to the model's decisions. KernelExplainer is model-agnostic
    and works with any Python function, making it robust for SB3 models.
    """
    
    def __init__(self, model_path: Path, vec_norm_path: Path, data_path: Path):
        """
        Initialize the model interpreter.
        
        Args:
            model_path: Path to trained PPO model (.zip file)
            vec_norm_path: Path to VecNormalize statistics (.pkl file)
            data_path: Path to training data (parquet file)
        """
        logger.info("=" * 80)
        logger.info("🔍 Initializing Model Interpreter")
        logger.info("=" * 80)
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model = PPO.load(str(model_path))
        logger.info("✓ Model loaded")
        
        # Load normalization stats
        logger.info(f"Loading normalization stats from {vec_norm_path}...")
        self.vec_norm = self._load_vec_normalize(vec_norm_path, data_path)
        logger.info("✓ Normalization stats loaded")
        
        # Store data path
        self.data_path = data_path
        
        # Feature names (will be set during analysis)
        self.feature_names: Optional[List[str]] = None
        
    def _extract_policy_network(self) -> torch.nn.Module:
        """
        Extract the PyTorch policy network from the SB3 PPO model.
        
        The policy network is the neural network that maps observations
        to actions. This is what we'll explain with SHAP.
        
        Returns:
            PyTorch Module representing the policy network
        """
        # SB3 PPO stores the policy in model.policy
        # The actual network is model.policy.mlp_extractor or model.policy.features_extractor
        # For PPO with MlpPolicy, the forward pass goes through:
        #   features_extractor -> mlp_extractor -> action_net/value_net
        
        # We need to create a wrapper that takes observations and returns actions
        # This is the "policy function" that SHAP will explain
        
        class PolicyWrapper(torch.nn.Module):
            """Wrapper to extract policy predictions for SHAP."""
            
            def __init__(self, sb3_policy):
                super().__init__()
                self.policy = sb3_policy
                self.eval()  # Set to evaluation mode
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Forward pass: observation -> action logits.
                
                Args:
                    x: Observation tensor (batch_size, n_features)
                
                Returns:
                    Action logits (batch_size, n_actions)
                """
                # Convert to tensor if needed
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                
                # Ensure correct device
                device = next(self.policy.parameters()).device
                x = x.to(device)
                
                # Extract features
                # For MlpPolicy, the policy has:
                #   - features_extractor: extracts features from observation
                #   - mlp_extractor: shared MLP
                #   - action_net: outputs action logits
                
                # Get latent features
                latent_pi = self.policy.get_latent(x)
                
                # Get action logits (before normalization)
                action_logits = self.policy.action_net(latent_pi)
                
                return action_logits
        
        return PolicyWrapper(self.model.policy)
    
    def _get_network_info(self) -> str:
        """Get information about the policy network architecture."""
        total_params = sum(p.numel() for p in self.policy_net.parameters())
        trainable_params = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        
        # Try to get input/output sizes
        try:
            # Get first layer input size
            if hasattr(self.policy_net.policy, 'features_extractor'):
                input_size = self.policy_net.policy.features_extractor[0].in_features
            else:
                input_size = "unknown"
            
            # Get output size (action space)
            if hasattr(self.policy_net.policy, 'action_net'):
                output_size = self.policy_net.policy.action_net.out_features
            else:
                output_size = "unknown"
            
            return f"{total_params:,} params ({trainable_params:,} trainable), {input_size} -> {output_size}"
        except:
            return f"{total_params:,} params ({trainable_params:,} trainable)"
    
    def _load_vec_normalize(self, vec_norm_path: Path, data_path: Path) -> VecNormalize:
        """
        Load VecNormalize statistics.
        
        Args:
            vec_norm_path: Path to VecNormalize .pkl file
            data_path: Path to training data (for creating dummy env)
        
        Returns:
            VecNormalize object with loaded statistics
        """
        # Create a dummy environment to load VecNormalize
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="shap_data_"))
        
        try:
            data_arrays = prepare_training_data(data_path, temp_dir)
            dummy_env = FastCryptoEnv(data_config=data_arrays)
            dummy_vec_env = DummyVecEnv([lambda: dummy_env])
            
            # Load normalization stats
            vec_norm = VecNormalize.load(str(vec_norm_path), dummy_vec_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
            
            return vec_norm
        finally:
            # Cleanup temp directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _prepare_background_data(self, sample_size: int = 1000) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare background and test data for SHAP explainer.
        
        Loads data from parquet, normalizes it, and splits into background
        (for baseline) and test (to explain) sets.
        
        Args:
            sample_size: Number of samples for background and test (each)
        
        Returns:
            Tuple of (background_data, test_data, feature_names)
        """
        logger.info(f"Preparing data ({sample_size} background + {sample_size} test samples)...")
        
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="shap_data_"))
        
        try:
            # Load and preprocess data
            data_arrays = prepare_training_data(self.data_path, temp_dir)
            has_ta = data_arrays.get('has_ta_features', False)
            
            # Create environment to generate observations
            env = FastCryptoEnv(data_config=data_arrays)
            
            # Generate observations by stepping through the environment
            observations = []
            n_samples = 0
            total_needed = sample_size * 2  # Background + test
            max_steps = min(env.n_steps, total_needed * 2)
            
            # Reset and collect observations
            obs, _ = env.reset()
            observations.append(obs.copy())
            n_samples += 1
            
            for step in range(max_steps):
                if n_samples >= total_needed:
                    break
                
                # Random action
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                
                if terminated or truncated:
                    obs, _ = env.reset()
                
                observations.append(obs.copy())
                n_samples += 1
            
            # Split into background and test
            all_obs = np.array(observations[:total_needed])
            background = all_obs[:sample_size]
            test = all_obs[sample_size:sample_size * 2]
            
            # Generate feature names
            feature_names = self._generate_feature_names(env, has_ta)
            self.feature_names = feature_names
            
            logger.info(f"✓ Prepared {len(background)} background + {len(test)} test samples")
            logger.info(f"  Observation shape: {background.shape}")
            logger.info(f"  Features: {len(feature_names)}")
            
            return background, test, feature_names
            
        finally:
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _generate_feature_names(self, env: FastCryptoEnv, has_ta: bool) -> List[str]:
        """
        Generate human-readable feature names.
        
        Args:
            env: FastCryptoEnv instance
            has_ta: Whether TA features are present
        
        Returns:
            List of feature names
        """
        names = []
        
        # Base features per asset
        base_features = ['returns', 'volatility', 'commit_count', 'utility_score']
        ta_features = ['RSI', 'MACD', 'BBW', 'ATR', 'SMA_RATIO']
        
        for asset in env.assets:
            # Base features
            for feat in base_features:
                names.append(f"{asset}_{feat}")
            
            # TA features (if present)
            if has_ta:
                for feat in ta_features:
                    names.append(f"{asset}_{feat}")
        
        # Portfolio-level features
        names.append("portfolio_value")
        names.append("cash_weight")
        
        return names
    
    def run_analysis(
        self,
        background: np.ndarray,
        test: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Run SHAP analysis using KernelExplainer.
        
        Args:
            background: Background data for baseline (n_samples, n_features)
            test: Test data to explain (n_samples, n_features)
            feature_names: List of feature names
        
        Returns:
            SHAP values array (n_test_samples, n_features, n_actions)
        """
        logger.info("=" * 80)
        logger.info("🧮 Computing SHAP Values (KernelExplainer)")
        logger.info("=" * 80)
        logger.info("  This may take 5-15 minutes depending on model size...")
        logger.info("  KernelExplainer is slower but more robust than DeepExplainer")
        logger.info()
        
        # Initialize KernelExplainer
        logger.info("Initializing KernelExplainer...")
        explainer = shap.KernelExplainer(
            self._predict_wrapper,
            background
        )
        logger.info("✓ Explainer initialized")
        
        # Compute SHAP values
        logger.info("Computing SHAP values (this is the slow part)...")
        logger.info("  Progress will be shown by SHAP...")
        
        shap_values = explainer.shap_values(test, nsamples=100)
        
        # Handle multi-output (if model outputs multiple actions)
        if isinstance(shap_values, list):
            # For multi-output, take the mean absolute SHAP values across outputs
            # This gives us overall feature importance
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        
        logger.info(f"✓ SHAP values computed: shape {shap_values.shape}")
        
        return shap_values
    
    def save_plot(
        self,
        shap_values: np.ndarray,
        test_data: np.ndarray,
        feature_names: List[str],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate and save SHAP summary plot (beeswarm plot).
        
        Args:
            shap_values: SHAP values array (n_samples, n_features) or (n_samples, n_features, n_actions)
            test_data: Test data used for explanation (n_samples, n_features)
            feature_names: List of feature names
            output_path: Path to save the plot (default: reports/shap_summary.png)
        
        Returns:
            Path to saved plot
        """
        if output_path is None:
            project_root = Path(__file__).parent.parent.parent
            output_path = project_root / "reports" / "shap_summary.png"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating SHAP summary plot...")
        
        # Handle multi-dimensional SHAP values (if we have multiple actions)
        # For summary plot, we want to show overall feature importance
        if len(shap_values.shape) == 3:
            # Average across actions
            shap_values_2d = np.mean(np.abs(shap_values), axis=2)
        else:
            shap_values_2d = shap_values
        
        # Create summary plot (beeswarm plot)
        shap.summary_plot(
            shap_values_2d,
            test_data,
            feature_names=feature_names,
            show=False,
            plot_size=(12, 8)
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Summary plot saved to {output_path}")
        
        return output_path
    
    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top K most important features based on mean absolute SHAP values.
        
        Args:
            shap_values: SHAP values array (n_samples, n_features)
            feature_names: List of feature names
            top_k: Number of top features to return
        
        Returns:
            List of (feature_name, mean_abs_shap_value) tuples, sorted by importance
        """
        # Calculate mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort by importance
        feature_importance = list(zip(feature_names, mean_abs_shap))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_k]

