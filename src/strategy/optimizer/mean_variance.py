"""
Mean-Variance Optimizer - The Negotiator

This module solves the Mean-Variance Optimization (MVO) problem to find
optimal portfolio weights that maximize risk-adjusted returns while enforcing
Halal constraints (Long Only, No Leverage).
"""

from typing import Optional

import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.exceptions import OptimizationError

from execution.models import TargetPortfolio


class MeanVarianceOptimizer:
    """
    Solves the Mean-Variance Optimization problem.
    
    This is the "Negotiator" in the Bicameral Brain architecture. It takes:
    - Expected returns (from ML Analyst)
    - Risk metrics (from Quant Risk Manager)
    And outputs optimal portfolio weights that maximize Sharpe Ratio while
    enforcing hard Halal constraints.
    
    Attributes:
        method: Optimization method ("max_sharpe", "min_volatility", etc.)
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
    """
    
    def __init__(
        self,
        method: str = "max_sharpe",
        risk_free_rate: float = 0.0
    ):
        """
        Initialize the mean-variance optimizer.
        
        Args:
            method: Optimization objective.
                   Options:
                   - "max_sharpe": Maximize Sharpe Ratio (default)
                   - "min_volatility": Minimize portfolio volatility
                   - "efficient_risk": Maximize return for given risk
                   - "efficient_return": Minimize risk for given return
            risk_free_rate: Risk-free rate for Sharpe ratio calculation.
                          Default: 0.0 (appropriate for crypto)
        """
        self.method = method
        self.risk_free_rate = risk_free_rate
        
        # Validate method
        valid_methods = ["max_sharpe", "min_volatility", "efficient_risk", "efficient_return"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )
    
    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_volatility: Optional[float] = None,
        target_return: Optional[float] = None
    ) -> TargetPortfolio:
        """
        Solve the Mean-Variance Optimization problem.
        
        The optimization problem:
        Maximize: Sharpe Ratio = (w^T * μ - r_f) / sqrt(w^T * Σ * w)
        
        Subject to:
        - Long Only: 0 ≤ w_i ≤ 1 (no shorting)
        - Fully Invested: Σ w_i = 1 (no leverage)
        
        Args:
            expected_returns: Series of expected returns (alpha scores from ML).
                            Index: Ticker symbols (e.g., 'BTC', 'ETH')
                            Values: Expected returns (e.g., 0.05 for 5% expected gain)
            cov_matrix: Covariance matrix from Quant Risk Manager.
                       Shape: (n_assets, n_assets)
                       Index and columns: Ticker symbols (must match expected_returns)
            target_volatility: Target volatility (for "efficient_risk" method)
            target_return: Target return (for "efficient_return" method)
        
        Returns:
            TargetPortfolio with optimized weights
        
        Raises:
            ValueError: If expected_returns and cov_matrix don't have matching assets
            OptimizationError: If optimization fails (e.g., infeasible constraints)
        """
        # Validate inputs
        if expected_returns.empty:
            raise ValueError("Expected returns Series is empty")
        
        if cov_matrix.empty:
            raise ValueError("Covariance matrix is empty")
        
        # Ensure expected_returns and cov_matrix have matching assets
        # Align them to common assets (intersection)
        common_assets = expected_returns.index.intersection(cov_matrix.index)
        if len(common_assets) == 0:
            raise ValueError(
                "No common assets between expected_returns and cov_matrix. "
                f"Expected returns: {list(expected_returns.index)}, "
                f"Covariance: {list(cov_matrix.index)}"
            )
        
        # Filter to common assets
        expected_returns_aligned = expected_returns[common_assets]
        cov_matrix_aligned = cov_matrix.loc[common_assets, common_assets]
        
        # Initialize Efficient Frontier
        # weight_bounds=(0, 1) enforces Long Only constraint (no shorting)
        ef = EfficientFrontier(
            expected_returns_aligned,
            cov_matrix_aligned,
            weight_bounds=(0, 1)  # Halal: Long Only
        )
        
        # Solve optimization problem based on method
        try:
            if self.method == "max_sharpe":
                # Maximize Sharpe Ratio
                weights_raw = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            elif self.method == "min_volatility":
                # Minimize portfolio volatility
                weights_raw = ef.min_volatility()
            elif self.method == "efficient_risk":
                # Maximize return for given risk
                if target_volatility is None:
                    raise ValueError("target_volatility required for 'efficient_risk' method")
                weights_raw = ef.efficient_risk(target_volatility)
            elif self.method == "efficient_return":
                # Minimize risk for given return
                if target_return is None:
                    raise ValueError("target_return required for 'efficient_return' method")
                weights_raw = ef.efficient_return(target_return)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Clean weights (round small values to zero, normalize)
            weights_cleaned = ef.clean_weights()
            
        except OptimizationError as e:
            # If optimization fails, return equal weights as fallback
            print(f"  ⚠ Warning: Optimization failed ({e}), using equal weights")
            n_assets = len(common_assets)
            weights_cleaned = {asset: 1.0 / n_assets for asset in common_assets}
        
        # Ensure weights sum to 1.0 (Fully Invested constraint)
        # This is critical for Halal compliance (No Leverage)
        total_weight = sum(weights_cleaned.values())
        if abs(total_weight - 1.0) > 1e-6:
            # Renormalize to ensure sum = 1.0
            if total_weight > 1e-8:
                weights_cleaned = {
                    asset: weight / total_weight
                    for asset, weight in weights_cleaned.items()
                }
            else:
                # All weights are zero or very small, use equal weights
                print(f"  ⚠ Warning: Total weight is {total_weight}, using equal weights")
                n_assets = len(common_assets)
                weights_cleaned = {asset: 1.0 / n_assets for asset in common_assets}
        
        # Convert to TargetPortfolio format
        # Note: PyPortfolioOpt returns weights as a dict with base symbols (e.g., 'BTC')
        # We need to convert to Alpaca format (e.g., 'BTC/USD') for the Execution Engine
        weights_dict = {}
        for asset, weight in weights_cleaned.items():
            if weight > 1e-6:  # Only include non-negligible weights
                # Convert to Alpaca crypto format if needed
                if '/' not in asset and not asset.endswith('USD'):
                    alpaca_symbol = f"{asset}/USD"
                else:
                    alpaca_symbol = asset
                weights_dict[alpaca_symbol] = float(weight)
        
        # Construct TargetPortfolio
        try:
            target = TargetPortfolio(weights=weights_dict)
            return target
        except Exception as e:
            # If validation fails, return empty portfolio (100% cash)
            print(f"  ⚠ Warning: TargetPortfolio validation failed: {e}")
            print(f"  Returning empty portfolio (100% cash)")
            return TargetPortfolio(weights={})

