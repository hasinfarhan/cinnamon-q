"""
Sortino Ratio Optimizer - The Negotiator (Sortino Edition)

This module solves the Mean-Variance Optimization problem using a Semicovariance
matrix to maximize the Sortino Ratio (return per unit of downside risk).

Key Insight: When you pass a Semicovariance matrix to EfficientFrontier.max_sharpe(),
it mathematically becomes a Sortino Ratio optimization because:
- Sharpe Ratio = (Return - RiskFree) / Total_Volatility
- Sortino Ratio = (Return - RiskFree) / Downside_Volatility

By using semicovariance (downside volatility) instead of covariance (total volatility),
we effectively optimize for Sortino Ratio.
"""

from typing import Optional

import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.exceptions import OptimizationError

from execution.models import TargetPortfolio


class SortinoRatioOptimizer:
    """
    Solves the Mean-Variance Optimization problem to maximize Sortino Ratio.
    
    This is the "Negotiator" in the Bicameral Brain architecture (Sortino Edition).
    It takes:
    - Expected returns (from ML Analyst)
    - Downside risk metrics (from Downside Risk Manager)
    And outputs optimal portfolio weights that maximize Sortino Ratio while
    enforcing hard Halal constraints.
    
    Mathematical Foundation:
    - When you pass a Semicovariance matrix to EfficientFrontier.max_sharpe(),
      it calculates: (Return - RiskFree) / Downside_Deviation
      This IS the Sortino Ratio (not Sharpe Ratio).
    
    Attributes:
        risk_free_rate: Risk-free rate for Sortino ratio calculation
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize the Sortino Ratio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sortino ratio calculation.
                          Default: 0.0 (appropriate for crypto)
        """
        self.risk_free_rate = risk_free_rate
    
    def optimize(
        self,
        expected_returns: pd.Series,
        downside_risk_matrix: pd.DataFrame
    ) -> TargetPortfolio:
        """
        Solve the Mean-Variance Optimization problem to maximize Sortino Ratio.
        
        The optimization problem:
        Maximize: Sortino Ratio = (w^T * μ - r_f) / sqrt(w^T * Σ_downside * w)
        
        Where:
        - μ = Expected returns (from ML Analyst)
        - Σ_downside = Semicovariance matrix (downside risk, from Risk Manager)
        - r_f = Risk-free rate
        
        Subject to:
        - Long Only: 0 ≤ w_i ≤ 1 (no shorting)
        - Fully Invested: Σ w_i = 1 (no leverage)
        
        Args:
            expected_returns: Series of expected returns (alpha scores from ML).
                            Index: Ticker symbols (e.g., 'BTC', 'ETH')
                            Values: Expected returns (e.g., 0.05 for 5% expected gain)
            downside_risk_matrix: Semicovariance matrix from Downside Risk Manager.
                                 Shape: (n_assets, n_assets)
                                 Index and columns: Ticker symbols (must match expected_returns)
        
        Returns:
            TargetPortfolio with optimized weights (maximizing Sortino Ratio)
        
        Raises:
            ValueError: If expected_returns and downside_risk_matrix don't have matching assets
            OptimizationError: If optimization fails (e.g., infeasible constraints)
        """
        # Validate inputs
        if expected_returns.empty:
            raise ValueError("Expected returns Series is empty")
        
        if downside_risk_matrix.empty:
            raise ValueError("Downside risk matrix is empty")
        
        # Ensure expected_returns and downside_risk_matrix have matching assets
        # Align them to common assets (intersection)
        common_assets = expected_returns.index.intersection(downside_risk_matrix.index)
        if len(common_assets) == 0:
            raise ValueError(
                "No common assets between expected_returns and downside_risk_matrix. "
                f"Expected returns: {list(expected_returns.index)}, "
                f"Downside risk: {list(downside_risk_matrix.index)}"
            )
        
        # Filter to common assets
        expected_returns_aligned = expected_returns[common_assets]
        downside_risk_aligned = downside_risk_matrix.loc[common_assets, common_assets]
        
        # Initialize Efficient Frontier
        # weight_bounds=(0, 1) enforces Long Only constraint (no shorting)
        # CRITICAL: We pass the Semicovariance matrix as the "covariance" parameter.
        # When EfficientFrontier.max_sharpe() uses this matrix, it effectively
        # calculates Sortino Ratio instead of Sharpe Ratio.
        ef = EfficientFrontier(
            expected_returns_aligned,
            downside_risk_aligned,  # This is semicovariance, not covariance!
            weight_bounds=(0, 1)  # Halal: Long Only
        )
        
        # Solve optimization problem
        # max_sharpe() with semicovariance matrix = max_sortino()
        try:
            # Maximize "Sharpe Ratio" (which becomes Sortino Ratio due to semicovariance input)
            weights_raw = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            
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

