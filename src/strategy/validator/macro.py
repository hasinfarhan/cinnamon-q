"""
Macro Regime Gate - Market Regime Validator

This module implements a "Macro-Aware" risk gate that overrides the PPO Agent's
decisions based on global market conditions (VIX, BTC Trend, DXY).

Phase 7 Sprint 2: The Regime Validator

Philosophy: "The AI sees individual assets. The Validator sees the entire market.
When the market crashes, individual asset signals become noise. The Validator
forces a retreat to safety."

Regime Levels:
- 🟢 BULL (Green): BTC > SMA_200 AND VIX < 20 → Full Freedom (100% allocation allowed)
- 🟡 CAUTION (Yellow): VIX > 20 → Risk Capped (50% max allocation)
- 🔴 BEAR/CRASH (Red): BTC < SMA_200 OR VIX > 30 → Defense Mode (100% Cash/PAXG)
"""

import logging
from typing import Optional

import pandas as pd

from execution.models import TargetPortfolio

logger = logging.getLogger(__name__)


class MacroRegimeGate:
    """
    Macro-Aware Risk Validator that acts as a "Regime Filter."
    
    This validator overrides the PPO Agent's target portfolio based on global
    market conditions. It prevents the bot from buying the dip during a global
    financial collapse.
    
    Features:
    - Tracks VIX (Volatility Index - Fear Gauge)
    - Tracks BTC Trend (Price vs SMA_200)
    - Tracks DXY (US Dollar Index - optional)
    - Applies regime-based allocation limits
    
    Example:
        gate = MacroRegimeGate(vix_threshold_caution=20.0, vix_threshold_panic=30.0)
        safe_target = gate.validate(ai_target, market_data_df)
    """
    
    def __init__(
        self,
        vix_threshold_caution: float = 20.0,  # VIX > 20 = Caution
        vix_threshold_panic: float = 30.0,     # VIX > 30 = Panic
        btc_symbol: str = "BTC"                # BTC ticker symbol (without /USD)
    ):
        """
        Initialize the Macro Regime Gate.
        
        Args:
            vix_threshold_caution: VIX level that triggers caution mode (default: 20.0)
            vix_threshold_panic: VIX level that triggers panic mode (default: 30.0)
            btc_symbol: BTC ticker symbol for trend detection (default: "BTC")
        """
        self.vix_threshold_caution = vix_threshold_caution
        self.vix_threshold_panic = vix_threshold_panic
        self.btc_symbol = btc_symbol
        
        logger.info(
            f"🛡️ Macro Regime Gate initialized: "
            f"Caution VIX={vix_threshold_caution}, Panic VIX={vix_threshold_panic}"
        )
    
    def validate(
        self,
        target: TargetPortfolio,
        market_data: pd.DataFrame
    ) -> TargetPortfolio:
        """
        Validate target portfolio against current market regime.
        
        This method extracts macro indicators from market_data and applies
        regime-based allocation limits to the target portfolio.
        
        Args:
            target: The PPO Agent's proposed TargetPortfolio
            market_data: DataFrame with market data (must have columns: vix, ticker, close, btc_sma_200)
                        Can also have macro_trend_score column
        
        Returns:
            TargetPortfolio: The original target or an overridden target based on regime
        """
        # Step 1: Extract Latest Macro Indicators
        try:
            # Get latest VIX (global indicator, same for all rows)
            if 'vix' in market_data.columns:
                latest_vix = market_data['vix'].dropna().iloc[-1] if not market_data['vix'].dropna().empty else None
            else:
                logger.warning("⚠ VIX column not found in market_data, skipping macro validation")
                return target
            
            # Get latest BTC data
            btc_data = market_data[market_data['ticker'] == self.btc_symbol].copy()
            if btc_data.empty:
                # Try alternative formats
                btc_variants = ['BTC', 'BTC/USD', 'BTCUSD']
                for variant in btc_variants:
                    btc_data = market_data[market_data['ticker'] == variant].copy()
                    if not btc_data.empty:
                        break
            
            if btc_data.empty:
                logger.warning(f"⚠ BTC data not found in market_data, skipping macro validation")
                return target
            
            # Sort by timestamp and get latest
            if 'timestamp' in btc_data.columns:
                btc_data = btc_data.sort_values('timestamp')
            latest_btc = btc_data.iloc[-1]
            
            btc_price = float(latest_btc['close'])
            
            # Get BTC SMA_200 (if available in data, otherwise calculate)
            if 'btc_sma_200' in latest_btc and pd.notna(latest_btc['btc_sma_200']):
                btc_sma_200 = float(latest_btc['btc_sma_200'])
            elif 'macro_trend_score' in latest_btc and pd.notna(latest_btc['macro_trend_score']):
                # Calculate SMA from trend_score: macro_trend_score = (price - sma) / sma
                # So: sma = price / (1 + macro_trend_score)
                macro_trend = float(latest_btc['macro_trend_score'])
                btc_sma_200 = btc_price / (1.0 + macro_trend) if macro_trend != -1.0 else btc_price
            else:
                # Fallback: Calculate SMA_200 from BTC data if we have enough history
                if len(btc_data) >= 200:
                    btc_sma_200 = float(btc_data['close'].tail(200).mean())
                else:
                    logger.warning("⚠ Cannot determine BTC SMA_200, skipping macro validation")
                    return target
            
            # Get latest DXY (optional, for future use)
            dxy = None
            if 'dxy' in market_data.columns:
                dxy_series = market_data['dxy'].dropna()
                if not dxy_series.empty:
                    dxy = float(dxy_series.iloc[-1])
        
        except Exception as e:
            logger.warning(f"⚠ Error extracting macro indicators: {e}. Skipping macro validation.")
            return target
        
        # Step 2: Determine Regime
        is_bear_trend = btc_price < btc_sma_200
        is_high_volatility = latest_vix > self.vix_threshold_caution if latest_vix is not None else False
        is_panic = latest_vix > self.vix_threshold_panic if latest_vix is not None else False
        
        # Step 3: Apply Regime Logic
        # CASE 1: PANIC / BEAR (Red) - Defense Mode
        if is_panic or is_bear_trend:
            regime = "🔴 BEAR/CRASH"
            logger.warning(
                f"🚨 MACRO KILL SWITCH: {regime} detected. "
                f"BTC Price: ${btc_price:,.2f}, SMA_200: ${btc_sma_200:,.2f}, "
                f"VIX: {latest_vix:.2f}. Forcing Safety Mode (100% Cash/PAXG)."
            )
            
            # Override: Set all crypto weights to 0.0, move to CASH or PAXG
            validated_weights = {}
            
            # Check if PAXG exists in target (preferred safe haven)
            if 'PAXG' in target.weights or 'PAXG/USD' in target.weights:
                paxg_key = 'PAXG' if 'PAXG' in target.weights else 'PAXG/USD'
                validated_weights[paxg_key] = 1.0
            else:
                # Fallback to CASH
                validated_weights['CASH'] = 1.0
            
            # Set all other weights to 0.0
            for asset in target.weights:
                if asset not in validated_weights:
                    validated_weights[asset] = 0.0
            
            return TargetPortfolio(weights=validated_weights)
        
        # CASE 2: CAUTION (Yellow) - Risk Capped
        elif is_high_volatility:
            regime = "🟡 CAUTION"
            logger.info(
                f"⚠️ MACRO CAUTION: {regime} - Elevated VIX ({latest_vix:.2f}). "
                f"Capping exposure to 50% (scaling down crypto weights)."
            )
            
            # Scale down all crypto weights by 0.5x, move remainder to CASH
            validated_weights = {}
            total_crypto_weight = 0.0
            
            for asset, weight in target.weights.items():
                if asset == 'CASH':
                    # Keep cash as-is (will add more later)
                    validated_weights[asset] = weight
                else:
                    # Scale down crypto weights by 0.5
                    scaled_weight = weight * 0.5
                    validated_weights[asset] = scaled_weight
                    total_crypto_weight += scaled_weight
            
            # Add remainder to CASH (ensure sum = 1.0)
            cash_weight = validated_weights.get('CASH', 0.0)
            validated_weights['CASH'] = cash_weight + (1.0 - total_crypto_weight - cash_weight)
            
            # Ensure weights sum to 1.0 (safety check)
            total = sum(validated_weights.values())
            if abs(total - 1.0) > 1e-6:
                # Normalize
                validated_weights = {k: v / total for k, v in validated_weights.items()}
            
            return TargetPortfolio(weights=validated_weights)
        
        # CASE 3: BULL (Green) - Full Freedom
        else:
            regime = "🟢 BULL"
            logger.info(
                f"✅ MACRO GREEN: {regime} - All systems go. "
                f"BTC Price: ${btc_price:,.2f} > SMA_200: ${btc_sma_200:,.2f}, "
                f"VIX: {latest_vix:.2f} < {self.vix_threshold_caution}. "
                f"Full allocation freedom."
            )
            
            # Return original target (no override)
            return target

