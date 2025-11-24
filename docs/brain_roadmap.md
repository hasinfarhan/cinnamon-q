# Brain Development Roadmap

## Current State Analysis

### ✅ What's Working

1. **Multi-Modal Data Pipeline**
   - ✅ Technical: Returns, Volatility
   - ✅ Fundamental: Commit Count (developer activity)
   - ✅ Sentiment: Utility Score (LLM-scored news)

2. **Bicameral Brain Architecture (v2.1)**
   - ✅ ML Analyst (PPO): Generates expected returns
   - ✅ Risk Manager (Semicovariance): Calculates downside risk
   - ✅ Optimizer (Sortino): Maximizes risk-adjusted returns

3. **Training Infrastructure**
   - ✅ Optimized for high-end hardware (RTX 4070 Ti Super)
   - ✅ Memory-mapped data loading
   - ✅ Automated retraining pipeline

### ⚠️ Gaps Identified

1. **Technical Analysis Features Not Integrated**
   - ✅ Created: `crypto_deep_v2.parquet` with RSI, MACD, BBW, ATR, SMA Ratio
   - ❌ **NOT USED**: Training still uses `crypto_training_data.parquet` (without TA)
   - **Impact**: Brain is missing professional TA signals

2. **Simple Model Architecture**
   - Current: MLP (Multi-Layer Perceptron) [512, 512, 512]
   - **Limitation**: No temporal awareness (LSTM/Transformer)
   - **Limitation**: No attention mechanisms for multi-asset relationships

3. **No Feature Importance Analysis**
   - Don't know which features matter most
   - Could be wasting compute on irrelevant features

4. **No Interpretability**
   - Black box: Can't explain why the bot made a decision
   - **Risk**: Hard to debug or trust with real capital

---

## 🎯 Priority Roadmap

### Phase 5.1: Integrate Technical Analysis Features (HIGH PRIORITY)

**Status**: Features created but not used  
**Impact**: High - Professional TA signals are ready but unused  
**Effort**: Medium (2-3 hours)

#### Tasks

1. **Update Training Pipeline**
   - Modify `src/pipeline/training.py` to use `crypto_deep_v2.parquet` instead of `crypto_training_data.parquet`
   - Update `FastCryptoEnv` to include TA features in observation space
   - Retrain model with enriched features

2. **Update Inference Engine**
   - Modify `PPOStrategy._prepare_observation()` to calculate TA features from live data
   - Ensure feature engineering matches training (prevent Training/Serving Skew)

3. **Feature Engineering Module**
   - Use `src/features/indicators.py` in both training and inference
   - Centralize feature calculation logic

**Expected Outcome**: Brain sees RSI, MACD, Bollinger Bands, ATR, SMA Ratio - should improve performance

---

### Phase 5.2: Model Architecture Upgrade (MEDIUM PRIORITY)

**Status**: Current MLP is simple but functional  
**Impact**: Medium-High - Could capture temporal patterns  
**Effort**: High (1-2 days)

#### Option A: LSTM/GRU for Temporal Patterns

**Why**: Current MLP treats each timestep independently. LSTM can learn sequences.

**Implementation**:
- Replace MLP policy with LSTM policy in PPO
- Use `MlpLstmPolicy` from stable-baselines3
- Observation: Keep current features, add sequence dimension

**Pros**:
- Captures temporal dependencies
- Better for trend-following strategies
- Built into stable-baselines3

**Cons**:
- More complex, slower training
- May overfit on short sequences

#### Option B: Transformer for Multi-Asset Attention

**Why**: Current model doesn't explicitly model relationships between assets.

**Implementation**:
- Custom policy network with Transformer encoder
- Self-attention across assets
- Positional encoding for time

**Pros**:
- Models asset correlations explicitly
- State-of-the-art architecture
- Interpretable attention weights

**Cons**:
- Requires custom implementation
- More hyperparameters to tune
- Higher memory usage

#### Recommendation: Start with Option A (LSTM)

**Rationale**: Easier to implement, proven in finance, incremental improvement

---

### Phase 5.3: Feature Importance & Selection (MEDIUM PRIORITY)

**Status**: No analysis of which features matter  
**Impact**: Medium - Could reduce noise, improve performance  
**Effort**: Low-Medium (4-6 hours)

#### Tasks

1. **Feature Importance Analysis**
   - Use SHAP (SHapley Additive exPlanations) to analyze feature contributions
   - Identify which features (returns, volatility, commits, utility, TA) matter most
   - Visualize feature importance

2. **Feature Selection**
   - Remove low-importance features
   - Reduce observation space (faster training, less overfitting)
   - Retrain with selected features

3. **Ablation Studies**
   - Train models with different feature subsets
   - Compare performance:
     - Baseline (current 4 features)
     - + TA features
     - + Only top N features
   - Document findings

**Expected Outcome**: Leaner, faster model with better performance

---

### Phase 5.4: Model Interpretability (LOW-MEDIUM PRIORITY)

**Status**: Black box - can't explain decisions  
**Impact**: Medium - Important for trust and debugging  
**Effort**: Medium (1 day)

#### Tasks

1. **SHAP Integration**
   - Add SHAP explainer to `PPOStrategy`
   - Generate explanations for each prediction
   - Store in database or log file

2. **Dashboard Enhancement**
   - Add "Brain Scan" tab to dashboard
   - Show feature contributions for current allocation
   - Visualize why bot chose each asset

3. **Decision Logging**
   - Log feature values and contributions for each decision
   - Enable post-hoc analysis of decisions
   - Help debug bad trades

**Expected Outcome**: Transparent, explainable AI decisions

---

### Phase 5.5: Multi-Strategy Ensemble (LOW PRIORITY)

**Status**: Single model approach  
**Impact**: Medium - Could improve robustness  
**Effort**: High (2-3 days)

#### Concept

Train multiple models with different:
- Architectures (MLP, LSTM, Transformer)
- Feature sets (Technical only, Fundamental only, All)
- Hyperparameters (learning rates, network sizes)

Then ensemble their predictions:
- **Voting**: Majority vote on allocations
- **Weighted Average**: Weight by historical performance
- **Stacking**: Meta-learner combines predictions

#### Implementation

1. **Multi-Model Training**
   - Extend training pipeline to train N models
   - Store all models in `models/ensemble/`

2. **Ensemble Strategy**
   - Create `EnsembleStrategy` class
   - Combines predictions from multiple models
   - Uses optimizer on combined expected returns

3. **Performance Tracking**
   - Track individual model performance
   - Track ensemble performance
   - Auto-remove underperforming models

**Expected Outcome**: More robust, diversified predictions

---

### Phase 5.6: Market Regime Detection (LOW PRIORITY)

**Status**: No regime awareness  
**Impact**: Medium - Could adapt strategy to market conditions  
**Effort**: Medium (1-2 days)

#### Concept

Detect market regimes:
- **Bull Market**: Trending up, low volatility
- **Bear Market**: Trending down, high volatility
- **Sideways**: Range-bound, mean-reverting

Switch strategies based on regime:
- Bull: Momentum/trend-following
- Bear: Defensive/cash-heavy
- Sideways: Mean-reversion

#### Implementation

1. **Regime Classifier**
   - Train classifier (SVM/Random Forest) on market features
   - Features: Volatility, trend, momentum, correlation
   - Output: Regime probability

2. **Regime-Aware Strategy**
   - Modify `PPOStrategy` to accept regime signal
   - Adjust risk aversion based on regime
   - Switch model weights in ensemble

3. **Backtesting**
   - Test regime detection accuracy
   - Compare regime-aware vs. regime-agnostic performance

**Expected Outcome**: Adaptive strategy that adjusts to market conditions

---

## 🚀 Quick Wins (Do First)

### 1. Integrate TA Features (Phase 5.1) ⭐ **START HERE**

**Why**: Features are already created, just need to wire them up. High impact, low risk.

**Steps**:
1. Update `FastCryptoEnv` to load TA features from `crypto_deep_v2.parquet`
2. Update observation space to include TA features (5 features per asset instead of 4)
3. Update `PPOStrategy._prepare_observation()` to calculate TA from live data
4. Retrain model with enriched features
5. Compare performance: Old vs. New

**Time**: 2-3 hours  
**Risk**: Low (backward compatible)

---

### 2. Feature Importance Analysis (Phase 5.3)

**Why**: Quick to implement, high value for understanding the model.

**Steps**:
1. Install SHAP: `pip install shap`
2. Add feature importance analysis script
3. Run on trained model
4. Visualize results
5. Document findings

**Time**: 2-3 hours  
**Risk**: None (analysis only, no code changes)

---

## 📊 Success Metrics

### Phase 5.1 (TA Integration)
- ✅ Model trained with TA features
- ✅ Inference uses TA features
- ✅ Performance improvement: +X% Sortino Ratio

### Phase 5.2 (Architecture Upgrade)
- ✅ LSTM/Transformer model trained
- ✅ Performance improvement: +X% Sortino Ratio
- ✅ Training time acceptable (< 2 hours)

### Phase 5.3 (Feature Selection)
- ✅ Feature importance report generated
- ✅ Reduced feature set identified
- ✅ Performance maintained or improved

### Phase 5.4 (Interpretability)
- ✅ SHAP explanations generated
- ✅ Dashboard shows feature contributions
- ✅ Decision logging implemented

---

## 🎯 Recommended Next Steps

1. **Immediate (This Week)**:
   - ✅ Phase 5.1: Integrate TA Features
   - ✅ Phase 5.3: Feature Importance Analysis

2. **Short Term (Next 2 Weeks)**:
   - Phase 5.2: LSTM Architecture Upgrade
   - Phase 5.4: Basic Interpretability

3. **Long Term (Next Month)**:
   - Phase 5.5: Multi-Strategy Ensemble
   - Phase 5.6: Market Regime Detection

---

## 💡 Additional Ideas (Future)

- **Online Learning**: Update model weights incrementally (not just retraining)
- **Transfer Learning**: Pre-train on stock data, fine-tune on crypto
- **Reinforcement Learning from Human Feedback (RLHF)**: Let human trader guide the bot
- **Multi-Timeframe**: Train separate models for different timeframes (15m, 1h, 1d)
- **Causal Inference**: Understand cause-and-effect relationships in features
- **Adversarial Training**: Make model robust to market manipulation

---

## Summary

**Current State**: Solid foundation with multi-modal data and Bicameral Brain architecture.

**Biggest Gap**: Technical Analysis features are created but not used.

**Recommended Priority**: 
1. **Phase 5.1** (Integrate TA) - High impact, low effort
2. **Phase 5.3** (Feature Importance) - Quick win, high value
3. **Phase 5.2** (LSTM Upgrade) - Medium effort, medium-high impact

**Goal**: Transform from "Good Brain" to "Great Brain" by leveraging all available data and improving architecture.

