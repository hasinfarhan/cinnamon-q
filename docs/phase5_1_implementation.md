# Phase 5.1: Deep Feature Engineering Integration

## Status: IN PROGRESS

### Goal
Integrate Technical Analysis (TA) features into the training and inference pipeline.

### Current State
- ✅ TA features implemented in `src/features/indicators.py` (manual implementation)
- ✅ `crypto_deep_v2.parquet` created with TA features (RSI, MACD, BBW, ATR, SMA Ratio)
- ❌ Training still uses `crypto_training_data.parquet` (without TA)
- ❌ Observation space hardcoded to 4 features per asset
- ❌ Inference doesn't calculate TA features from live data

### Implementation Tasks

#### Task 1: Update Data Loader ✅ (DONE)
- [x] Modified `src/strategy/rl/data_loader.py` to:
  - Detect TA features in input data
  - Normalize TA features per asset
  - Pivot TA features to wide format
  - Save TA arrays to disk for memory-mapped access
  - Return `has_ta_features` flag in data_config

#### Task 2: Update FastCryptoEnv (IN PROGRESS)
- [ ] Load TA arrays from data_config
- [ ] Update observation space: 4 → 9 features per asset
- [ ] Update observation construction to include TA features
- [ ] Handle backward compatibility (fallback to zeros if no TA)

#### Task 3: Update PPOStrategy (PENDING)
- [ ] Calculate TA features from live data using `TechnicalAnalysis`
- [ ] Normalize TA features to match training
- [ ] Include TA features in observation

#### Task 4: Update Pipeline (PENDING)
- [ ] Update `src/pipeline/training.py` to use `crypto_deep_v2.parquet`
- [ ] Update `src/pipeline/runner.py` to pass correct data path

#### Task 5: Testing (PENDING)
- [ ] Test training with TA features
- [ ] Test inference with TA features
- [ ] Compare performance: Old vs. New

### Technical Details

#### Feature Count
- **Before**: 4 features per asset (returns, volatility, commit_count, utility_score)
- **After**: 9 features per asset (+ RSI, MACD, BBW, ATR, SMA_RATIO)

#### Observation Space
- **Before**: `(n_assets * 4) + 2 = 4n + 2`
- **After**: `(n_assets * 9) + 2 = 9n + 2`

#### Normalization
- RSI: 0-100 → 0-1 (divide by 100)
- MACD: Min-max per asset
- BBW: Min-max per asset
- ATR: Already normalized (0-1)
- SMA_RATIO: Already a ratio (clip 0-2)

### Files Modified
1. `src/strategy/rl/data_loader.py` ✅
2. `src/strategy/rl/fast_env.py` (IN PROGRESS)
3. `src/strategy/ppo_strategy.py` (PENDING)
4. `src/pipeline/training.py` (PENDING)

### Notes
- Using manual TA implementation (not pandas-ta) for Python 3.11 compatibility
- Backward compatible: Falls back to zeros if TA features not present
- All TA calculations done per-ticker to prevent data leakage

