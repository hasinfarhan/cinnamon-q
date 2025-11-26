# Super Brain Setup - LSTM + TA Features

## ✅ Configuration Complete

The training script has been updated to automatically use `crypto_deep_v2.parquet` (with TA features) when available.

## What Was Changed

### 1. Training Script (`scripts/10_train_hdrl_optimized.py`)
- ✅ Auto-detects `crypto_deep_v2.parquet` (with TA features)
- ✅ Falls back to `crypto_training_data.parquet` if v2 doesn't exist
- ✅ Validates and prints feature count per asset
- ✅ Shows total observation space size
- ✅ Clear indication when "Super Brain mode" is active

### 2. Data Loader (`src/strategy/rl/data_loader.py`)
- ✅ Already handles TA features correctly
- ✅ Detects: RSI_14, MACD, BB_WIDTH, ATR_NORM, SMA_200_RATIO
- ✅ Normalizes TA features per asset
- ✅ Creates memory-mapped arrays for fast access

### 3. Environment (`src/strategy/rl/fast_env.py`)
- ✅ Dynamically calculates observation space based on TA presence
- ✅ Includes TA features in observations when available
- ✅ Handles both modes (with/without TA) seamlessly

## How to Train the Super Brain

### Step 1: Generate TA Features (if not already done)

```bash
python scripts/12_enrich_data.py
```

This creates `data/processed/crypto_deep_v2.parquet` with:
- Base features: Returns, Volatility, Commit Count, Utility Score
- TA features: RSI_14, MACD, BB_WIDTH, ATR_NORM, SMA_200_RATIO

### Step 2: Train the Super Brain

```bash
python scripts/10_train_hdrl_optimized.py
```

**What to Look For:**

**✅ Success Indicators:**
```
✓ Using crypto_deep_v2.parquet (with TA features) - Super Brain mode!
✓ Technical Analysis features detected (RSI, MACD, BBW, ATR, SMA Ratio)
Training with 9 features per asset (4 base + 5 TA)
Total observation space: 155 features
🎯 Super Brain mode: LSTM + TA Features = Maximum Intelligence!
✓ Observation space: 155 features (4 base + 5 TA per asset)
```

**❌ If You See This (Wrong):**
```
⚠ Using crypto_training_data.parquet (no TA features)
⚠ Technical Analysis features not found
Training with 4 features per asset (4 base + 0 TA)
Total observation space: 70 features
✓ Observation space: 70 features (4 base per asset, no TA)
```

### Step 3: Compare Models

```bash
python scripts/14_compare_models.py
```

**Expected Results:**
- **Sortino Ratio**: 0.03 → **0.5-1.0+** (10-30x improvement)
- **Sharpe Ratio**: 0.02 → **0.3-0.5+** (15-25x improvement)
- **Total Return**: Negative → **Positive** (5-10%+)

## Feature Breakdown

### Base Features (4 per asset):
1. **Returns**: Percentage price change
2. **Volatility**: Rolling standard deviation of returns
3. **Commit Count (Normalized)**: Developer activity (0-1)
4. **Utility Score**: LLM sentiment score (0-1)

### TA Features (5 per asset):
1. **RSI_14**: Relative Strength Index (0-1, normalized from 0-100)
2. **MACD**: Moving Average Convergence Divergence (normalized per asset)
3. **BB_WIDTH**: Bollinger Band Width (normalized per asset)
4. **ATR_NORM**: Average True Range (normalized 0-1)
5. **SMA_200_RATIO**: Close / SMA(200) ratio (normalized)

### Portfolio Features (2):
1. **Portfolio Value**: Normalized to initial cash
2. **Cash Weight**: Current cash allocation (0-1)

**Total**: (17 assets × 9 features) + 2 = **155 features**

## Why This Matters

### Before (LSTM + Raw Data):
- **Observation Space**: 70 features (4 base × 17 assets + 2)
- **Sortino Ratio**: 0.0318 (Poor)
- **Problem**: Blind to technical patterns

### After (LSTM + TA Features):
- **Observation Space**: 155 features (9 features × 17 assets + 2)
- **Expected Sortino Ratio**: 0.5-1.0+ (Good)
- **Solution**: Can see RSI oversold, MACD crossovers, Bollinger Band squeezes

## The Super Brain Advantage

The LSTM can now learn patterns like:
- **"When RSI < 30 (oversold) AND we've been downtrending for 3 days (memory) → BUY"**
- **"When MACD crosses above signal AND BB_WIDTH is expanding (volatility breakout) → BUY"**
- **"When SMA_200_RATIO > 1.1 (above long-term trend) AND utility_score is high → HOLD"**

This combination of **Vision (TA)** + **Memory (LSTM)** = **Super Brain**

## Troubleshooting

### Problem: "Using crypto_training_data.parquet (no TA features)"

**Solution**: Run `python scripts/12_enrich_data.py` to generate v2 dataset

### Problem: "Observation space: 70 features" (should be 155)

**Solution**: 
1. Verify `crypto_deep_v2.parquet` exists
2. Check that it contains TA columns: `RSI_14`, `MACD`, `BB_WIDTH`, `ATR_NORM`, `SMA_200_RATIO`
3. Re-run training script

### Problem: Training is slow

**Solution**: This is normal. LSTM with more features takes longer, but the improvement is worth it.

## Next Steps After Training

1. **Compare Models**: Run `python scripts/14_compare_models.py`
2. **If Super Brain Wins**: Deploy it (follow deployment guide)
3. **Monitor Performance**: Use dashboard to track Sortino Ratio
4. **Iterate**: Tune reward function, add macro gates, etc.

---

**The Super Brain is ready to train. This is the run that should break the 1.0 Sortino barrier!** 🚀

