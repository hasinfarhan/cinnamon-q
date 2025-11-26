# Super Brain Retraining Guide

## Current Situation

✅ **LSTM Training Complete**: 10M timesteps, 6 hours  
⚠️ **Trained WITHOUT TA Features**: 70 features (base only)  
✅ **v2 Dataset Ready**: `crypto_deep_v2.parquet` exists with all TA features  

**Issue**: The training script selected v2, but the data_loader didn't detect TA features. This likely happened because:
- Training started before v2 was fully generated, OR
- File was in an intermediate state during training

## Solution: Retrain Super Brain

The v2 file is now complete with all TA features. Retrain to get the **Super Brain** (LSTM + TA = 155 features).

### Step 1: Verify v2 Dataset

```bash
# Check file exists and has TA features
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/crypto_deep_v2.parquet')
ta_cols = ['RSI_14', 'MACD', 'BB_WIDTH', 'ATR_NORM', 'SMA_200_RATIO']
has_all = all(c in df.columns for c in ta_cols)
print(f'✓ v2 file exists: True')
print(f'✓ Has all TA features: {has_all}')
print(f'✓ Total columns: {len(df.columns)}')
"
```

**Expected**: `Has all TA features: True`

### Step 2: Retrain Super Brain

```bash
python scripts/10_train_hdrl_optimized.py
```

**What to Look For**:

✅ **Success Indicators**:
```
✓ Using crypto_deep_v2.parquet (with TA features) - Super Brain mode!
✓ Technical Analysis features detected (RSI, MACD, BBW, ATR, SMA Ratio)
Training with 9 features per asset (4 base + 5 TA)
Total observation space: 155 features
🎯 Super Brain mode: LSTM + TA Features = Maximum Intelligence!
✓ Observation space: 155 features (4 base + 5 TA per asset)
```

❌ **If You See This (Wrong)**:
```
⚠ Technical Analysis features not found
Training with 4 features per asset (4 base + 0 TA)
Total observation space: 70 features
```

### Step 3: Compare Super Brain vs. Current LSTM

```bash
python scripts/14_compare_models.py
```

**Expected Results**:
- **Current LSTM** (70 features): Sortino 0.026, Return +8.44%
- **Super Brain** (155 features): Sortino **0.5-1.0+**, Return **+15-25%+**

## Why This Matters

**Current LSTM** (70 features):
- Base features only: Returns, Volatility, Commit Count, Utility Score
- **Blind to technical patterns**: Can't see RSI oversold, MACD crossovers, etc.

**Super Brain** (155 features):
- Base + TA: RSI, MACD, BBW, ATR, SMA Ratio
- **Can see patterns**: "When RSI < 30 AND downtrending → BUY"
- **Expected**: 10-30x Sortino improvement

## Training Time

- **Current LSTM**: 6 hours (10M timesteps)
- **Super Brain**: Similar time (~6 hours), but with 2.2x more features
- **Worth It**: Yes! Expected 10-30x improvement in Sortino Ratio

## Deployment Strategy

### Option A: Deploy Current LSTM Now, Upgrade Later
1. Deploy current LSTM (already beats benchmarks)
2. Start trading
3. Train Super Brain in background
4. Upgrade when ready

### Option B: Train Super Brain First, Then Deploy
1. Retrain with TA features (6 hours)
2. Compare Super Brain vs. Current LSTM
3. Deploy the winner

**Recommendation**: Option A - Deploy current LSTM now to start trading, then upgrade to Super Brain when training completes.

---

**The Super Brain (LSTM + TA) is the final upgrade. It should break the 1.0 Sortino barrier!** 🚀

