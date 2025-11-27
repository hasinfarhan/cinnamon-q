# Phase 5.5 Sprint 2: The Correlation Eye - COMPLETE ✅

## 🎯 Goal
Add rolling correlation to BTC as a feature so the AI can identify hedging assets (like PAXG) that decouple from the broad market.

---

## ✅ Completed Tasks

### Sprint 2.1: Correlation Feature Engineering ✅
**File**: `src/features/indicators.py`

**Implementation**:
- Added `add_correlation()` method to `TechnicalAnalysis` class
- Calculates 30-day (96 period) rolling correlation of each asset to BTC
- Logic:
  1. Pivot DataFrame to wide format (timestamp × ticker)
  2. Calculate percentage returns
  3. Calculate rolling correlation for each asset vs. benchmark
  4. Melt back to long format
  5. Merge into original DataFrame
- Normalizes correlation [-1, 1] → [0, 1] (corr_btc_norm)
- Defaults to 0.5 (neutral correlation) if missing

**Key Features**:
- Handles benchmark ticker variants (BTC, BTC/USD, etc.)
- Efficient vectorized operations
- Robust error handling

---

### Sprint 2.2: Enrichment Script Update ✅
**File**: `scripts/12_enrich_data.py`

**Changes**:
- Added call to `ta.add_correlation()` after TA features
- Correlation calculated for all assets in universe
- Integrated into existing enrichment pipeline

---

### Sprint 2.3: Training Pipeline Integration ✅
**Files**: `src/strategy/rl/data_loader.py`, `src/strategy/rl/fast_env.py`, `scripts/10_train_hdrl_optimized.py`

**Changes**:
1. **Data Loader**:
   - Detects `corr_btc` column
   - Normalizes to `corr_btc_norm` [0, 1]
   - Pivots correlation to wide format
   - Saves as `corr_array.npy` (memory-mapped)
   - Includes `has_correlation` flag in config

2. **Environment**:
   - Loads `corr_array` from memory-mapped file
   - Updates observation space: **5 base features** (was 4)
   - Includes `corr_btc_norm` in observation vector
   - Handles legacy mode (backward compatibility)

3. **Training Script**:
   - Updated feature count calculation
   - Displays correlation status in logs

---

## 📊 Feature Count Changes

### Before Sprint 2:
- **Base features**: 4 (returns, volatility, commit_count, utility_score)
- **TA features**: 5 (if present)
- **Total (with TA)**: (17 assets × 9 features) + 2 = **155 features**

### After Sprint 2:
- **Base features**: 5 (returns, volatility, commit_count, utility_score, **corr_btc_norm**)
- **TA features**: 5 (if present)
- **Total (with TA)**: (17 assets × 10 features) + 2 = **172 features**
- **Total (without TA)**: (17 assets × 5 features) + 2 = **87 features**

---

## 🧠 Expected AI Behavior

### Hypothesis:
- **High Correlation (+0.9)**: High Beta. Good for Bull Runs.
- **Low/Negative Correlation (< 0.0)**: Hedge. Good for Bear Markets.

### The Brain's Lesson:
"When Volatility is High AND Market Trend is Down → Shift weight to assets with Low Correlation (PAXG)."

### Expected Outcomes:
1. **Bear Market**: Agent learns to increase PAXG allocation (low correlation)
2. **Bull Market**: Agent learns to increase BTC/ETH allocation (high correlation)
3. **Risk Management**: Agent uses correlation as a hedge signal

---

## 🚀 Next Steps

### Immediate:
1. **Re-Enrich Data**:
   ```bash
   python scripts/12_enrich_data.py
   ```
   - Will add `corr_btc` column to `crypto_deep_v2.parquet`
   - Verify: Check that `corr_btc` column exists

2. **Re-Train Model**:
   ```bash
   python scripts/10_train_hdrl_optimized.py
   ```
   - Look for: "Observation space: 172 features (5 base + 5 TA per asset)"
   - Look for: "🛡️ Fortress Brain mode: Correlation Eye enabled"

### Expected Training Behavior:
- **Observation Space**: Should show 172 features (with TA) or 87 (without TA)
- **Reward**: May drop initially (fees hurt), but agent learns correlation patterns
- **Allocation**: Should see more PAXG during simulated bear markets

### Sprint 3 (Next):
- **Regime Benchmark Suite**: Test agent in Bull, Bear, and Chop markets
- **Update `scripts/14_compare_models.py`** with regime-specific evaluation

---

## ✅ Acceptance Criteria Met

- [x] Correlation feature added to `indicators.py`
- [x] Enrichment script calls `add_correlation()`
- [x] Data loader processes `corr_btc_norm`
- [x] Environment includes correlation in observations
- [x] Observation space updated (5 base features)
- [x] Training script reflects new feature count
- [x] Backward compatibility maintained

---

## 📝 Key Files Modified

1. `src/features/indicators.py` - Added `add_correlation()` method
2. `scripts/12_enrich_data.py` - Calls correlation calculation
3. `src/strategy/rl/data_loader.py` - Processes correlation feature
4. `src/strategy/rl/fast_env.py` - Includes correlation in observations
5. `scripts/10_train_hdrl_optimized.py` - Updated feature count

---

## 🔍 Verification

**Test Correlation Calculation**:
```python
from src.features.indicators import TechnicalAnalysis
ta = TechnicalAnalysis()
df_corr = ta.add_correlation(df, benchmark_ticker='BTC', window=96)
# Should have 'corr_btc' column with values in [-1, 1]
```

**Check Enriched Data**:
```python
import pandas as pd
df = pd.read_parquet('data/processed/crypto_deep_v2.parquet')
print('Has corr_btc:', 'corr_btc' in df.columns)
if 'corr_btc' in df.columns:
    print('Correlation range:', df['corr_btc'].min(), 'to', df['corr_btc'].max())
```

---

**Sprint 2 Complete! Ready for Sprint 3: The Regime Benchmark Suite** 🎯

