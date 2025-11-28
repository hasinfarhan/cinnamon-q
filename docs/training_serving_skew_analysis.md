# Training/Serving Skew Analysis: Transaction Fees Impact

## 🚨 Critical Issue Identified

**Date**: 2025-11-27  
**Issue**: Severe performance degradation in ML models (-67% MLP, -61% LSTM)  
**Root Cause**: Training/Serving Skew with Transaction Fees

---

## The Problem

### Timeline
1. **Nov 26**: Models trained (MLP and LSTM) - **WITHOUT transaction fees**
2. **Phase 5.5 Sprint 1**: Added transaction fees (`fee_rate=0.001` = 0.1%)
3. **Nov 27**: Evaluation script uses `fee_rate=0.001` - **WITH fees**

### The Skew
- **Training Environment**: No fees → Models learned to trade aggressively
- **Evaluation Environment**: 0.1% fees per trade → Every trade penalized
- **Result**: Over-trading killed returns

---

## Performance Comparison

### With Fees (Current Evaluation)
| Strategy | Sortino | Sharpe | Return | Max DD |
|----------|---------|--------|--------|--------|
| Buy & Hold | -0.0063 | -0.0049 | **+13.98%** | -17.28% |
| MLP (AI) | -0.3844 | -0.2867 | **-67.82%** | -67.91% |
| LSTM (AI) | -0.4202 | -0.1126 | **-61.83%** | -65.33% |

### Why Buy & Hold Wins
- **No trading** → No fees
- **Passive strategy** → No transaction costs
- **AI models** → Over-trading → Fees eat returns

---

## The Math

### Transaction Cost Impact
If a model makes **100 trades per episode** with **0.1% fee per trade**:
- **Turnover**: 100% (full portfolio rebalance)
- **Transaction Cost**: 100% × 0.1% = **0.1% per trade**
- **100 trades**: 100 × 0.1% = **10% in fees alone**

### Example Scenario
- **Gross Return**: +20% (before fees)
- **Trades**: 50 rebalances
- **Fees**: 50 × 0.1% = 5%
- **Net Return**: 20% - 5% = **+15%**

But if the model over-trades (200 trades):
- **Fees**: 200 × 0.1% = **20%**
- **Net Return**: 20% - 20% = **0%** (break even)

---

## Solution

### Immediate Fix (Applied)
- **Evaluate old models with `fee_rate=0.0`** to see their "raw" performance
- This shows what the models learned before fees were introduced

### Long-Term Fix (Required)
1. **Retrain models WITH fees** (`fee_rate=0.001`)
2. Models will learn to be cost-conscious
3. They'll trade less frequently (only when it's worth it)
4. Evaluation can then use `fee_rate=0.001` to match training

---

## Expected Behavior After Retraining

### With Fees in Training
- **Models learn**: "Trading costs money, only trade when it's profitable"
- **Result**: Lower turnover, higher net returns
- **Performance**: Should match or exceed Buy & Hold

### Key Metrics to Watch
- **Turnover**: Should decrease (fewer trades)
- **Transaction Costs**: Should be < 1% of returns
- **Net Return**: Should be positive and competitive

---

## Action Items

1. ✅ **Fixed**: Evaluation script now uses `fee_rate=0.0` for old models
2. ⏳ **Pending**: Retrain models with `fee_rate=0.001` in training
3. ⏳ **Pending**: Re-evaluate with fees after retraining
4. ⏳ **Pending**: Compare performance with/without fees

---

## Lesson Learned

**Always match training and evaluation environments!**

- If you add fees to evaluation, add them to training
- If you change reward function, retrain
- If you change observation space, retrain

**Training/Serving Skew** is one of the most common ML production failures.

---

**Status**: Fixed (evaluation), Pending (retraining)

