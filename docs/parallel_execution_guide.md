# Parallel Execution Plan: Deploy Champion + Train Challenger

## 🎯 Strategy

**SRE Principle**: Availability over Perfection  
- **Track 1**: Deploy current LSTM (Champion) - Get bot live NOW
- **Track 2**: Train Super Brain (Challenger) - Build the upgrade in parallel

---

## ✅ Track 1: Deploy the Current LSTM (COMPLETE)

### Status: ✅ READY FOR DEPLOYMENT

**Actions Completed**:
1. ✅ MLP model backed up to `models/archive/`
2. ✅ LSTM model promoted to `models/hdrl_optimized/`
3. ✅ Config verified: Points to correct paths, 15m interval

### Launch Command

```bash
# In Terminal 1 (Production)
python main.py
```

**Expected Output**:
```
✓ Bot Orchestrator initialized
✓ Strategy: crypto_hdrl_v1 (PPO, 15m interval)
✓ Trading Engine ready
Starting main loop...
```

**Result**: Bot is live, trading with LSTM (Memory-enabled, 70 features)

---

## 🧪 Track 2: Train the Super Brain (READY TO START)

### Status: ✅ CONFIGURED

**Actions Completed**:
1. ✅ Training script updated to use `hdrl_super_brain` directory (won't overwrite production)
2. ✅ v2 dataset verified: All TA features present
3. ✅ Auto-detection logic: Will use v2 if available

### Launch Command

```bash
# In Terminal 2 (Training)
python scripts/10_train_hdrl_optimized.py
```

### What to Watch For

#### ✅ Success Indicators (Super Brain Mode):

```
✓ Using crypto_deep_v2.parquet (with TA features) - Super Brain mode!
  🧠 Super Brain mode: Training with TA features!
  ✓ Technical Analysis features detected (RSI, MACD, BBW, ATR, SMA Ratio)
  Total observation space: 155 features
  Observation space: Box(155,)
```

#### ❌ Wrong Mode (Regular LSTM):

```
⚠ Using crypto_training_data.parquet (no TA features)
  ⚠ Regular LSTM mode: No TA features available
  ⚠ Technical Analysis features not found
  Total observation space: 70 features
```

### Performance Expectations

**Current LSTM** (70 features):
- `ep_rew_mean`: ~3,920 (baseline)
- Training time: ~6 hours

**Super Brain** (155 features):
- `ep_rew_mean`: **Expected 5,000-10,000+** (much higher due to TA "glasses")
- Training time: ~6 hours (similar, but with 2.2x more features)
- **Why higher reward**: Can see RSI oversold, MACD crossovers, Bollinger squeezes

### Monitoring

**TensorBoard** (in another terminal):
```bash
tensorboard --logdir logs/hdrl_training_super_brain/tensorboard
```

**Key Metrics to Watch**:
- `rollout/ep_rew_mean`: Should climb much faster than baseline
- `train/loss`: Should decrease smoothly
- `train/explained_variance`: Should be > 0.9

---

## 🏆 The Handover (When Training Finishes)

### Step 1: Compare Models

```bash
python scripts/14_compare_models.py
```

**Expected Tournament**:
- **Champion** (Current LSTM): Sortino 0.026, Return +8.44%
- **Challenger** (Super Brain): Sortino **0.5-1.0+**, Return **+15-25%+**

### Step 2: Deploy Super Brain (If It Wins)

```bash
# Backup current production model
cp models/hdrl_optimized/final_model.zip models/archive/lstm_70feat_$(date +%Y%m%d).zip
cp models/hdrl_optimized/vec_normalize.pkl models/archive/lstm_70feat_$(date +%Y%m%d).pkl

# Promote Super Brain
cp models/hdrl_super_brain/final_model.zip models/hdrl_optimized/final_model.zip
cp models/hdrl_super_brain/vec_normalize.pkl models/hdrl_optimized/vec_normalize.pkl

# Restart bot (will auto-reload on next heartbeat)
# Or manually restart: python main.py
```

### Step 3: Verify Deployment

Check bot logs for:
```
✓ Strategy: crypto_hdrl_v1 (PPO, 15m interval)
✓ Model loaded: models/hdrl_optimized/final_model.zip
✓ Observation space: 155 features (Super Brain active!)
```

---

## 📊 Expected Improvements

| Metric | Current LSTM (70) | Super Brain (155) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Sortino Ratio** | 0.026 | 0.5-1.0+ | **10-30x** |
| **Total Return** | +8.44% | +15-25%+ | **2-3x** |
| **Max Drawdown** | -19.15% | -10-15% | **Better** |
| **Features** | 70 (base) | 155 (base + TA) | **2.2x** |

**Why Super Brain Wins**:
- **RSI**: Sees oversold/overbought conditions
- **MACD**: Detects trend changes early
- **Bollinger Bands**: Identifies volatility regimes
- **ATR**: Understands risk per trade
- **SMA Ratio**: Sees momentum vs. long-term trend

---

## 🚨 Troubleshooting

### Issue: Training shows 70 features instead of 155

**Check**:
```bash
ls -lh data/processed/crypto_deep_v2.parquet
python -c "import pandas as pd; df = pd.read_parquet('data/processed/crypto_deep_v2.parquet'); print('TA cols:', [c for c in df.columns if 'RSI' in c or 'MACD' in c])"
```

**Fix**: If v2 is missing, run:
```bash
python scripts/12_enrich_data.py
```

### Issue: Training uses wrong directory

**Check**: Look for this in training output:
```
🧠 Super Brain mode: Training with TA features!
```

**Fix**: The script auto-detects v2 and uses `hdrl_super_brain` directory.

### Issue: Bot doesn't reload new model

**Fix**: Restart manually:
```bash
# Stop bot (Ctrl+C)
python main.py
```

---

## ✅ Checklist

### Track 1 (Deployment):
- [x] MLP backed up
- [x] LSTM promoted to production
- [x] Config verified
- [ ] Bot launched (`python main.py`)
- [ ] Bot trading successfully

### Track 2 (Training):
- [x] Training script configured
- [x] v2 dataset verified
- [ ] Training started (`python scripts/10_train_hdrl_optimized.py`)
- [ ] Confirmed 155 features in output
- [ ] Training completes (~6 hours)
- [ ] Comparison run (`scripts/14_compare_models.py`)
- [ ] Super Brain deployed (if winner)

---

**You're ready to execute both tracks in parallel!** 🚀

