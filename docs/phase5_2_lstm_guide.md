# Phase 5.2: LSTM Upgrade - Step-by-Step Guide

## Overview

This guide walks you through training the LSTM model and comparing it with the existing MLP model to see the improvements.

## Prerequisites

✅ **Dependencies Installed**: `sb3-contrib` is already installed  
✅ **Training Data**: `data/processed/crypto_training_data.parquet` should exist  
✅ **Existing MLP Model**: `models/hdrl_optimized/final_model.zip` (for comparison)

## Step-by-Step Process

### Step 1: Verify Prerequisites

```bash
# Check if training data exists
ls -lh data/processed/crypto_training_data.parquet

# Check if MLP model exists (for comparison)
ls -lh models/hdrl_optimized/final_model.zip
```

If training data is missing, run:
```bash
python scripts/08_fetch_crypto_deep.py
python scripts/09_generate_sentiment.py
```

### Step 2: Train the LSTM Model

This will train a RecurrentPPO model with LSTM memory. Training time depends on your hardware:
- **RTX 4070 Ti Super**: ~1-2 hours for 10M timesteps
- **CPU only**: ~4-6 hours

```bash
python scripts/10_train_hdrl_optimized.py
```

**What to expect:**
- The script will detect your hardware (GPU/CPU)
- Load and preprocess data (memory-mapped for speed)
- Create 16 parallel environments
- Train RecurrentPPO with LSTM policy
- Save model to `models/hdrl_lstm/final_model.zip`
- Save normalization stats to `models/hdrl_lstm/vec_normalize.pkl`

**Watch for:**
- Increasing `ep_rew_mean` (episode reward mean) - should trend upward
- Early stopping if performance plateaus (saves time)
- Final model saved in `models/hdrl_lstm/`

### Step 3: Compare Models (The "Champion vs. Champion" Battle)

Once training completes, compare the LSTM model against the MLP model:

```bash
python scripts/14_compare_models.py
```

**What this does:**
- Loads both MLP and LSTM models
- Runs each model on the same unseen data (last 10% of dataset)
- Evaluates 5 episodes per model
- Calculates metrics:
  - **Sortino Ratio** (primary): Risk-adjusted returns (downside risk)
  - **Sharpe Ratio**: Risk-adjusted returns (total risk)
  - **Total Return**: Absolute performance
  - **Max Drawdown**: Worst peak-to-trough decline

**Output:**
```
🏁 THE VERDICT
================================================================================
Metric               MLP (Champion)      LSTM (Challenger)   Winner
--------------------------------------------------------------------------------
Sortino Ratio        0.5234              0.6789              LSTM 🏆
Sharpe Ratio         0.4123              0.5123              LSTM 🏆
Total Return (%)     12.45               18.67               LSTM 🏆
Max Drawdown (%)     -8.23               -6.45               LSTM 🏆

✅ LSTM MODEL WINS! (Higher Sortino Ratio)
```

### Step 4: Deploy the Winner (If LSTM Wins)

If the LSTM model outperforms the MLP model, deploy it:

```bash
# 1. Create archive directory
mkdir -p models/archive

# 2. Backup current MLP model (with timestamp)
cp models/hdrl_optimized/final_model.zip \
   models/archive/mlp_model_$(date +%Y%m%d).zip
cp models/hdrl_optimized/vec_normalize.pkl \
   models/archive/mlp_vecnorm_$(date +%Y%m%d).pkl

# 3. Deploy LSTM model to production location
cp models/hdrl_lstm/final_model.zip \
   models/hdrl_optimized/final_model.zip
cp models/hdrl_lstm/vec_normalize.pkl \
   models/hdrl_optimized/vec_normalize.pkl

# 4. Restart the bot
python main.py
```

**Note**: The bot will automatically detect it's an LSTM model and use stateful inference.

### Step 5: Monitor Performance

After deployment, monitor the bot's performance:

```bash
# View dashboard
streamlit run dashboard/app.py

# Or check database directly
sqlite3 arynosq.db "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 10;"
```

## Understanding the Improvements

### What LSTM Adds

1. **Temporal Memory**: The model remembers market trends from previous 15-minute ticks
   - MLP: Only sees current candle (amnesia)
   - LSTM: Sees sequence of 128 candles (has memory)

2. **Pattern Recognition**: Can learn:
   - Momentum trends (following trends)
   - Mean reversion (buying dips)
   - Volatility clustering (risk management)

3. **Stateful Inference**: Maintains context across predictions
   - Each 15-minute tick builds on previous context
   - Better for trend-following strategies

### Expected Improvements

- **Sortino Ratio**: +10-30% improvement (better risk-adjusted returns)
- **Sharpe Ratio**: +5-20% improvement (better overall risk management)
- **Max Drawdown**: -10-30% reduction (better downside protection)
- **Total Return**: Variable (depends on market conditions)

## Troubleshooting

### Training Issues

**Problem**: "CUDA out of memory"
- **Solution**: Reduce `batch_size` in training script (line 202)
- Change `batch_size = 256` to `batch_size = 128`

**Problem**: Training is too slow
- **Solution**: Reduce `total_timesteps` (line 313)
- Change `total_timesteps = 10_000_000` to `total_timesteps = 5_000_000`

**Problem**: "ImportError: No module named 'sb3_contrib'"
- **Solution**: `pipenv install sb3-contrib`

### Comparison Issues

**Problem**: "Model not found" errors
- **Solution**: Ensure both models exist:
  - `models/hdrl_optimized/final_model.zip` (MLP)
  - `models/hdrl_lstm/final_model.zip` (LSTM)

**Problem**: Comparison script crashes
- **Solution**: Check that training data exists:
  - `data/processed/crypto_training_data.parquet`

### Deployment Issues

**Problem**: Bot crashes after deploying LSTM
- **Solution**: Check logs for LSTM state errors
- Ensure `PPOStrategy` correctly handles LSTM states (already implemented)

## Next Steps After Deployment

1. **Monitor for 1 week**: Track performance metrics
2. **Compare live performance**: LSTM vs MLP in production
3. **Fine-tune if needed**: Adjust LSTM hyperparameters if performance is suboptimal
4. **Consider Phase 5.1**: Integrate TA features for even better performance

## Summary

1. ✅ Install dependencies: `pipenv install sb3-contrib` (already done)
2. 🚀 Train LSTM: `python scripts/10_train_hdrl_optimized.py`
3. 🏆 Compare: `python scripts/14_compare_models.py`
4. 📦 Deploy: Follow Step 4 commands if LSTM wins
5. 📊 Monitor: Use dashboard or database queries

The LSTM upgrade gives your brain **memory** - it can now learn temporal patterns and make better decisions based on market trends!

