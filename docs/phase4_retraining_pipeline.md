# Phase 4: Automated Retraining Pipeline - Implementation Guide

## Overview

The Automated Retraining Pipeline implements the **Challenger Protocol** - a safe, automated MLOps system that continuously adapts the PPO model to changing market conditions (Concept Drift) without human intervention.

## Problem Solved

**The "Model Rot" Problem:**
- Models trained on historical data become stale over time
- Market conditions change (bull → bear → crab markets)
- A model that worked in 2024 may fail in 2025
- Manual retraining is time-consuming and error-prone

**The Solution:**
- **Automated Weekly Retraining**: Runs every Sunday via cron job
- **Challenger Protocol**: New model must beat current model to deploy
- **Transfer Learning**: Fine-tune from production model (faster convergence)
- **Safe Deployment**: Archive old model before promoting new one

## Architecture: The Challenger Protocol

```
┌─────────────────────────────────────────────────────────────┐
│              Automated Retraining Pipeline                   │
└─────────────────────────────────────────────────────────────┘

1. Data Refresh
   │
   ├─> Fetch latest market data (last 3 years)
   └─> Update crypto_training_data.parquet

2. Train Challenger
   │
   ├─> Load Champion (Production Model)
   ├─> Transfer Learning: Continue training from Champion weights
   ├─> Fine-tune: 500K timesteps (vs 10M for initial training)
   └─> Save to models/challenger/

3. The Arena (Evaluation)
   │
   ├─> Split data: Last 10% for out-of-sample evaluation
   ├─> Run Champion: Calculate Sortino Ratio
   ├─> Run Challenger: Calculate Sortino Ratio
   └─> Compare: Challenger vs Champion

4. The Verdict (Deployment)
   │
   ├─> If Challenger > Champion:
   │   ├─> Archive Champion to models/archive/
   │   ├─> Promote Challenger to production
   │   └─> Update vec_normalize.pkl
   │
   └─> Else:
       └─> Keep Champion, log "No improvement"
```

## Implementation Details

### Script: `scripts/11_retrain_model.py`

**Main Components:**

1. **Data Refresh (`refresh_training_data()`)**
   - Verifies training data exists
   - Checks data freshness
   - TODO: In production, calls data fetcher to get latest data

2. **Train Challenger (`train_challenger()`)**
   - Loads production model (Champion)
   - Creates training environment with updated data
   - Uses Transfer Learning (continues from Champion weights)
   - Fine-tunes for 500K timesteps (vs 10M initial training)
   - Saves to `models/challenger/challenger.zip`

3. **The Arena (`evaluate_models()`)**
   - Splits data: Last 10% for evaluation (out-of-sample)
   - Loads both Champion and Challenger models
   - Runs each model for 10 episodes
   - Calculates Sortino Ratio for each
   - Returns scores for comparison

4. **Deployment (`deploy_challenger()`)**
   - Archives current Champion with timestamp
   - Moves Challenger to production path
   - Updates normalization statistics

### Key Features

**Transfer Learning:**
- Starts from production model weights (not random initialization)
- Converges 20x faster (500K vs 10M timesteps)
- Preserves learned patterns while adapting to new data

**Evaluation Metric: Sortino Ratio**
- Risk-adjusted return focusing on downside protection
- Better than Sharpe Ratio for asymmetric returns (crypto)
- Formula: `(Mean Return - Risk Free Rate) / Downside Deviation`

**Safe Deployment:**
- Never overwrites production model without backup
- Archives old models with timestamps
- Can rollback if needed

## Usage

### Manual Execution

```bash
cd /root/projects/cinnamon-q
python scripts/11_retrain_model.py
```

### Automated Execution (Cron Job)

Add to crontab (`crontab -e`):

```bash
# Run every Sunday at 2 AM
0 2 * * 0 cd /root/projects/cinnamon-q && python scripts/11_retrain_model.py >> logs/retrain.log 2>&1
```

### Expected Output

```
================================================================================
🔄 Automated Retraining Pipeline - The Challenger Protocol
================================================================================

Started at: 2025-11-24 02:00:00

================================================================================
Step 1: Data Refresh
================================================================================

✓ Using existing training data: data/processed/crypto_training_data.parquet
  File size: 45.23 MB
  Latest data: 2025-11-23 23:00:00
  Data age: 1 days

================================================================================
Step 2: Training Challenger Model
================================================================================

📦 Loading Champion model: models/hdrl_optimized/final_model.zip
📊 Training data: data/processed/crypto_training_data.parquet

🔧 Creating training environment...
✓ Environment created: 8 parallel environments

🎓 Loading Champion model for Transfer Learning...
✓ Champion model loaded (Transfer Learning enabled)
  Fine-tuning learning rate: 0.0003

🚀 Fine-tuning Challenger model (500,000 timesteps)...
   This is Transfer Learning - starting from Champion weights

[Training progress...]

💾 Saving Challenger model...
✓ Challenger model saved: models/challenger/challenger.zip
✓ Normalization stats saved: models/challenger/vec_normalize.pkl

================================================================================
Step 3: The Arena - Model Evaluation
================================================================================

📊 Evaluation dataset: 3000 rows (last 10% of data)
   Date range: 2025-10-24 to 2025-11-23

✓ Using production normalization stats for evaluation

🛡️ Evaluating Champion (Production Model)...
   Champion Sortino Ratio: 2.3456

⚔️ Evaluating Challenger (New Model)...
   Challenger Sortino Ratio: 2.4567

================================================================================
Step 4: The Verdict
================================================================================

📊 Evaluation Results:
   Champion Sortino:  2.3456
   Challenger Sortino: 2.4567
   Improvement:        +0.1111 (+4.74%)

🏆 CHALLENGER WINS!
   Challenger outperforms Champion by 4.74%

================================================================================
Step 4: Deployment - Promoting Challenger
================================================================================

📦 Archiving current Champion...
   Archived to: models/archive/final_model_20251124_020000.zip

🚀 Deploying Challenger to production...
   ✓ Model deployed: models/hdrl_optimized/final_model.zip
   ✓ Normalization stats deployed: models/hdrl_optimized/vec_normalize.pkl

✅ Challenger successfully promoted to production!

================================================================================
✅ Retraining pipeline complete!
================================================================================
```

## Configuration

### Hyperparameters

Located at top of `scripts/11_retrain_model.py`:

```python
# Training hyperparameters (fine-tuning, not full training)
FINE_TUNE_TIMESTEPS = 500_000  # Shorter than initial training (10M)
FINE_TUNE_BATCH_SIZE = 2048
FINE_TUNE_LEARNING_RATE = 3e-4

# Evaluation parameters
EVAL_EPISODES = 10  # Number of episodes to evaluate each model
EVAL_DATA_SPLIT = 0.1  # Use last 10% of data for evaluation
```

### Paths

```python
PROD_MODEL_PATH = "models/hdrl_optimized/final_model.zip"
PROD_VECNORM_PATH = "models/hdrl_optimized/vec_normalize.pkl"
CHALLENGER_DIR = "models/challenger"
ARCHIVE_DIR = "models/archive"
DATA_PATH = "data/processed/crypto_training_data.parquet"
```

## Troubleshooting

### Issue: "Production model not found"
**Solution:** Ensure you've run initial training (`scripts/10_train_hdrl_optimized.py`) first.

### Issue: "Training data not found"
**Solution:** Run data fetcher (`scripts/08_fetch_crypto_deep.py`) to generate training data.

### Issue: "Challenger never wins"
**Solution:** This is normal! Not every retraining improves performance. The model will be retrained again next week. Consider:
- Increasing fine-tuning timesteps
- Adjusting learning rate
- Checking if market conditions have changed significantly

### Issue: "CUDA out of memory"
**Solution:** Reduce number of parallel environments in `train_challenger()`:
```python
train_env = SubprocVecEnv([make_env for _ in range(4)])  # Reduce from 8 to 4
```

### Issue: "Evaluation takes too long"
**Solution:** Reduce `EVAL_EPISODES` or `EVAL_DATA_SPLIT`:
```python
EVAL_EPISODES = 5  # Reduce from 10
EVAL_DATA_SPLIT = 0.05  # Reduce from 0.1 (last 5% instead of 10%)
```

## Best Practices

1. **Monitor Logs**: Check `logs/retrain.log` weekly to ensure pipeline runs successfully
2. **Archive Management**: Periodically clean old archives (keep last 10 models)
3. **Data Freshness**: Ensure data fetcher runs before retraining (or integrate into pipeline)
4. **Performance Tracking**: Track Sortino Ratio trends over time
5. **Rollback Plan**: Keep last 3 production models in archive for emergency rollback

## Future Enhancements

1. **Data Integration**: Automatically call data fetcher before retraining
2. **Rolling Window**: Implement true 3-year rolling window (forget old data)
3. **Multi-Metric Evaluation**: Evaluate on multiple metrics (Sharpe, Max Drawdown, etc.)
4. **A/B Testing**: Run both models in paper trading for 1 week before promotion
5. **Alerting**: Send notifications when Challenger wins or pipeline fails

## Conclusion

The Automated Retraining Pipeline transforms the bot from a "static model" system to a "continuously learning" system. The Challenger Protocol ensures safe, automated model updates that adapt to changing market conditions without human intervention.

This is the final piece of Phase 4, completing the Adaptive Intelligence Upgrade. The bot now:
- ✅ Sees current market data (Live Data Bridge)
- ✅ Continuously learns and adapts (Retraining Pipeline)

The system is now truly "self-correcting" and ready for long-term autonomous operation.

