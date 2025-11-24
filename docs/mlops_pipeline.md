# Autonomous MLOps Pipeline - Complete Guide

## Overview

The Autonomous MLOps Pipeline (Phase 4.5) consolidates all training-related scripts into a unified, automated system that runs in the background while the bot continues trading.

**Key Principle**: One Script Rule - Just run `python main.py`, everything else is automatic.

---

## Architecture

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│              MLOps Pipeline (Background Process)            │
└─────────────────────────────────────────────────────────────┘

Stage 1: Data Ingestion
  └─> Fetch raw market data (Alpaca + CoinGecko)
  └─> Output: crypto_deep_master.parquet

Stage 2: Feature Engineering
  └─> Add technical analysis indicators (RSI, MACD, etc.)
  └─> Output: crypto_deep_v2.parquet

Stage 3: Model Training
  └─> Train challenger model (PPO + GPU)
  └─> Output: models/challenger/final_model.zip

Stage 4: Model Evaluation
  └─> Compare challenger vs champion (Sortino Ratio)
  └─> Output: Evaluation scores

Stage 5: Model Promotion (if challenger wins)
  └─> Archive current production model
  └─> Promote challenger to production
  └─> Update latest_model.json
```

### Integration with Trading Bot

```
┌─────────────────────────────────────────────────────────────┐
│              Main Trading Loop (Foreground)                  │
│  - Executes strategies every 15 minutes                      │
│  - Continues trading uninterrupted                          │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ (Every 6 hours)
                    ▼
┌─────────────────────────────────────────────────────────────┐
│         Training Scheduler (Background Process)              │
│  - Spawns MLOps pipeline in separate process               │
│  - Non-blocking: Trading never stops                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

### config.yaml

```yaml
system:
  training_interval_hours: 6  # Run training every 6 hours (0 = disabled)
```

### Pipeline Parameters

The pipeline can be configured via `run_full_pipeline()` parameters:
- `total_timesteps`: Training timesteps (default: 10M)
- `skip_ingestion`: Skip data ingestion (use existing data)
- `skip_features`: Skip feature engineering (use existing data)
- `skip_training`: Skip training (use existing challenger)
- `skip_evaluation`: Skip evaluation (promote directly)

---

## Usage

### Automatic (Recommended)

The pipeline runs automatically when you start the bot:

```bash
python main.py
```

The bot will:
1. Start trading (executes strategies every 15 minutes)
2. Trigger background training every 6 hours
3. Continue trading while training runs

### Manual Execution

You can also run the pipeline manually:

```python
from pipeline.runner import run_full_pipeline

# Run full pipeline
success = run_full_pipeline()

# Or run with custom parameters
success = run_full_pipeline(
    total_timesteps=5_000_000,
    skip_ingestion=True,  # Use existing data
    skip_features=True    # Use existing features
)
```

### Individual Stages

You can also run individual stages:

```python
from pipeline.ingestion import run_ingestion
from pipeline.features import run_feature_engineering
from pipeline.training import run_training
from pipeline.evaluation import run_evaluation

# Stage 1: Data Ingestion
data_path = run_ingestion()

# Stage 2: Feature Engineering
features_path = run_feature_engineering(input_path=data_path)

# Stage 3: Training
model_path, vec_norm_path = run_training(
    data_path=features_path,
    model_output_dir=Path("models/challenger")
)

# Stage 4: Evaluation
champion_score, challenger_score = run_evaluation(...)
```

---

## Model Management

### Directory Structure

```
models/
├── hdrl_optimized/      # Original training location (legacy)
│   ├── final_model.zip
│   └── vec_normalize.pkl
├── challenger/           # Staging area for new models
│   ├── final_model.zip
│   └── vec_normalize.pkl
├── production/           # Production models (hot-swapped)
│   ├── final_model.zip
│   ├── vec_normalize.pkl
│   └── latest_model.json  # Metadata for hot-swapping
└── archive/              # Archived models (timestamped)
    └── model_YYYYMMDD_HHMMSS/
        ├── final_model.zip
        └── vec_normalize.pkl
```

### Hot-Swapping

When a challenger model outperforms the champion:
1. Current production model is archived (timestamped)
2. Challenger is promoted to `models/production/`
3. `latest_model.json` is updated with metadata
4. Bot can reload model on next strategy execution

---

## Logging

### Pipeline Logs

The pipeline logs to:
- **File**: `logs/mlops_pipeline.log`
- **Console**: Standard output (if running manually)

### Log Format

```
2024-01-15 10:00:00 - [MLOps] - INFO - Stage 1: Data Ingestion
2024-01-15 10:05:00 - [MLOps] - INFO - ✓ Ingestion complete: 100,000 rows
2024-01-15 10:10:00 - [MLOps] - INFO - Stage 2: Feature Engineering
...
```

### Trading Bot Logs

The main bot continues logging normally:
- Strategy execution logs
- Trade execution logs
- Training trigger notifications

---

## Monitoring

### Check Training Status

```bash
# Check if training process is running
ps aux | grep MLOpsPipeline

# Check training logs
tail -f logs/mlops_pipeline.log

# Check last training time
grep "Last training" logs/mlops_pipeline.log | tail -1
```

### Check Model Versions

```bash
# Check production model metadata
cat models/production/latest_model.json

# List archived models
ls -la models/archive/
```

---

## Troubleshooting

### Training Not Triggering

1. **Check configuration**: Ensure `training_interval_hours > 0` in `config.yaml`
2. **Check logs**: Look for errors in `logs/mlops_pipeline.log`
3. **Check process**: Verify no previous training process is stuck

### Training Fails

1. **Check data**: Ensure `crypto_deep_master.parquet` exists
2. **Check GPU**: Verify GPU is available (if using CUDA)
3. **Check disk space**: Training requires significant disk space
4. **Check logs**: Review `logs/mlops_pipeline.log` for errors

### Model Not Promoting

1. **Check evaluation**: Verify challenger actually outperformed champion
2. **Check permissions**: Ensure write access to `models/production/`
3. **Check logs**: Review evaluation stage logs

---

## Best Practices

### 1. Start with Manual Training

For the first run, execute the pipeline manually to verify everything works:

```bash
python -c "from src.pipeline.runner import run_full_pipeline; run_full_pipeline()"
```

### 2. Monitor First Few Cycles

Watch the logs for the first few training cycles to ensure:
- Data is being fetched correctly
- Features are being calculated correctly
- Training is completing successfully
- Models are being promoted correctly

### 3. Adjust Training Frequency

Based on your needs:
- **High-frequency markets**: 6 hours (default)
- **Stable markets**: 12-24 hours
- **Testing**: 1 hour (for rapid iteration)

### 4. Archive Management

Periodically clean up old archives:

```bash
# Keep only last 10 archives
ls -t models/archive/ | tail -n +11 | xargs rm -rf
```

---

## Performance Considerations

### Training Time

- **Full training** (10M timesteps): ~1-2 hours on RTX 4070 Ti Super
- **Fine-tuning** (300K timesteps): ~15-30 minutes
- **Data ingestion**: ~5-10 minutes (depends on API rate limits)
- **Feature engineering**: ~2-5 minutes

### Resource Usage

- **CPU**: Training uses all cores (16 parallel environments)
- **GPU**: Training saturates GPU (large batch sizes)
- **RAM**: ~4-8 GB during training
- **Disk**: ~500 MB per model version

### Non-Blocking Design

The pipeline runs in a separate process, so:
- Trading continues every 15 minutes
- No performance impact on strategy execution
- Training can use full system resources

---

## Future Enhancements

### Planned Improvements

1. **Incremental Training**: Only retrain on new data (not full dataset)
2. **A/B Testing**: Run multiple challengers simultaneously
3. **Model Ensembles**: Combine multiple models for better performance
4. **Cloud Integration**: Offload training to cloud GPU instances
5. **Monitoring Dashboard**: Real-time visualization of training progress

---

## Summary

The Autonomous MLOps Pipeline transforms the bot from a "manual training" system to a "set and forget" autonomous system. You no longer need to:
- Remember which scripts to run
- Manually trigger training
- Monitor training progress
- Manually promote models

Just run `python main.py` and the bot handles everything automatically.

**Status**: ✅ **Production-Ready**  
**Next**: Monitor first few training cycles and adjust intervals as needed

