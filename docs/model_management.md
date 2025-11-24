# Model Management Best Practices

Comprehensive guide for managing ML model artifacts in a production trading system.

## Table of Contents

1. [Git Strategy](#git-strategy)
2. [Versioning Workflow](#versioning-workflow)
3. [Storage Solutions](#storage-solutions)
4. [Deployment Checklist](#deployment-checklist)
5. [Rollback Procedures](#rollback-procedures)

---

## Git Strategy

### What to Track in Git

✅ **DO Track:**
- Model metadata (hyperparameters, performance metrics)
- Training scripts and configuration
- Model architecture definitions
- Documentation (README.md, VERSIONS.md)
- Configuration files (config.yaml)

❌ **DON'T Track:**
- Model weights (.zip, .pkl, .pt, .pth files)
- Large data files (.parquet, .csv)
- Training logs and checkpoints
- Database files (.db, .sqlite3)

### Current .gitignore Rules

```gitignore
# Model files (too large for git)
*.zip
*.pkl
*.pt
*.pth
*.ckpt
*.onnx

# Model directories (keep structure, ignore contents)
models/**/*.zip
models/**/*.pkl
!models/.gitkeep
!models/**/.gitkeep
!models/**/*.md
```

### Directory Structure

```
models/
├── .gitkeep                    # Tracked (preserves directory)
├── README.md                   # Tracked (documentation)
└── hdrl_optimized/
    ├── .gitkeep               # Tracked (preserves directory)
    ├── VERSIONS.md            # Tracked (version history)
    ├── final_model.zip        # Ignored (binary file)
    ├── vec_normalize.pkl      # Ignored (binary file)
    └── final_model_v1.0.zip   # Ignored (versioned binary)
```

---

## Versioning Workflow

### Step-by-Step Process

#### 1. After Training

```bash
# Training completes, creates:
# - models/hdrl_optimized/final_model.zip
# - models/hdrl_optimized/vec_normalize.pkl
```

#### 2. Create Versioned Copy

```bash
# Determine version number
VERSION="v1.0"  # or v1.1, v2.0, etc.

# Create versioned copies
cp models/hdrl_optimized/final_model.zip \
   models/hdrl_optimized/final_model_${VERSION}.zip

cp models/hdrl_optimized/vec_normalize.pkl \
   models/hdrl_optimized/vec_normalize_${VERSION}.pkl
```

#### 3. Document Version

Edit `models/hdrl_optimized/VERSIONS.md`:

```markdown
## v1.0

- **Training Date**: 2025-01-XX
- **Training Timesteps**: 5,944,320
- **Hyperparameters**:
  - Learning Rate: 0.0003
  - Batch Size: 4096
  - Network: [512, 512, 512]
- **Performance**:
  - Episode Reward Mean: 2,620
  - Explained Variance: 0.962
- **Status**: ✅ Production
```

#### 4. Commit Metadata

```bash
git add models/hdrl_optimized/VERSIONS.md
git commit -m "docs: Add model v1.0 to version history"
```

**Note**: Model files themselves are NOT committed (they're in .gitignore)

---

## Storage Solutions

### Option 1: Local Versioning (Current Approach)

**Pros:**
- Simple, no external dependencies
- Fast access
- Works offline

**Cons:**
- Limited by local disk space
- No automatic backups
- Hard to share across machines

**Best For:**
- Development and testing
- Single-machine deployments
- Small teams

### Option 2: Cloud Storage (Recommended for Production)

**AWS S3:**
```bash
# Upload model
aws s3 cp models/hdrl_optimized/final_model_v1.0.zip \
   s3://your-bucket/models/hdrl_optimized/v1.0/

# Download model
aws s3 cp s3://your-bucket/models/hdrl_optimized/v1.0/final_model.zip \
   models/hdrl_optimized/final_model.zip
```

**Google Cloud Storage:**
```bash
gsutil cp models/hdrl_optimized/final_model_v1.0.zip \
   gs://your-bucket/models/hdrl_optimized/v1.0/
```

**Benefits:**
- Automatic backups
- Versioning support
- Access from multiple machines
- Cost-effective storage

### Option 3: Model Registry (Best for Production)

**MLflow:**
```python
import mlflow

# Log model
mlflow.sklearn.log_model(model, "model")
mlflow.log_param("learning_rate", 0.0003)
mlflow.log_metric("ep_rew_mean", 2620)

# Register model
mlflow.register_model("runs:/<run_id>/model", "hdrl_crypto")
```

**Weights & Biases:**
```python
import wandb

wandb.init(project="crypto-hdrl")
wandb.log_artifact("models/hdrl_optimized/final_model.zip")
```

**Benefits:**
- Automatic versioning
- Metadata tracking
- Easy rollback
- Model serving capabilities

---

## Deployment Checklist

### Pre-Deployment

- [ ] Model files exist and are accessible
- [ ] Normalization stats file exists
- [ ] Model version documented in VERSIONS.md
- [ ] Configuration updated in config.yaml
- [ ] Test model loading in isolation
- [ ] Verify predictions are reasonable
- [ ] Previous version backed up

### Deployment Steps

1. **Backup Current Model:**
   ```bash
   # If replacing existing model
   cp models/hdrl_optimized/final_model.zip \
      models/hdrl_optimized/final_model_backup_$(date +%Y%m%d).zip
   ```

2. **Deploy New Model:**
   ```bash
   # Copy versioned model to production name
   cp models/hdrl_optimized/final_model_v1.1.zip \
      models/hdrl_optimized/final_model.zip
   
   cp models/hdrl_optimized/vec_normalize_v1.1.pkl \
      models/hdrl_optimized/vec_normalize.pkl
   ```

3. **Update Configuration:**
   ```yaml
   # config.yaml (if using versioned paths)
   params:
     model_path: "models/hdrl_optimized/final_model.zip"
     vec_norm_path: "models/hdrl_optimized/vec_normalize.pkl"
   ```

4. **Test Deployment:**
   ```bash
   # Test loading
   python -c "
   from src.strategy.ppo_strategy import PPOStrategy
   from pathlib import Path
   strategy = PPOStrategy(...)
   target = strategy.get_target_portfolio()
   print('✓ Model loads successfully')
   "
   ```

5. **Start Bot:**
   ```bash
   python main.py
   ```

6. **Monitor:**
   - Watch first few predictions
   - Check database logs
   - Monitor portfolio performance

---

## Rollback Procedures

### Quick Rollback

If you need to revert to a previous model version:

```bash
# 1. Identify previous version
ls -lt models/hdrl_optimized/final_model_v*.zip

# 2. Restore previous version
VERSION="v1.0"  # Previous working version
cp models/hdrl_optimized/final_model_${VERSION}.zip \
   models/hdrl_optimized/final_model.zip

cp models/hdrl_optimized/vec_normalize_${VERSION}.pkl \
   models/hdrl_optimized/vec_normalize.pkl

# 3. Restart bot
python main.py
```

### Emergency Rollback Script

Create `scripts/rollback_model.sh`:

```bash
#!/bin/bash
# Usage: ./scripts/rollback_model.sh v1.0

VERSION=${1:-"v1.0"}
MODEL_DIR="models/hdrl_optimized"

echo "Rolling back to version $VERSION..."

# Backup current model
if [ -f "$MODEL_DIR/final_model.zip" ]; then
    cp "$MODEL_DIR/final_model.zip" \
       "$MODEL_DIR/final_model_backup_$(date +%Y%m%d_%H%M%S).zip"
fi

# Restore previous version
cp "$MODEL_DIR/final_model_${VERSION}.zip" \
   "$MODEL_DIR/final_model.zip"

cp "$MODEL_DIR/vec_normalize_${VERSION}.pkl" \
   "$MODEL_DIR/vec_normalize.pkl"

echo "✓ Rolled back to version $VERSION"
echo "Restart the bot to apply changes"
```

---

## Model Lifecycle

### Development Phase

1. Train model
2. Test locally
3. Version model (v1.0-dev)
4. Document in VERSIONS.md

### Testing Phase

1. Deploy to test environment
2. Monitor performance
3. Compare with baseline
4. Iterate if needed

### Production Phase

1. Promote to production (v1.0)
2. Update config.yaml
3. Deploy to live bot
4. Monitor closely
5. Document deployment

### Maintenance Phase

1. Regular performance reviews
2. Retrain if needed
3. Version new models
4. Archive old models

---

## File Naming Conventions

### Model Files

```
final_model.zip              # Production model (current)
final_model_v1.0.zip         # Versioned model
final_model_v1.1.zip         # Next version
final_model_backup_20250101.zip  # Backup
```

### Normalization Files

```
vec_normalize.pkl            # Production stats (current)
vec_normalize_v1.0.pkl       # Versioned stats
vec_normalize_v1.1.pkl       # Next version
```

### Checkpoints

```
hdrl_crypto_optimized_1000000_steps.zip
hdrl_crypto_optimized_2000000_steps.zip
hdrl_crypto_optimized_3000000_steps.zip
```

---

## Best Practices Summary

1. **Always version models** after training
2. **Document everything** in VERSIONS.md
3. **Never commit binaries** to git
4. **Keep backups** of production models
5. **Test before deploying** to production
6. **Monitor after deployment** closely
7. **Use cloud storage** for production
8. **Implement rollback procedures**
9. **Track performance metrics** over time
10. **Archive old models** (don't delete)

---

## Quick Reference

### Common Commands

```bash
# List model versions
ls -lh models/hdrl_optimized/final_model_v*.zip

# Check model file size
du -h models/hdrl_optimized/final_model.zip

# Test model loading
python -c "from src.strategy.ppo_strategy import PPOStrategy; ..."

# Rollback to version
cp models/hdrl_optimized/final_model_v1.0.zip \
   models/hdrl_optimized/final_model.zip
```

### File Locations

```
models/hdrl_optimized/
├── final_model.zip          # ← Production (symlink or copy)
├── vec_normalize.pkl        # ← Production stats
├── final_model_v*.zip       # ← Versioned models
├── vec_normalize_v*.pkl     # ← Versioned stats
└── VERSIONS.md             # ← Version history (in git)
```

---

**Last Updated**: 2025-01-XX  
**See Also**: `docs/hdrl_training_guide.md`

