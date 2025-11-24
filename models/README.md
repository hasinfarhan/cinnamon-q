# Model Artifacts

This directory contains trained model artifacts (neural network weights, normalization statistics, etc.).

## Git Policy

**Model files are NOT tracked in git** because they are:
- Large binary files (often 100MB+)
- Frequently regenerated during training
- Not suitable for version control (binary diffs are meaningless)

## Model Versioning Strategy

### 1. Semantic Versioning in Filenames

Use versioned filenames to track different training iterations:

```
models/hdrl_optimized/
├── final_model_v1.0.zip          # Initial training
├── final_model_v1.1.zip          # Improved hyperparameters
├── final_model_v1.2.zip          # More training data
├── vec_normalize_v1.0.pkl        # Matching normalization stats
├── vec_normalize_v1.1.pkl
└── vec_normalize_v1.2.pkl
```

### 2. Model Registry (Recommended for Production)

For production systems, use a proper model registry:

- **MLflow**: `mlflow models serve`
- **Weights & Biases**: Artifact storage
- **S3/GCS**: Cloud object storage with versioning
- **DVC**: Data Version Control for ML

### 3. Metadata Tracking

Track model metadata in git (not the models themselves):

```yaml
# models/hdrl_optimized/metadata_v1.0.yaml
version: "1.0"
training_date: "2025-01-XX"
training_timesteps: 5944320
hyperparameters:
  learning_rate: 0.0003
  batch_size: 4096
  network_arch: [512, 512, 512]
performance:
  ep_rew_mean: 2620
  explained_variance: 0.962
data:
  training_data: "crypto_training_data.parquet"
  n_assets: 6
  n_timesteps: 26277
```

### 4. Git Tags for Releases

Tag code releases that correspond to model versions:

```bash
git tag -a v1.0-model -m "HDRL v1.0 model release"
git push origin v1.0-model
```

## Current Model

The orchestrator expects models at:
- Model: `models/hdrl_optimized/final_model.zip`
- Normalization: `models/hdrl_optimized/vec_normalize.pkl`

To use a different version, update `config.yaml`:

```yaml
strategies:
  - name: "crypto_hdrl_v1"
    signal: "ppo"
    params:
      model_path: "models/hdrl_optimized/final_model_v1.1.zip"
      vec_norm_path: "models/hdrl_optimized/vec_normalize_v1.1.pkl"
```

## Backup Strategy

1. **Local**: Keep versioned copies in `models/` directory
2. **Cloud**: Upload to S3/GCS with version tags
3. **Registry**: Use MLflow/W&B for production models
4. **Documentation**: Track which model is in production in this README

## Example Workflow

```bash
# After training
cp models/hdrl_optimized/final_model.zip \
   models/hdrl_optimized/final_model_v1.0.zip
cp models/hdrl_optimized/vec_normalize.pkl \
   models/hdrl_optimized/vec_normalize_v1.0.pkl

# Update metadata
echo "v1.0: Trained on 5.9M timesteps, ep_rew_mean=2620" >> models/hdrl_optimized/VERSIONS.md

# Upload to cloud (optional)
aws s3 cp models/hdrl_optimized/final_model_v1.0.zip \
   s3://your-bucket/models/hdrl_optimized/v1.0/
```

