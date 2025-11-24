# Documentation Index

Complete documentation for Project arynosQ (cinnamon-q).

## Core Documentation

### 🧠 HDRL Training & Deployment
- **[HDRL Training Guide](hdrl_training_guide.md)**: Complete guide for training, versioning, and deploying the Hierarchical Deep Reinforcement Learning agent
  - Prerequisites and hardware requirements
  - Step-by-step training workflow
  - Monitoring and metrics
  - Troubleshooting common issues
  - Best practices

### 📦 Model Management
- **[Model Management Guide](model_management.md)**: Best practices for managing ML model artifacts
  - Git strategy (what to track, what to ignore)
  - Versioning workflow
  - Storage solutions (local, cloud, registries)
  - Deployment checklist
  - Rollback procedures

### 🏗️ Architecture
- **[Architecture Overview](architecture.md)**: System architecture and component overview
  - Component breakdown
  - Data flow diagrams
  - System capabilities

### 📁 Project Structure
- **[Project Structure](project_structure.md)**: Directory structure and file organization
  - Directory layout
  - File naming conventions
  - Code organization

## Quick Reference

### Training a New Model

1. **Prepare Data:**
   ```bash
   python scripts/09_generate_sentiment.py
   ```

2. **Train Model:**
   ```bash
   python scripts/10_train_hdrl_optimized.py
   ```

3. **Version Model:**
   ```bash
   VERSION="v1.1"
   cp models/hdrl_optimized/final_model.zip \
      models/hdrl_optimized/final_model_${VERSION}.zip
   ```

4. **Document:**
   - Update `models/hdrl_optimized/VERSIONS.md`
   - Commit metadata to git

### Deploying a Model

1. **Update Config:**
   ```yaml
   # config.yaml
   strategies:
     - name: "crypto_hdrl_v1"
       signal: "ppo"
       params:
         model_path: "models/hdrl_optimized/final_model.zip"
   ```

2. **Test Loading:**
   ```bash
   python -c "from src.strategy.ppo_strategy import PPOStrategy; ..."
   ```

3. **Start Bot:**
   ```bash
   python main.py
   ```

### Troubleshooting

- **Missing vec_normalize.pkl**: Run `python scripts/recover_stats.py`
- **Model not found**: Check `models/hdrl_optimized/` directory
- **Normalization drift**: Ensure `PPOStrategy` loads normalization stats
- **Training issues**: See [HDRL Training Guide - Troubleshooting](hdrl_training_guide.md#troubleshooting)

## Documentation by Topic

### Training
- [HDRL Training Guide](hdrl_training_guide.md) - Complete training workflow
- [Model Management](model_management.md) - Versioning and deployment

### Development
- [Architecture](architecture.md) - System design
- [Project Structure](project_structure.md) - Code organization

### Operations
- [Model Management](model_management.md) - Model lifecycle
- [HDRL Training Guide - Deployment](hdrl_training_guide.md#deployment) - Production deployment

## Contributing to Documentation

When adding new features or making changes:

1. **Update relevant guides** if workflows change
2. **Add to VERSIONS.md** when creating new model versions
3. **Update this index** if adding new documentation files
4. **Keep examples current** with actual code paths

## External Resources

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **TensorBoard**: https://www.tensorflow.org/tensorboard
- **MLflow**: https://mlflow.org/

---

**Last Updated**: 2025-01-XX

