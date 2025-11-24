# HDRL Model Versions

Track different training iterations and their performance here.

## v1.0 (Current)

- **Training Date**: 2025-01-XX
- **Training Timesteps**: 5,944,320 (59% of 10M target)
- **Training Time**: 45.72 minutes
- **Hardware**: RTX 4070 Ti Super, Ryzen 3700X (16 cores)
- **Hyperparameters**:
  - Learning Rate: 0.0003
  - Batch Size: 4096
  - Network Architecture: [512, 512, 512]
  - Parallel Environments: 16
- **Performance Metrics**:
  - Episode Reward Mean: 2,620
  - Explained Variance: 0.962
  - FPS: 2,165
  - Policy Gradient Loss: -0.0447
  - Value Loss: 0.0475
- **Files**:
  - Model: `final_model.zip`
  - Normalization: `vec_normalize.pkl`
- **Status**: ✅ Ready for production

---

## Future Versions

Add new entries here as you train improved models.

