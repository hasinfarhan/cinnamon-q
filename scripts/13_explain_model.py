"""
Model Explainability Script - SHAP Analysis

Phase 5.3: The "Glass Box" - Understanding why the AI makes decisions.

This script uses SHAP (Shapley Additive Explanations) to analyze which
features drive the PPO agent's trading decisions.

Research Validation:
- If RSI/MACD are top features: Deep Feature Engineering worked ✓
- If Utility_Score is top: LLM/Sentiment hypothesis is valid ✓
- If random noise is top: Model hasn't converged or features are useless ✗
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from analysis.interpret import ModelInterpreter

# Configuration
MODEL_PATH = project_root / "models" / "hdrl_optimized" / "final_model.zip"
VECNORM_PATH = project_root / "models" / "hdrl_optimized" / "vec_normalize.pkl"
DATA_PATH = project_root / "data" / "processed" / "crypto_deep_v2.parquet"  # Prefer v2 with TA

# Fallback to v1 if v2 doesn't exist
if not DATA_PATH.exists():
    DATA_PATH = project_root / "data" / "processed" / "crypto_training_data.parquet"
    print(f"⚠ Using {DATA_PATH} (v2 not found)")

OUTPUT_DIR = project_root / "reports"


def main():
    """Main execution function."""
    print("=" * 80)
    print("🔍 Model Explainability Analysis - SHAP")
    print("Phase 5.3: The 'Glass Box'")
    print("=" * 80)
    print()
    
    # Verify files exist
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Please train a model first using scripts/10_train_hdrl_optimized.py"
        )
    
    if not VECNORM_PATH.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found: {VECNORM_PATH}\n"
            "Please train a model first to generate normalization statistics"
        )
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}\n"
            "Please run scripts/12_enrich_data.py to generate training data"
        )
    
    # Initialize interpreter
    print("Step 1: Initializing Model Interpreter...")
    interpreter = ModelInterpreter(
        model_path=MODEL_PATH,
        vec_norm_path=VECNORM_PATH,
        data_path=DATA_PATH
    )
    print()
    
    # Prepare data
    print("Step 2: Preparing Data...")
    background, test, feature_names = interpreter.prepare_data(sample_size=50)
    print()
    
    # Run SHAP analysis
    print("Step 3: Running SHAP Analysis...")
    print("  Using KernelExplainer (robust, but slower)")
    print("  This may take 5-15 minutes...")
    print()
    
    shap_values = interpreter.run_analysis(background, test, feature_names)
    print()
    
    # Get top features
    print("Step 4: Analyzing Feature Importance...")
    top_features = interpreter.get_top_features(shap_values, feature_names, top_k=15)
    
    print()
    print("=" * 80)
    print("📊 Top 15 Most Important Features (Mean Absolute SHAP Value)")
    print("=" * 80)
    print()
    
    for rank, (feature, importance) in enumerate(top_features, 1):
        bar_length = int(importance * 100)  # Scale for visualization
        bar = "█" * min(bar_length, 50)  # Cap bar length
        print(f"{rank:2d}. {feature:40s} {importance:8.4f} {bar}")
    
    print()
    print("=" * 80)
    print("🧪 Research Validation")
    print("=" * 80)
    print()
    
    # Check for TA features in top 10
    top_10_features = [f[0] for f in top_features[:10]]
    ta_features_found = [f for f in top_10_features if any(ta in f for ta in ['RSI', 'MACD', 'BBW', 'ATR', 'SMA'])]
    utility_found = [f for f in top_10_features if 'utility' in f.lower()]
    
    if ta_features_found:
        print("✅ SUCCESS: Technical Analysis features are driving decisions!")
        print(f"   Found in top 10: {', '.join(ta_features_found)}")
        print("   → The model IS using RSI, MACD, etc. (Deep Feature Engineering worked!)")
    else:
        print("⚠ WARNING: No TA features in top 10. Model may not be using them effectively.")
        print("   → Consider: Model was trained without TA features, or TA features need better normalization")
    
    if utility_found:
        print("✅ SUCCESS: Utility Score (sentiment) is driving decisions!")
        print(f"   Found in top 10: {', '.join(utility_found)}")
        print("   → The model IS using LLM sentiment scores (Halal validation working!)")
    else:
        print("⚠ WARNING: Utility Score not in top 10. Sentiment may not be influential.")
        print("   → Consider: Sentiment scores may be too noisy, or model needs more training")
    
    print()
    
    # Generate visualization
    print("Step 5: Generating Visualization...")
    plot_path = interpreter.save_plot(shap_values, test, feature_names)
    print(f"✓ Summary plot saved to: {plot_path}")
    print()
    
    print("=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Review the SHAP summary plot: reports/shap_summary.png")
    print("  2. Verify that TA features (RSI, MACD) are in the top features")
    print("  3. If not, consider:")
    print("     - Retraining with more timesteps")
    print("     - Adjusting feature normalization")
    print("     - Checking for feature leakage")
    print()


if __name__ == "__main__":
    main()

