"""
MLOps Pipeline Runner - Master Orchestrator

This module orchestrates the entire ML lifecycle pipeline:
1. Data Ingestion
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Model Promotion

All stages run sequentially, with proper error handling and logging.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline.evaluation import run_evaluation
from pipeline.features import run_feature_engineering
from pipeline.ingestion import run_ingestion
from pipeline.training import run_training

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def promote_model(
    challenger_model_path: Path,
    challenger_vecnorm_path: Path,
    production_dir: Path,
    archive_dir: Path
) -> None:
    """
    Promote challenger model to production (hot swap).
    
    This function:
    1. Archives the current production model
    2. Moves challenger to production
    3. Updates latest_model.json metadata
    
    Args:
        challenger_model_path: Path to challenger model
        challenger_vecnorm_path: Path to challenger normalization stats
        production_dir: Production model directory
        archive_dir: Archive directory for old models
    """
    logger.info("=" * 80)
    logger.info("Stage 5: Model Promotion (Hot Swap)")
    logger.info("=" * 80)
    
    # Create directories
    production_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Archive current production model if it exists
    prod_model_path = production_dir / "final_model.zip"
    prod_vecnorm_path = production_dir / "vec_normalize.pkl"
    
    if prod_model_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_subdir = archive_dir / f"model_{timestamp}"
        archive_subdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Archiving current production model to {archive_subdir}...")
        shutil.copy2(prod_model_path, archive_subdir / "final_model.zip")
        shutil.copy2(prod_vecnorm_path, archive_subdir / "vec_normalize.pkl")
        logger.info("  ✓ Production model archived")
    
    # Promote challenger to production
    logger.info("Promoting challenger to production...")
    shutil.copy2(challenger_model_path, prod_model_path)
    shutil.copy2(challenger_vecnorm_path, prod_vecnorm_path)
    logger.info("  ✓ Challenger promoted to production")
    
    # Update metadata
    metadata = {
        "model_path": str(prod_model_path),
        "vec_norm_path": str(prod_vecnorm_path),
        "promoted_at": datetime.now().isoformat(),
        "version": timestamp if prod_model_path.exists() else "initial"
    }
    
    metadata_path = production_dir / "latest_model.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Metadata updated: {metadata_path}")
    logger.info("=" * 80)


def run_full_pipeline(
    data_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    challenger_dir: Optional[Path] = None,
    archive_dir: Optional[Path] = None,
    production_dir: Optional[Path] = None,
    total_timesteps: int = 10_000_000,
    skip_ingestion: bool = False,
    skip_features: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False
) -> bool:
    """
    Run the complete MLOps pipeline.
    
    This function orchestrates all stages:
    1. Data Ingestion (fetch raw data)
    2. Feature Engineering (add TA indicators)
    3. Model Training (train challenger model)
    4. Model Evaluation (compare challenger vs champion)
    5. Model Promotion (hot swap if challenger wins)
    
    Args:
        data_dir: Directory for data files (default: data/processed)
        model_dir: Directory for model files (default: models/hdrl_optimized)
        challenger_dir: Directory for challenger models (default: models/challenger)
        archive_dir: Directory for archived models (default: models/archive)
        production_dir: Directory for production models (default: models/production)
        total_timesteps: Training timesteps (default: 10M)
        skip_ingestion: Skip data ingestion stage
        skip_features: Skip feature engineering stage
        skip_training: Skip training stage
        skip_evaluation: Skip evaluation stage
    
    Returns:
        True if pipeline completed successfully, False otherwise
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Set default paths
    if data_dir is None:
        data_dir = project_root / "data" / "processed"
    if model_dir is None:
        model_dir = project_root / "models" / "hdrl_optimized"
    if challenger_dir is None:
        challenger_dir = project_root / "models" / "challenger"
    if archive_dir is None:
        archive_dir = project_root / "models" / "archive"
    if production_dir is None:
        production_dir = project_root / "models" / "production"
    
    # Create directories
    for dir_path in [data_dir, model_dir, challenger_dir, archive_dir, production_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🚀 MLOps Pipeline - Full Lifecycle")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("")
    
    try:
        # Stage 1: Data Ingestion
        if not skip_ingestion:
            ingestion_output = run_ingestion(
                output_path=data_dir / "crypto_deep_master.parquet"
            )
        else:
            ingestion_output = data_dir / "crypto_deep_master.parquet"
            logger.info("Skipping ingestion stage (using existing data)")
        
        # Stage 2: Feature Engineering
        if not skip_features:
            features_output = run_feature_engineering(
                input_path=ingestion_output,
                output_path=data_dir / "crypto_deep_v2.parquet"
            )
        else:
            features_output = data_dir / "crypto_deep_v2.parquet"
            logger.info("Skipping feature engineering stage (using existing data)")
        
        # Stage 3: Model Training
        if not skip_training:
            challenger_model_path, challenger_vecnorm_path = run_training(
                data_path=features_output,
                model_output_dir=challenger_dir,
                total_timesteps=total_timesteps
            )
        else:
            challenger_model_path = challenger_dir / "final_model.zip"
            challenger_vecnorm_path = challenger_dir / "vec_normalize.pkl"
            logger.info("Skipping training stage (using existing challenger)")
        
        # Stage 4: Model Evaluation
        if not skip_evaluation:
            champion_model_path = production_dir / "final_model.zip"
            champion_vecnorm_path = production_dir / "vec_normalize.pkl"
            
            # If no production model exists, skip evaluation and promote directly
            if not champion_model_path.exists():
                logger.warning("No production model found, promoting challenger directly")
                promote_model(
                    challenger_model_path,
                    challenger_vecnorm_path,
                    production_dir,
                    archive_dir
                )
                return True
            
            champion_score, challenger_score = run_evaluation(
                champion_model_path=champion_model_path,
                challenger_model_path=challenger_model_path,
                champion_vecnorm_path=champion_vecnorm_path,
                challenger_vecnorm_path=challenger_vecnorm_path,
                data_path=features_output
            )
            
            # Stage 5: Model Promotion (if challenger wins)
            if challenger_score > champion_score:
                promote_model(
                    challenger_model_path,
                    challenger_vecnorm_path,
                    production_dir,
                    archive_dir
                )
                logger.info("✓ Pipeline complete: New model promoted to production")
                return True
            else:
                logger.info("✗ Pipeline complete: Challenger did not outperform, keeping current production model")
                return False
        else:
            logger.info("Skipping evaluation stage")
            return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False
    
    finally:
        logger.info("")
        logger.info(f"Pipeline finished at: {datetime.now().isoformat()}")
        logger.info("=" * 80)


if __name__ == "__main__":
    # Run pipeline when executed directly
    success = run_full_pipeline()
    exit(0 if success else 1)

