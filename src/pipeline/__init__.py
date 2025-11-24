"""
MLOps Pipeline Package - Autonomous Training Lifecycle

This package provides a unified pipeline for the entire ML lifecycle:
- Data Ingestion
- Feature Engineering
- Model Training
- Model Evaluation
- Model Promotion

All stages are designed to run autonomously in the background while
the main trading bot continues operating.
"""

from .runner import run_full_pipeline

__all__ = ["run_full_pipeline"]

