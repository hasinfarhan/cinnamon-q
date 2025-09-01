"""
Visualization script for creating plots and charts.

This script contains functions for creating various visualizations
to understand the data and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_data_distribution(df: pd.DataFrame, columns: List[str], save_path: Optional[Path] = None) -> None:
    """
    Create distribution plots for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to plot
        save_path: Optional path to save the plot
    """
    logger.info("Creating distribution plots...")
    
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            if df[col].dtype in ['object', 'category']:
                df[col].value_counts().plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].tick_params(axis='x', rotation=45)
            else:
                df[col].hist(bins=30, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Plot saved to %s", save_path)
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    Create a correlation heatmap for numerical columns.
    
    Args:
        df: Input DataFrame
        save_path: Optional path to save the plot
    """
    logger.info("Creating correlation matrix...")
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) == 0:
        logger.warning("No numerical columns found for correlation matrix")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Correlation matrix saved to %s", save_path)
    
    plt.show()


def plot_feature_importance(model, feature_names: List[str], save_path: Optional[Path] = None) -> None:
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        save_path: Optional path to save the plot
    """
    logger.info("Creating feature importance plot...")
    
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for easier plotting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Feature importance plot saved to %s", save_path)
    
    plt.show()


def plot_model_performance(y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[Path] = None) -> None:
    """
    Plot model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
    """
    logger.info("Creating model performance plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # ROC Curve (for binary classification)
    from sklearn.metrics import roc_curve, auc
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
    else:
        # For multi-class, show prediction distribution
        axes[1].hist(y_pred, bins=20, alpha=0.7, label='Predictions')
        axes[1].hist(y_true, bins=20, alpha=0.7, label='True Values')
        axes[1].set_title('Prediction Distribution')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info("Model performance plot saved to %s", save_path)
    
    plt.show()


def main():
    """Main function to run visualization examples."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    figures_path = project_root / "reports" / "figures"
    
    try:
        # Example usage (uncomment when you have actual data)
        # df = pd.read_csv(project_root / "data" / "processed" / "features.csv")
        
        # # Plot data distributions
        # numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        # plot_data_distribution(df, numerical_cols, 
        #                       figures_path / "data_distributions.png")
        
        # # Plot correlation matrix
        # plot_correlation_matrix(df, figures_path / "correlation_matrix.png")
        
        logger.info("Visualization pipeline completed successfully")
        
    except Exception as e:
        logger.error("Error in visualization pipeline: %s", str(e))
        raise


if __name__ == "__main__":
    main()
