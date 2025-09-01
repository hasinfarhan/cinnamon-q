# Project Structure Documentation

## Overview
This document describes the standard file structure for data science projects and explains the purpose of each directory and file.

## Directory Structure

### `/data/`
Contains all data files organized by processing stage:
- **`raw/`**: Original, immutable data files
- **`processed/`**: Cleaned and transformed data ready for analysis
- **`external/`**: Data from external sources (APIs, downloads, etc.)

### `/notebooks/`
Jupyter notebooks for exploratory data analysis and prototyping:
- **`01_exploratory_data_analysis.ipynb`**: Initial data exploration
- **`02_feature_engineering.ipynb`**: Feature creation and selection
- **`03_model_experimentation.ipynb`**: Model testing and comparison
- **`04_model_evaluation.ipynb`**: Final model assessment

### `/src/`
Source code organized by functionality:
- **`data/`**: Data processing and cleaning scripts
- **`features/`**: Feature engineering and selection
- **`models/`**: Model training, evaluation, and prediction
- **`visualization/`**: Plotting and visualization utilities

### `/tests/`
Unit tests for all source code:
- **`test_data.py`**: Tests for data processing functions
- **`test_features.py`**: Tests for feature engineering
- **`test_models.py`**: Tests for model training and evaluation
- **`conftest.py`**: Pytest configuration and fixtures

### `/config/`
Configuration files:
- **`config.yaml`**: Main configuration file with parameters
- **`logging.yaml`**: Logging configuration (optional)

### `/docs/`
Project documentation:
- **`README.md`**: Project overview and setup instructions
- **`project_structure.md`**: This file
- **`api_documentation.md`**: API documentation (if applicable)

### `/reports/`
Generated outputs:
- **`figures/`**: Plots, charts, and visualizations
- **`models/`**: Saved model artifacts and weights

### `/logs/`
Log files from script execution and model training

## File Naming Conventions

### Data Files
- Use descriptive names: `customer_data_2023.csv`
- Include version numbers: `processed_features_v2.parquet`
- Use consistent formats: `.csv`, `.parquet`, `.pkl`

### Scripts
- Use snake_case: `train_model.py`, `build_features.py`
- Include functionality in name: `load_data.py`, `evaluate_model.py`

### Notebooks
- Use numbered prefixes: `01_eda.ipynb`, `02_feature_engineering.ipynb`
- Include descriptive names: `model_comparison.ipynb`

## Best Practices

### Version Control
- Use `.gitignore` to exclude large data files and model artifacts
- Commit code changes frequently with descriptive messages
- Use branches for experimental features

### Documentation
- Document all functions with docstrings
- Include examples in README
- Keep notebooks clean and well-commented

### Testing
- Write tests for all functions
- Use fixtures for common test data
- Aim for high test coverage

### Configuration
- Use configuration files for parameters
- Never hardcode paths or parameters
- Use environment variables for sensitive data

## Getting Started

1. **Clone the repository**
2. **Set up virtual environment**
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Place data in `/data/raw/`**
5. **Run data processing**: `python src/data/make_dataset.py`
6. **Start with notebooks**: Open `notebooks/01_exploratory_data_analysis.ipynb`
7. **Run tests**: `pytest tests/`

## Maintenance

### Regular Tasks
- Update dependencies in `requirements.txt`
- Clean up old model artifacts in `/reports/models/`
- Archive old notebooks in `/notebooks/archive/`
- Review and update documentation

### Code Quality
- Run linting: `flake8 src/`
- Format code: `black src/`
- Run tests: `pytest tests/`
- Check coverage: `pytest --cov=src tests/`
