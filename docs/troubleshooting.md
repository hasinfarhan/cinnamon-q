# Troubleshooting Guide

## Import Errors

### urllib3.packages.six.moves Import Error

**Error Message:**
```
ModuleNotFoundError: No module named 'urllib3.packages.six.moves'
```

**Root Cause:**
- `alpaca-trade-api` depends on `urllib3<1.25`
- `urllib3` 1.24.x and 1.26.x have compatibility issues with Python 3.12
- The bundled `six` package in urllib3 doesn't work correctly with Python 3.12

**Solutions:**

1. **Use Python 3.11 (Recommended for now)**
   ```bash
   # Create new virtualenv with Python 3.11
   pyenv install 3.11.9
   pyenv local 3.11.9
   pipenv install
   ```

2. **Use Direct REST API Calls (Future)**
   - Replace `alpaca-trade-api` with direct `requests` calls
   - Similar to how `scripts/08_fetch_crypto_deep.py` works
   - This is the long-term solution

3. **Wait for Updated SDK**
   - Monitor `alpaca-trade-api` for Python 3.12 support
   - Or switch to `alpaca-py` (newer SDK) when stable

**Current Status:**
- The bot works on Python 3.11
- Python 3.12 requires workarounds or alternative implementation
- Direct REST API implementation is planned for v0.3

---

## Other Common Issues

### Missing Model Files

**Error:** `FileNotFoundError: models/hdrl_optimized/final_model.zip`

**Solution:**
```bash
# Train the model first
python scripts/10_train_hdrl_optimized.py

# Or recover normalization stats
python scripts/recover_stats.py
```

### Missing Data Files

**Error:** `FileNotFoundError: data/processed/crypto_training_data.parquet`

**Solution:**
```bash
# Generate the data pipeline
python scripts/08_fetch_crypto_deep.py
python scripts/09_generate_sentiment.py
```

### Database Locked

**Error:** `sqlite3.OperationalError: database is locked`

**Solution:**
- Close any other processes accessing `arynosq.db`
- Check for stale database connections
- Restart the bot

---

## Getting Help

1. Check this troubleshooting guide
2. Review the relevant documentation:
   - [HDRL Training Guide](hdrl_training_guide.md)
   - [Model Management](model_management.md)
   - [Phase 3 Summary](phase3_summary.md)
3. Check GitHub issues (if applicable)
4. Review error logs in detail

