# Phase 4: Live Data Bridge - Implementation Summary

## Overview

The Live Data Bridge solves the "Model Rot" problem by ensuring the PPO strategy uses **real-time market data** instead of stale static parquet files. This is Phase 4, Task 1 of the Adaptive Intelligence Upgrade.

## Problem Solved

**Before (Phase 3):**
- PPO strategy read static `crypto_training_data.parquet` file
- If you ran the bot on Tuesday, it still thought it was Sunday
- Bot was "blind" to current market prices
- Concept drift: Model trained on old data, making decisions on old data

**After (Phase 4, Task 1):**
- PPO strategy fetches live data from Alpaca API every inference
- Bot sees current market prices (last 500 hours)
- Feature engineering uses recent history for accurate calculations
- Model still frozen (trained yesterday), but sees today's prices

## Implementation Details

### 1. AlpacaProvider.fetch_crypto_history()

**Location:** `src/execution/providers/alpaca.py`

**Method Signature:**
```python
def fetch_crypto_history(
    self,
    symbols: List[str],
    timeframe: str = "1Hour",
    hours: int = 500,
    include_fundamentals: bool = False
) -> pd.DataFrame
```

**Features:**
- Fetches crypto price history from Alpaca Data API
- Returns DataFrame matching training data format
- Supports pagination for large history requests
- Fetches 500 hours (~21 days) of history for rolling feature calculation
- Handles symbol format conversion (BTC/USD format)

**Data Format:**
- Columns: `timestamp`, `ticker`, `open`, `high`, `low`, `close`, `volume`
- Optional: `commit_count_4_weeks`, `utility_score` (synthetic/neutral values)

### 2. PPOStrategy Updates

**Location:** `src/strategy/ppo_strategy.py`

**Changes:**
1. **Added `provider` parameter** to `__init__()`:
   - Optional ExecutionProvider for live data fetching
   - If None, falls back to static parquet file

2. **New `_fetch_live_data()` method**:
   - Fetches live market data using provider
   - Converts asset list to Alpaca symbols (adds /USD)
   - Fetches 500 hours of history for feature calculation
   - Returns DataFrame in training data format

3. **Updated `get_target_portfolio()`**:
   - Replaced `pd.read_parquet(self.data_path)` with `self._fetch_live_data()`
   - Now uses real-time data for predictions

### 3. Orchestrator Integration

**Location:** `src/orchestrator.py`

**Changes:**
- Modified `_create_strategy_engine()` to inject `trading_engine.provider` into PPOStrategy
- Enables live data fetching for all PPO strategies automatically

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Live Data Bridge Flow                           │
└─────────────────────────────────────────────────────────────┘

1. Bot Wakes Up (Every Hour)
   │
   ├─> Orchestrator.run_strategy()
   │
   ├─> PPOStrategy.get_target_portfolio()
   │
   ├─> PPOStrategy._fetch_live_data()
   │   │
   │   ├─> AlpacaProvider.fetch_crypto_history()
   │   │   │
   │   │   ├─> Alpaca Data API: /v1beta3/crypto/us/bars
   │   │   │   ├─> Fetches last 500 hours (1Hour bars)
   │   │   │   ├─> Handles pagination
   │   │   │   └─> Returns DataFrame
   │   │   │
   │   │   └─> Returns: DataFrame with OHLCV + fundamentals
   │   │
   │   └─> Returns: Live DataFrame (500 hours of history)
   │
   ├─> PPOStrategy._prepare_observation()
   │   │
   │   ├─> Calculates returns (pct_change)
   │   ├─> Calculates volatility (rolling std)
   │   ├─> Normalizes commit_count (MinMax)
   │   └─> Extracts utility_score
   │
   ├─> VecNormalize.normalize_obs()
   │   └─> Applies frozen training statistics
   │
   ├─> PPO Model.predict()
   │   └─> Generates portfolio weights
   │
   └─> Returns: TargetPortfolio (with current market data)
```

## Benefits

### 1. Real-Time Price Awareness
- Bot sees current market prices before making decisions
- No more "trading on Sunday's prices on Tuesday"

### 2. Accurate Feature Engineering
- Rolling volatility uses recent 7-day window
- Returns calculated from current prices
- Features reflect current market conditions

### 3. Backward Compatible
- Falls back to static data if provider not set
- Maintains compatibility with existing code
- No breaking changes

### 4. Performance Optimized
- Fetches only necessary history (500 hours)
- Uses efficient Alpaca Data API
- Caches nothing (always fresh)

## Configuration

The Live Data Bridge is **automatically enabled** when:
- PPOStrategy is initialized with a `provider` parameter
- Orchestrator injects `trading_engine.provider` (default behavior)

**No configuration changes needed!** The bot will automatically use live data.

## Testing

To verify the Live Data Bridge is working:

1. **Check Logs:**
   ```
   📡 Fetching live market data for 6 assets...
   ✓ Fetched 500 bars for BTC/USD
   ✓ Fetched 500 bars for ETH/USD
   ...
   ✓ Live data fetched: 3000 bars across 6 assets
   ```

2. **Verify Data Freshness:**
   - Check that timestamps in logs are recent (within last hour)
   - Compare with static parquet file timestamps

3. **Monitor Performance:**
   - Live data fetch should take < 5 seconds
   - Feature engineering should complete quickly

## Next Steps (Phase 4, Task 2)

The next phase will implement:
- **Automated Retraining Pipeline**: Weekly model refresh
- **Rolling Window Training**: Forget ancient history, prioritize recent patterns
- **Model Evaluation**: Compare new vs. old model before promotion
- **Automated Promotion**: Deploy better models automatically

## Troubleshooting

### Issue: "No provider set, using static data"
**Solution:** Ensure orchestrator is injecting provider. Check `src/orchestrator.py` line 140.

### Issue: "No data fetched for any symbols"
**Solution:** 
- Check Alpaca API credentials
- Verify symbols are in correct format (BTC/USD, not BTCUSD)
- Check Alpaca Data API availability

### Issue: "Missing required columns"
**Solution:** Ensure `fetch_crypto_history()` returns all required columns. Check method implementation.

## Performance Considerations

- **API Rate Limits**: Alpaca Data API has rate limits. 500 hours × 6 assets = ~3000 bars per fetch.
- **Network Latency**: Live data fetch adds ~2-5 seconds to inference time.
- **Data Volume**: 500 hours × 6 assets = ~3000 rows. Memory usage is minimal.

## Conclusion

The Live Data Bridge transforms the bot from a "static snapshot" system to a "real-time aware" system. The model remains frozen (trained on historical data), but it now sees current market conditions before making predictions.

This is the foundation for Phase 4, Task 2: Automated Retraining Pipeline, which will periodically refresh the model itself.

