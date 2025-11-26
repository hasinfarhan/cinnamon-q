# Benchmark Suite Guide - Control Groups for AI Validation

## Overview

The Benchmark Suite provides simple, rule-based trading strategies that serve as "Control Groups" for validating our AI models. If your complex AI can't beat a simple "Buy the Dip" script, the AI is likely overfitted noise.

## The Three Benchmarks

### 1. Buy and Hold (Passive Benchmark)

**Philosophy**: Equal-weight indexing - the "lazy investor"

**Logic**: 
- Assign equal weight (1/N) to every asset in the universe
- Fully invested (0% cash)
- No trading, just hold

**Why**: Tests if AI adds value over "doing nothing"

**Config**:
```yaml
strategies:
  - name: "buy_and_hold_baseline"
    signal: "benchmark_hold"
    interval: "1h"
    params:
      data_path: "data/processed/crypto_deep_v2.parquet"
```

### 2. Mean Reversion (Contrarian Benchmark)

**Philosophy**: "Buy the Dip, Sell the Rip"

**Logic**:
- Calculate 24h price change for all assets
- **Buy**: Assets that dropped > `buy_threshold` (e.g., -5%)
- **Sell**: Assets that rose > `sell_threshold` (e.g., +5%)
- Equal weight among selected assets

**Why**: Tests if AI handles volatility better than simple mean reversion math

**Config**:
```yaml
strategies:
  - name: "mean_reversion_baseline"
    signal: "benchmark_mean_reversion"
    interval: "1h"
    params:
      data_path: "data/processed/crypto_deep_v2.parquet"
      buy_threshold: -0.05  # Buy when price drops 5%
      sell_threshold: 0.05   # Sell when price rises 5%
      lookback_hours: 24     # 24-hour lookback for price change
```

### 3. Trend Following (Momentum Benchmark)

**Philosophy**: Golden Cross - follow the trend

**Logic**:
- Calculate SMA(window) for each asset
- **Buy**: Assets where `current_price > SMA(window)` (uptrend)
- **Sell**: Assets where `current_price < SMA(window)` (downtrend)
- Equal weight among selected assets

**Why**: Tests if AI catches big runs better than a simple moving average

**Config**:
```yaml
strategies:
  - name: "trend_following_baseline"
    signal: "benchmark_trend"
    interval: "1h"
    params:
      data_path: "data/processed/crypto_deep_v2.parquet"
      window: 200  # SMA(200) - long-term trend
```

## How to Use

### Option 1: Run via Orchestrator (Live Trading)

Add benchmark strategies to `config.yaml`:

```yaml
strategies:
  # Your AI strategies
  - name: "crypto_hdrl_v1"
    signal: "ppo"
    interval: "15m"
    params:
      model_path: "models/hdrl_optimized/final_model.zip"
      # ...
  
  # Benchmark strategies
  - name: "buy_and_hold"
    signal: "benchmark_hold"
    interval: "1h"
    params:
      data_path: "data/processed/crypto_deep_v2.parquet"
  
  - name: "mean_reversion"
    signal: "benchmark_mean_reversion"
    interval: "1h"
    params:
      data_path: "data/processed/crypto_deep_v2.parquet"
      buy_threshold: -0.05
      sell_threshold: 0.05
```

Then run:
```bash
python main.py
```

The orchestrator will run all strategies on their schedules.

### Option 2: Run Tournament (Comparison)

Compare all strategies on the same dataset:

```bash
python scripts/14_compare_models.py
```

This will:
1. Evaluate MLP model
2. Evaluate LSTM model
3. Evaluate all benchmarks
4. Rank them by Sortino Ratio
5. Provide analysis

**Expected Output**:
```
🏁 THE TOURNAMENT - FULL RESULTS
================================================================================

Rank   Strategy                  Sortino      Sharpe       Return %     Max DD %    
--------------------------------------------------------------------------------
🥇 1   LSTM (AI)                 0.5234       0.4123       12.45        -8.23
🥈 2   Trend Following           0.4567       0.3891       10.23        -9.45
🥉 3   MLP (AI)                  0.4123       0.3456       8.90         -10.12
  4   Mean Reversion             0.3456       0.3012       6.78         -12.34
  5   Buy & Hold                 0.2345       0.2123       4.56         -15.67
```

## Interpreting Results

### ✅ AI Wins (Good)
If AI models rank #1 or #2:
- **Conclusion**: Complexity adds value
- **Action**: Deploy the winning AI model
- **Insight**: The brain is working, not just overfitting

### ⚠️ Benchmark Wins (Warning)
If a benchmark ranks #1:
- **Conclusion**: Market regime favors simple strategies
- **Possible Causes**:
  - AI is overfitting
  - AI is over-trading (friction costs)
  - Market is in a specialized regime (e.g., massive bull run)
- **Actions**:
  - Reduce trading frequency
  - Add regime detection (switch strategies based on market conditions)
  - Retrain with more data
  - Adjust reward function

### 📊 Regime Detection

**If Trend Following wins**:
- Market is trending (bull or bear)
- AI should learn to follow trends better

**If Mean Reversion wins**:
- Market is choppy/sideways
- AI should learn to buy dips, sell rips

**If Buy & Hold wins**:
- Market is in a massive bull run
- "Doing nothing" beats active trading
- AI is likely over-trading

## The Sanity Floor

**The Golden Rule**: If your AI can't beat Buy & Hold, it's broken.

Buy & Hold is the absolute floor. If your sophisticated LSTM with 155 features can't beat "buy everything and hold," then:
- The AI is overfitted
- The reward function is wrong
- The training data is flawed
- The market regime doesn't favor active trading

## Example: Full Tournament

```bash
python scripts/14_compare_models.py
```

**Output Analysis**:
```
📊 ANALYSIS
================================================================================

✅ AI WINS! LSTM (AI) has the highest Sortino Ratio
   AI outperformed all benchmarks - complexity adds value!

Best AI Model: LSTM (Sortino: 0.5234)
Best Benchmark: Trend Following (Sortino: 0.4567)

✅ AI beats benchmarks - The brain is working!
```

## Integration with Orchestrator

All benchmarks implement the same `get_target_portfolio()` interface as AI strategies, so they're fully compatible with the orchestrator:

```python
# In orchestrator.py
if strategy_config.signal == "benchmark_hold":
    return BuyAndHoldStrategy(data_path=data_path)

if strategy_config.signal == "benchmark_mean_reversion":
    return MeanReversionStrategy(
        data_path=data_path,
        buy_threshold=params.get("buy_threshold", -0.05),
        sell_threshold=params.get("sell_threshold", 0.05)
    )

if strategy_config.signal == "benchmark_trend":
    return TrendFollowingStrategy(
        data_path=data_path,
        window=params.get("window", 200)
    )
```

## Next Steps

1. **Run Tournament**: Compare all strategies
2. **Analyze Results**: See if AI beats benchmarks
3. **If AI Wins**: Deploy with confidence
4. **If Benchmarks Win**: Investigate and improve AI
5. **Regime Detection**: Use benchmark performance to detect market regimes

---

**The Benchmark Suite is your Reality Check. Use it to validate that your AI is actually smart, not just complex.** 🎯

