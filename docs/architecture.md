# Project arynosQ - System Architecture

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Project arynosQ - Complete Pipeline        │
└─────────────────────────────────────────────────────────┘

1. DATA LAYER (Module 1)
   scripts/01_fetch_market_data.py
   ├── Downloads 5-year history (yfinance)
   ├── Processes: AAPL, MSFT, GOOGL, TSLA, NVDA
   └── Saves: data/processed/market_data.parquet

2. STRATEGY LAYER (Module 2 - Brain)
   src/strategy/
   ├── interfaces.py (SignalGenerator, Allocator)
   ├── signals/momentum.py (MomentumSignal)
   ├── allocators/top_k.py (TopKEqualWeight)
   └── engine.py (StrategyEngine orchestrator)

3. EXECUTION LAYER (Module 5 - Body)
   src/execution/
   ├── models.py (TargetPortfolio, PortfolioState, TradeOrder)
   ├── interface.py (ExecutionProvider)
   ├── providers/alpaca.py (AlpacaProvider)
   └── engine.py (TradingEngine with rebalance)

4. INTEGRATION (Nervous System)
   scripts/run_bot.py
   └── Connects Brain → Body → Broker
```

## System Capabilities

### 1. Data Pipeline
- Downloads historical market data
- Processes and normalizes data format
- Stores efficiently in parquet format

### 2. Strategy Engine (Brain)
- Generates momentum signals from price data
- Allocates capital with Halal constraints
- Outputs validated TargetPortfolio

### 3. Execution Engine (Body)
- Connects to Alpaca paper trading
- Gets current portfolio state
- Rebalances to target weights
- Executes trades automatically

### 4. Safety Features
- **Long Only constraint** (no short selling)
- **No Leverage constraint** (sum ≤ 1.0)
- **Bear market safety switch** (100% cash)
- **Idempotency** (prevents duplicate orders)
- **Price slippage prevention** (quantity-based exits)

## Usage

```bash
# Step 1: Fetch market data (one-time setup)
python scripts/01_fetch_market_data.py

# Step 2: Run the bot
python scripts/run_bot.py
```

## What You've Built

✅ **Modular Architecture**: Each component is independent and testable
✅ **Broker-Agnostic**: Easy to add new brokers (IBKR, etc.)
✅ **Type-Safe**: Pydantic validation at every boundary
✅ **Halal-Compliant**: Enforces constraints at the data layer
✅ **Production-Ready**: Error handling, logging, idempotency
✅ **State Reconciliation**: Declarative execution (like Kubernetes)

## Next Steps (Future Enhancements)

- **Smarter Brain**: LLM-based signal generation, Deep RL strategies
- **Stronger Safety**: VaR calculations, risk limits
- **More Data**: News sentiment, alternative data sources
- **Performance**: Caching, optimization, backtesting

---

**The core infrastructure is complete. The hard engineering work is done. You have a working, autonomous, Halal-compliant trading robot.**

