# Project Context Summary - arynosQ (cinnamon-q)

**Last Updated**: Current Session  
**Python Version**: 3.11 (required - 3.12 has compatibility issues)  
**Project Status**: Production-Ready (v1.2 High-Frequency Upgrade)

---

## 🎯 Project Overview

**arynosQ** (also known as **cinnamon-q**) is a Halal-compliant, autonomous crypto trading bot built with:
- **Hierarchical Deep Reinforcement Learning (HDRL)** agent for portfolio allocation
- **Multi-modal data synthesis**: Technical analysis, fundamental metrics, and LLM-scored sentiment
- **Broker-agnostic execution** using the Adapter Pattern
- **State reconciliation** pattern (Kubernetes/Terraform-style declarative rebalancing)
- **Continuous learning pipeline** with automated retraining

### Core Principles
- **Halal Compliance**: Long-only, no leverage (sum of weights ≤ 1.0)
- **Strict Typing**: Pydantic v2 models throughout
- **Idempotency**: All trades use `client_order_id` for safety
- **Configuration as Code**: YAML-based configuration system
- **State Persistence**: SQLite database for trades and portfolio snapshots

---

## 🏗️ System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │  Alpaca  │  │CoinGecko │  │CryptoPanic│                 │
│  │ (Price)  │  │(Fundamentals)│ (News)   │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       └─────────────┴─────────────┘                          │
│                    │                                         │
│            crypto_training_data.parquet                      │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              Strategy Engine (Brain)                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  PPOStrategy (RL Agent)                      │           │
│  │  - Loads trained PPO model                   │           │
│  │  - Fetches live data from Alpaca            │           │
│  │  - Applies normalization stats               │           │
│  │  - Generates TargetPortfolio                │           │
│  └──────────────────────────────────────────────┘           │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              Execution Engine (Body)                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  TradingEngine                              │           │
│  │  - Compares Target vs Current State         │           │
│  │  - Generates TradeOrders                    │           │
│  │  - Executes via AlpacaProvider              │           │
│  │  - Logs to Database                        │           │
│  └──────────────────────────────────────────────┘           │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              Orchestrator (Nervous System)                    │
│  - Multi-timeframe strategy execution                        │
│  - State persistence                                         │
│  - Error handling and recovery                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 Key Components

### 1. **Data Layer**
- **`scripts/08_fetch_crypto_deep.py`**: Fetches 3 years of crypto data (15-minute bars)
  - Sources: Alpaca Data API (price), CoinGecko (fundamentals)
  - Universe: 28 Halal-compliant assets (from `data/raw/crypto_universe.csv`)
  - Output: `data/processed/crypto_deep_master.parquet`

- **`scripts/09_generate_sentiment.py`**: Adds LLM-scored sentiment
  - Sources: CryptoPanic API (primary), yfinance (fallback)
  - LLM: OpenAI GPT-4o-mini for "Halal Utility Scale" (0.0 = Hype, 1.0 = Utility)
  - Output: `data/processed/crypto_training_data.parquet` (with `utility_score`)

- **`src/data/news.py`**: `NewsFetcher` class
  - CryptoPanic API integration (Developer tier: 100 req/mo, 24h delay)
  - yfinance fallback for broader coverage
  - Deduplication and sorting by timestamp

### 2. **Strategy Layer**
- **`src/strategy/rl/fast_env.py`**: `FastCryptoEnv` (Gymnasium environment)
  - Observation: Returns, Volatility, Commit Count, Utility Score (per asset)
  - Action: Portfolio weights (N assets + cash)
  - Reward: Sortino Ratio (risk-adjusted return, downside focus)
  - Optimized: Memory-mapped data loading, Numpy arrays

- **`src/strategy/rl/data_loader.py`**: Data preprocessing
  - Vectorized Pandas operations (no Python loops)
  - Memory-mapped `.npy` files for zero-copy subprocess access
  - Pivots long format → wide format (Time × Assets × Features)

- **`src/strategy/ppo_strategy.py`**: `PPOStrategy` (Inference Engine)
  - Loads trained PPO model and `VecNormalize` stats
  - Fetches live data via `AlpacaProvider.fetch_crypto_history()`
  - Feature engineering (matches training environment)
  - Normalization drift prevention (critical!)
  - Returns `TargetPortfolio` with Halal constraints

- **`scripts/10_train_hdrl_optimized.py`**: Training script
  - 16 parallel environments (`SubprocVecEnv`)
  - GPU optimization: batch_size=4096, network=[512,512,512]
  - Early stopping (`StopTrainingOnNoModelImprovement`)
  - Memory-mapped data for instant subprocess loading
  - Output: `models/hdrl_optimized/final_model.zip` + `vec_normalize.pkl`

- **`scripts/11_retrain_model.py`**: Automated retraining (Challenger Protocol)
  - Transfer learning from Champion model
  - Rolling window training (last 6 months)
  - Out-of-sample evaluation (Sortino Ratio comparison)
  - Automated promotion if Challenger wins

### 3. **Execution Layer**
- **`src/execution/models.py`**: Core data models
  - `TargetPortfolio`: Desired state (weights per asset)
  - `PortfolioState`: Current state (cash, equity, positions, valuations)
  - `TradeOrder`: Order specification
  - All Pydantic v2 models with strict validation

- **`src/execution/interface.py`**: `ExecutionProvider` (ABC)
  - Abstract methods: `get_state()`, `execute_order()`, `fetch_crypto_history()`

- **`src/execution/providers/alpaca.py`**: `AlpacaProvider`
  - Implements `ExecutionProvider` for Alpaca API
  - Handles symbol normalization (BTC/USD vs BTCUSD)
  - Crypto vs Stock detection (time_in_force: 'gtc' vs 'day')
  - Live data fetching: `fetch_crypto_history()` (500 hours, 15-minute bars)
  - Notional → Quantity conversion for orders

- **`src/execution/engine.py`**: `TradingEngine`
  - State reconciliation: Compares `TargetPortfolio` vs `PortfolioState`
  - Generates `TradeOrder` objects
  - Executes sells before buys (liquidity management)
  - Full exit handling (quantity-based for 0% targets)
  - Database logging via `DatabaseService`

### 4. **Orchestration Layer**
- **`src/orchestrator.py`**: `BotOrchestrator`
  - Multi-timeframe strategy execution (15m, 1h, 1d)
  - Strategy factory: Creates `PPOStrategy` or `StrategyEngine` from config
  - Injects `TradingEngine.provider` into `PPOStrategy` for live data
  - State persistence via `DatabaseService`
  - Main loop with heartbeat interval

- **`main.py`**: Entry point
  - Loads `config.yaml`
  - Initializes `BotOrchestrator`
  - Runs continuous loop

### 5. **Configuration & Persistence**
- **`config.yaml`**: Configuration as Code
  - System settings (heartbeat, database URL)
  - Strategy definitions (signal type, interval, params)
  - Execution settings (provider, paper/live)
  - Data settings (tickers, period, output path)

- **`src/config/schema.py`**: Pydantic models for config validation
  - `SystemConfig`, `StrategyConfig`, `ExecutionConfig`, `DataConfig`, `AppConfig`

- **`src/config/loader.py`**: YAML loader with validation

- **`src/database/`**: SQLAlchemy ORM
  - `models.py`: `Trade`, `PortfolioSnapshot` models
  - `service.py`: `DatabaseService` for logging
  - SQLite database: `arynosq.db` (gitignored)

---

## 🚀 Current Status (v1.2 High-Frequency Upgrade)

### ✅ Completed Features

1. **Phase 3: HDRL Brain**
   - ✅ Multi-modal data pipeline (Technical + Fundamental + Sentiment)
   - ✅ Custom Gymnasium environment (`FastCryptoEnv`)
   - ✅ PPO training infrastructure (optimized for RTX 4070 Ti Super)
   - ✅ RL inference engine (`PPOStrategy`)
   - ✅ Model versioning and management

2. **Phase 4: Adaptive Intelligence**
   - ✅ Live Data Bridge (`PPOStrategy` fetches real-time data)
   - ✅ Automated retraining pipeline (`scripts/11_retrain_model.py`)
   - ✅ Challenger Protocol (model evaluation and promotion)

3. **v1.2 High-Frequency Upgrade**
   - ✅ Expanded universe: 28 Halal-compliant crypto assets
   - ✅ 15-minute strategy refresh (96 decisions/day)
   - ✅ Daily retraining schedule
   - ✅ CryptoPanic API integration for news

### 📊 Current Configuration

**Strategy**: `crypto_hdrl_v1`
- **Signal**: `ppo` (PPO RL agent)
- **Interval**: `15m` (every 15 minutes)
- **Window Size**: 96 (24 hours of 15m bars)
- **Training Window**: 300 (3 days for normalization)

**Universe**: 28 assets from `data/raw/crypto_universe.csv`
- BTC, ETH, SOL, LINK, UNI, AAVE, ALGO, AVAX, BAT, BCH, GRT, LTC, MKR, SUSHI, PAXG, YFI, CRV, etc.

**Execution**: Alpaca Paper Trading
- Provider: `alpaca`
- Paper mode: `true`

---

## 🔧 Known Issues & Pending Tasks

### ⚠️ Current Issues

1. **Training Performance** (Partially Resolved)
   - **Issue**: 30-minute hang during environment initialization
   - **Root Cause**: Redundant data loading in parallel environments
   - **Fix Applied**: Memory-mapped data loading (`data_loader.py`)
   - **Status**: Optimized, but may still need monitoring

2. **Data Preprocessing Performance**
   - **Issue**: Slow pivoting for large datasets (105k timestamps × 17 assets)
   - **Fix Applied**: Vectorized Pandas operations (no Python loops)
   - **Status**: Should be fast now (~2 seconds)

3. **CryptoPanic API Limits**
   - **Current Tier**: Developer (100 req/mo, 24h delay, 2 req/sec)
   - **Impact**: Limited news coverage, delayed sentiment
   - **Mitigation**: yfinance fallback, neutral scores for older data

### 📋 Pending Tasks

1. **Documentation** (v1.2)
   - [ ] Update documentation for v1.2 changes
   - [ ] Document CryptoPanic API setup and limits
   - [ ] Add troubleshooting for high-frequency execution

2. **Monitoring & Observability**
   - [ ] Real-time performance dashboard
   - [ ] Alerting for model drift
   - [ ] Trade execution quality metrics

3. **Optimization Opportunities**
   - [ ] Cache normalization stats in memory
   - [ ] Batch API calls for multiple assets
   - [ ] Incremental data updates (append-only)

---

## 🗂️ Important Files & Directories

### Configuration
- `config.yaml` - Main configuration file
- `.env` - API keys (gitignored, use `.env.example` as template)

### Data
- `data/raw/crypto_universe.csv` - Asset universe (28 Halal assets)
- `data/processed/crypto_deep_master.parquet` - Raw multi-modal data
- `data/processed/crypto_training_data.parquet` - Training data (with sentiment)

### Models
- `models/hdrl_optimized/final_model.zip` - Production PPO model
- `models/hdrl_optimized/vec_normalize.pkl` - Normalization statistics (CRITICAL!)
- `models/challenger/` - Challenger model directory (for retraining)

### Scripts
- `scripts/08_fetch_crypto_deep.py` - Data pipeline (price + fundamentals)
- `scripts/09_generate_sentiment.py` - Sentiment generation (LLM scoring)
- `scripts/10_train_hdrl_optimized.py` - PPO training
- `scripts/11_retrain_model.py` - Automated retraining
- `scripts/recover_stats.py` - Recover missing normalization stats

### Source Code
- `src/strategy/ppo_strategy.py` - RL inference engine
- `src/strategy/rl/fast_env.py` - Training environment
- `src/strategy/rl/data_loader.py` - Data preprocessing
- `src/execution/providers/alpaca.py` - Alpaca API adapter
- `src/orchestrator.py` - Main orchestrator
- `main.py` - Entry point

### Documentation
- `docs/phase3_summary.md` - Phase 3 complete summary
- `docs/phase4_live_data_bridge.md` - Live data bridge implementation
- `docs/phase4_retraining_pipeline.md` - Retraining pipeline docs
- `docs/hdrl_training_guide.md` - Training guide
- `docs/model_management.md` - Model versioning guide
- `docs/cryptopanic_setup.md` - CryptoPanic API setup

### Database
- `arynosq.db` - SQLite database (gitignored)
  - `trades` table: All executed trades
  - `portfolio_snapshots` table: Portfolio state history

---

## 🔑 Key Technical Decisions

### 1. **Python 3.11 Requirement**
- **Reason**: `alpaca-trade-api` incompatible with Python 3.12 (`urllib3` issues)
- **Workaround**: Direct `requests` calls for data fetching (bypasses `alpaca-trade-api`)

### 2. **Memory-Mapped Data Loading**
- **Reason**: Eliminate redundant data loading in parallel environments
- **Implementation**: Preprocess once, save `.npy` files, memory-map in subprocesses
- **Benefit**: 30-minute hang → ~3 seconds

### 3. **Vectorized Pandas Operations**
- **Reason**: Python loops too slow for large datasets (105k timestamps)
- **Implementation**: `groupby().transform()`, `pivot()`, vectorized operations
- **Benefit**: Data preprocessing ~2 seconds (vs 30+ minutes)

### 4. **Normalization Drift Prevention**
- **Reason**: RL models trained on normalized data must use same normalization in production
- **Implementation**: `VecNormalize` stats saved/loaded with model
- **Critical**: Missing stats = broken inference (silent failure)

### 5. **Live Data Bridge**
- **Reason**: Model rot - static data becomes stale
- **Implementation**: `PPOStrategy` fetches live data via `AlpacaProvider`
- **Benefit**: Model sees current prices before making decisions

### 6. **Challenger Protocol**
- **Reason**: Continuous learning to adapt to market changes
- **Implementation**: Weekly retraining, evaluation, automated promotion
- **Benefit**: Model stays current without manual intervention

---

## 🚦 Quick Start Guide

### 1. **Initial Setup**
```bash
# Install Python 3.11 (see docs/setup_python311.md)
# Create virtual environment
pipenv install --python 3.11

# Install dependencies
pipenv install

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
# - ALPACA_API_KEY
# - ALPACA_SECRET_KEY
# - ALPACA_BASE_URL (paper: https://paper-api.alpaca.markets)
# - OPENAI_API_KEY (optional, for LLM sentiment)
# - CRYPTOPANIC_API_KEY (optional, for news)
```

### 2. **Data Pipeline**
```bash
# Step 1: Fetch price + fundamentals
python scripts/08_fetch_crypto_deep.py

# Step 2: Generate sentiment scores
python scripts/09_generate_sentiment.py

# Output: data/processed/crypto_training_data.parquet
```

### 3. **Train Model**
```bash
# Train PPO agent
python scripts/10_train_hdrl_optimized.py

# Output:
# - models/hdrl_optimized/final_model.zip
# - models/hdrl_optimized/vec_normalize.pkl
```

### 4. **Run Bot**
```bash
# Start the trading bot
python main.py

# Bot will:
# - Load config.yaml
# - Initialize PPOStrategy with live data fetching
# - Execute trades every 15 minutes
# - Log to arynosq.db
```

### 5. **Retrain Model** (Weekly/Daily)
```bash
# Automated retraining (Challenger Protocol)
python scripts/11_retrain_model.py

# This will:
# - Load latest data
# - Train challenger model
# - Evaluate vs champion
# - Promote if better
```

---

## 📈 Performance Metrics

### Training Metrics (Target)
- Episode Reward Mean: >2000
- Explained Variance: >0.9
- Training Time: <1 hour (10M timesteps)
- FPS: >2000

### Production Metrics (Target)
- Sortino Ratio: >1.0
- Maximum Drawdown: <20%
- Win Rate: >50%
- Sharpe Ratio: >1.0

### System Metrics
- Inference Latency: <5 seconds
- Data Freshness: <15 minutes
- API Rate Limits: Monitor CryptoPanic (100 req/mo)

---

## 🔍 Troubleshooting

### Common Issues

1. **"Unknown signal type 'ppo'"**
   - **Fix**: Ensure `'ppo'` is in `valid_signals` in `src/config/loader.py`

2. **"ModuleNotFoundError: urllib3.packages.six.moves"**
   - **Fix**: Use Python 3.11 (see `docs/troubleshooting.md`)

3. **"Missing normalization stats"**
   - **Fix**: Run `python scripts/recover_stats.py`

4. **"No news found for [asset]"**
   - **Fix**: Check CryptoPanic API key, verify API tier limits

5. **"30-minute hang during training"**
   - **Fix**: Ensure memory-mapped data loading is working (check `data_loader.py`)

### Debug Commands
```bash
# Test PPOStrategy loading
python -c "from src.strategy.ppo_strategy import PPOStrategy; ..."

# Check database
sqlite3 arynosq.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10;"

# Verify data freshness
python -c "import pandas as pd; df = pd.read_parquet('data/processed/crypto_training_data.parquet'); print(df['timestamp'].max())"
```

---

## 📚 Additional Resources

- **Architecture**: `docs/architecture.md`
- **Training Guide**: `docs/hdrl_training_guide.md`
- **Model Management**: `docs/model_management.md`
- **Troubleshooting**: `docs/troubleshooting.md`
- **CryptoPanic Setup**: `docs/cryptopanic_setup.md`
- **Python 3.11 Setup**: `docs/setup_python311.md`

---

## 🎯 Next Steps

1. **Monitor Production Performance**
   - Track Sortino Ratio, drawdown, win rate
   - Review database logs regularly
   - Compare RL agent vs baseline strategies

2. **Optimize Data Pipeline**
   - Upgrade CryptoPanic API tier (if needed)
   - Add more news sources
   - Real-time data streaming

3. **Enhance Model**
   - Longer training (complete 10M timesteps)
   - Hyperparameter tuning
   - Ensemble methods

4. **Infrastructure**
   - Cloud deployment
   - Real-time monitoring dashboard
   - Automated alerting

---

**Project Status**: ✅ **Production-Ready (v1.2)**  
**Last Major Update**: v1.2 High-Frequency Upgrade  
**Next Milestone**: Production monitoring and optimization

