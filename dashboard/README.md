# arynosQ Dashboard

A Streamlit-based web dashboard for monitoring the trading bot's performance, portfolio state, and trade history in real-time.

## Features

### 📈 Performance Tab
- **Equity Curve**: Line chart showing portfolio value over time
- **Cash vs Equity**: Dual-line chart tracking cash and total equity
- **Returns Distribution**: Histogram of portfolio returns

### 🧠 Portfolio Tab
- **Current Allocations**: Pie chart of current portfolio holdings
- **Position Details**: Table showing quantity, average price, market value, and weight for each position

### 📜 Audit Log Tab
- **Trade History**: Searchable, filterable table of all trades
- **Filters**: Filter by strategy, symbol, and side (buy/sell)
- **Export**: Download trades as CSV

### 📊 Key Performance Indicators
- Total P&L (absolute and percentage)
- Current Equity and Cash
- Max Drawdown
- Sharpe Ratio
- Sortino Ratio
- Peak Equity

## Installation

```bash
# Install dependencies
pipenv install streamlit plotly streamlit-autorefresh

# Or using pip
pip install streamlit plotly streamlit-autorefresh
```

## Usage

### Start the Dashboard

```bash
# From project root
streamlit run dashboard/app.py

# Or using pipenv
pipenv run streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Access from Network

To access from other devices (e.g., your phone):

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

Then access via `http://<your-ip>:8501`

## Features

### Auto-Refresh
- Toggle auto-refresh in the sidebar
- Refreshes every 30 seconds when enabled
- Manual refresh button available

### Status Indicator
- 🟢 Green: Bot active (DB updated in last 5 minutes)
- 🟡 Yellow: Bot idle (DB updated in last 30 minutes)
- 🔴 Red: Bot inactive (DB not updated in 30+ minutes)

### Data Caching
- Data is cached for 30 seconds to improve performance
- Click "Refresh Now" to force immediate update

## Troubleshooting

### Dashboard shows "No data available"
- Ensure the bot has run at least once (`python main.py`)
- Check that `arynosq.db` exists in the project root
- Verify the database contains trades and snapshots

### Auto-refresh not working
- Install `streamlit-autorefresh`: `pip install streamlit-autorefresh`
- Or use the manual refresh button

### Port already in use
- Change the port: `streamlit run dashboard/app.py --server.port 8502`

## Architecture

The dashboard reads directly from the SQLite database (`arynosq.db`), which serves as the single source of truth. No API layer is needed - the dashboard queries the database directly.

## Future Enhancements

- Real-time WebSocket updates
- Email/SMS alerts for significant events
- Strategy performance comparison
- Backtesting results visualization
- Model training progress monitoring

