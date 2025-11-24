"""
arynosQ Dashboard - Observability Cockpit

A Streamlit-based web dashboard for monitoring the trading bot's performance,
portfolio state, and trade history in real-time.

Phase 5: The Glass Box - Complete visibility into the bot's operations.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# Add src to path for database models
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from database.models import PortfolioSnapshot, Trade

# Page configuration
st.set_page_config(
    page_title="arynosQ Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_data():
    """
    Fetch data from SQLite database.
    
    Returns:
        Tuple of (trades_df, snapshots_df)
    """
    # Get database path from config or use default
    db_path = project_root / "arynosq.db"
    
    if not db_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    
    # Create SQLAlchemy engine
    engine = create_engine(f"sqlite:///{db_path}")
    
    # Fetch trades
    try:
        trades_query = text("SELECT * FROM trades ORDER BY timestamp DESC")
        trades_df = pd.read_sql(trades_query, engine)
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        trades_df = pd.DataFrame()
    
    # Fetch snapshots
    try:
        snapshots_query = text("SELECT * FROM portfolio_snapshots ORDER BY timestamp ASC")
        snapshots_df = pd.read_sql(snapshots_query, engine)
        if not snapshots_df.empty:
            snapshots_df['timestamp'] = pd.to_datetime(snapshots_df['timestamp'])
    except Exception as e:
        st.error(f"Error loading snapshots: {e}")
        snapshots_df = pd.DataFrame()
    
    return trades_df, snapshots_df


def calculate_current_positions(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate current positions by replaying trades.
    
    Args:
        trades_df: DataFrame of all trades
    
    Returns:
        DataFrame with columns: symbol, qty, total_cost, avg_price
    """
    if trades_df.empty:
        return pd.DataFrame(columns=['symbol', 'qty', 'total_cost', 'avg_price'])
    
    positions = {}
    
    # Sort by timestamp (oldest first) to replay chronologically
    trades_sorted = trades_df.sort_values('timestamp')
    
    for _, trade in trades_sorted.iterrows():
        symbol = trade['symbol']
        side = trade['side'].lower()
        qty = float(trade['qty']) if pd.notna(trade['qty']) else 0.0
        price = float(trade['price']) if pd.notna(trade['price']) else 0.0
        
        if symbol not in positions:
            positions[symbol] = {'qty': 0.0, 'total_cost': 0.0}
        
        if side == 'buy':
            # Add to position
            old_qty = positions[symbol]['qty']
            old_cost = positions[symbol]['total_cost']
            new_qty = old_qty + qty
            new_cost = old_cost + (qty * price)
            positions[symbol] = {'qty': new_qty, 'total_cost': new_cost}
        elif side == 'sell':
            # Reduce position (FIFO-like, but we'll use average cost)
            old_qty = positions[symbol]['qty']
            old_cost = positions[symbol]['total_cost']
            new_qty = max(0.0, old_qty - qty)  # Can't go negative
            # Reduce cost proportionally
            if old_qty > 0:
                cost_per_unit = old_cost / old_qty
                new_cost = new_qty * cost_per_unit
            else:
                new_cost = 0.0
            positions[symbol] = {'qty': new_qty, 'total_cost': new_cost}
    
    # Convert to DataFrame
    positions_list = []
    for symbol, data in positions.items():
        if data['qty'] > 0:  # Only include non-zero positions
            avg_price = data['total_cost'] / data['qty'] if data['qty'] > 0 else 0.0
            positions_list.append({
                'symbol': symbol,
                'qty': data['qty'],
                'total_cost': data['total_cost'],
                'avg_price': avg_price
            })
    
    return pd.DataFrame(positions_list)


def calculate_metrics(snapshots_df: pd.DataFrame) -> dict:
    """
    Calculate key performance metrics.
    
    Args:
        snapshots_df: DataFrame of portfolio snapshots
    
    Returns:
        Dictionary of metrics
    """
    if snapshots_df.empty:
        return {
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0,
            'current_equity': 0.0,
            'current_cash': 0.0,
            'max_equity': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0
        }
    
    # Basic metrics
    first_equity = snapshots_df['total_equity'].iloc[0]
    last_equity = snapshots_df['total_equity'].iloc[-1]
    current_cash = snapshots_df['cash'].iloc[-1]
    
    total_pnl = last_equity - first_equity
    total_pnl_pct = (total_pnl / first_equity * 100) if first_equity > 0 else 0.0
    
    # Max drawdown
    running_max = snapshots_df['total_equity'].expanding().max()
    drawdown = (snapshots_df['total_equity'] - running_max) / running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    # Risk-adjusted returns (simplified)
    # Calculate daily returns
    snapshots_df = snapshots_df.copy()
    snapshots_df['returns'] = snapshots_df['total_equity'].pct_change()
    
    if len(snapshots_df) > 1:
        mean_return = snapshots_df['returns'].mean()
        std_return = snapshots_df['returns'].std()
        downside_returns = snapshots_df['returns'][snapshots_df['returns'] < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
        
        sharpe_ratio = (mean_return / std_return) if std_return > 0 else 0.0
        sortino_ratio = (mean_return / downside_std) if downside_std > 0 else 0.0
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
    
    return {
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'current_equity': last_equity,
        'current_cash': current_cash,
        'max_equity': snapshots_df['total_equity'].max(),
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio
    }


def main():
    """Main dashboard application."""
    # Header
    st.title("🤖 arynosQ Halal Fund Dashboard")
    st.markdown("**The Glass Box - Complete visibility into the trading bot**")
    st.markdown("---")
    
    # Sidebar - Auto-refresh and controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Status indicator
        st.header("📊 Status")
        db_path = project_root / "arynosq.db"
        if db_path.exists():
            db_mtime = datetime.fromtimestamp(db_path.stat().st_mtime)
            time_since_update = datetime.now() - db_mtime
            
            if time_since_update < timedelta(minutes=5):
                st.success("🟢 Bot Active")
            elif time_since_update < timedelta(minutes=30):
                st.warning("🟡 Bot Idle")
            else:
                st.error("🔴 Bot Inactive")
            
            st.caption(f"Last DB update: {time_since_update}")
        else:
            st.error("🔴 Database not found")
        
        st.markdown("---")
        
        # Database info
        if db_path.exists():
            db_size = db_path.stat().st_size / 1024  # KB
            st.caption(f"Database size: {db_size:.1f} KB")
    
    # Load data
    trades_df, snapshots_df = get_data()
    
    # KPI Row
    st.header("📈 Key Performance Indicators")
    
    if not snapshots_df.empty:
        metrics = calculate_metrics(snapshots_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total P&L",
                f"${metrics['total_pnl']:,.2f}",
                f"{metrics['total_pnl_pct']:+.2f}%"
            )
        
        with col2:
            st.metric(
                "Current Equity",
                f"${metrics['current_equity']:,.2f}"
            )
        
        with col3:
            st.metric(
                "Current Cash",
                f"${metrics['current_cash']:,.2f}"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.2%}"
            )
        
        # Risk metrics row
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.3f}"
            )
        
        with col6:
            st.metric(
                "Sortino Ratio",
                f"{metrics['sortino_ratio']:.3f}"
            )
        
        with col7:
            st.metric(
                "Peak Equity",
                f"${metrics['max_equity']:,.2f}"
            )
    else:
        st.warning("⚠ No snapshot data available. Bot may not have run yet.")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🧠 Portfolio", "📜 Audit Log"])
    
    # Tab 1: Performance
    with tab1:
        st.header("Performance Analytics")
        
        if not snapshots_df.empty:
            # Equity Curve
            st.subheader("Equity Curve")
            fig_equity = px.line(
                snapshots_df,
                x='timestamp',
                y='total_equity',
                title='Portfolio Equity Over Time',
                labels={'total_equity': 'Total Equity ($)', 'timestamp': 'Time'}
            )
            fig_equity.update_traces(line_color='#1f77b4', line_width=2)
            st.plotly_chart(fig_equity, use_container_width=True)
            
            # Cash and Equity
            st.subheader("Cash vs Equity")
            fig_cash = go.Figure()
            fig_cash.add_trace(go.Scatter(
                x=snapshots_df['timestamp'],
                y=snapshots_df['total_equity'],
                name='Total Equity',
                line=dict(color='#1f77b4', width=2)
            ))
            fig_cash.add_trace(go.Scatter(
                x=snapshots_df['timestamp'],
                y=snapshots_df['cash'],
                name='Cash',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig_cash.update_layout(
                title='Cash and Equity Over Time',
                xaxis_title='Time',
                yaxis_title='Amount ($)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_cash, use_container_width=True)
            
            # Returns distribution
            if len(snapshots_df) > 1:
                st.subheader("Returns Distribution")
                snapshots_df_copy = snapshots_df.copy()
                snapshots_df_copy['returns'] = snapshots_df_copy['total_equity'].pct_change()
                returns_data = snapshots_df_copy['returns'].dropna()
                
                if len(returns_data) > 0:
                    fig_returns = px.histogram(
                        returns_data,
                        nbins=50,
                        title='Distribution of Returns',
                        labels={'value': 'Return', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
        else:
            st.info("No performance data available yet. Run the bot to generate snapshots.")
    
    # Tab 2: Portfolio
    with tab2:
        st.header("Current Portfolio")
        
        if not trades_df.empty:
            # Calculate current positions
            positions_df = calculate_current_positions(trades_df)
            
            if not positions_df.empty:
                # Calculate current market values (simplified - use avg_price * qty)
                positions_df['market_value'] = positions_df['qty'] * positions_df['avg_price']
                total_value = positions_df['market_value'].sum()
                positions_df['weight'] = (positions_df['market_value'] / total_value * 100) if total_value > 0 else 0.0
                
                # Pie Chart
                st.subheader("Portfolio Allocation")
                fig_pie = px.pie(
                    positions_df,
                    values='market_value',
                    names='symbol',
                    title='Current Portfolio Allocation'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Positions Table
                st.subheader("Position Details")
                display_positions = positions_df[['symbol', 'qty', 'avg_price', 'market_value', 'weight']].copy()
                display_positions.columns = ['Symbol', 'Quantity', 'Avg Price', 'Market Value', 'Weight (%)']
                display_positions['Market Value'] = display_positions['Market Value'].apply(lambda x: f"${x:,.2f}")
                display_positions['Avg Price'] = display_positions['Avg Price'].apply(lambda x: f"${x:.4f}")
                display_positions['Weight (%)'] = display_positions['Weight (%)'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(display_positions, use_container_width=True, hide_index=True)
            else:
                st.info("No open positions. Portfolio is 100% cash.")
        else:
            st.info("No trade data available yet. Run the bot to generate trades.")
    
    # Tab 3: Audit Log
    with tab3:
        st.header("Trade Audit Log (Burhan)")
        
        if not trades_df.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                strategy_filter = st.multiselect(
                    "Filter by Strategy",
                    options=sorted(trades_df['strategy_name'].unique()) if 'strategy_name' in trades_df.columns else [],
                    default=[]
                )
            
            with col2:
                symbol_filter = st.multiselect(
                    "Filter by Symbol",
                    options=sorted(trades_df['symbol'].unique()) if 'symbol' in trades_df.columns else [],
                    default=[]
                )
            
            with col3:
                side_filter = st.multiselect(
                    "Filter by Side",
                    options=['buy', 'sell'] if 'side' in trades_df.columns else [],
                    default=[]
                )
            
            # Apply filters
            filtered_trades = trades_df.copy()
            
            if strategy_filter and 'strategy_name' in filtered_trades.columns:
                filtered_trades = filtered_trades[filtered_trades['strategy_name'].isin(strategy_filter)]
            
            if symbol_filter and 'symbol' in filtered_trades.columns:
                filtered_trades = filtered_trades[filtered_trades['symbol'].isin(symbol_filter)]
            
            if side_filter and 'side' in filtered_trades.columns:
                filtered_trades = filtered_trades[filtered_trades['side'].isin(side_filter)]
            
            # Display trades
            st.subheader(f"Trades ({len(filtered_trades)} total)")
            
            # Format for display
            display_trades = filtered_trades.copy()
            if 'timestamp' in display_trades.columns:
                display_trades['timestamp'] = display_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Select and rename columns for display
            display_columns = {
                'timestamp': 'Time',
                'strategy_name': 'Strategy',
                'symbol': 'Symbol',
                'side': 'Side',
                'qty': 'Quantity',
                'price': 'Price',
                'notional': 'Notional'
            }
            
            available_columns = {k: v for k, v in display_columns.items() if k in display_trades.columns}
            display_trades = display_trades[list(available_columns.keys())]
            display_trades = display_trades.rename(columns=available_columns)
            
            # Format numeric columns
            if 'Price' in display_trades.columns:
                display_trades['Price'] = display_trades['Price'].apply(lambda x: f"${x:.4f}" if pd.notna(x) else "N/A")
            if 'Notional' in display_trades.columns:
                display_trades['Notional'] = display_trades['Notional'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
            if 'Quantity' in display_trades.columns:
                display_trades['Quantity'] = display_trades['Quantity'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
            
            st.dataframe(
                display_trades,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Download button
            csv = filtered_trades.to_csv(index=False)
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv,
                file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trade data available yet. Run the bot to generate trades.")
    
    # Auto-refresh (using streamlit-autorefresh if available)
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            # Auto-refresh every 30 seconds
            st_autorefresh(interval=30000, key="dashboard_refresh")
        except ImportError:
            # Fallback: manual refresh button
            st.caption("💡 Install streamlit-autorefresh for automatic refresh")


if __name__ == "__main__":
    main()

