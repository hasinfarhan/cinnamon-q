"""
Thesis Backtesting & Evaluation Script
=======================================
Runs the trained RecurrentPPO (LSTM) model on a held-out test period
(last 20% of data, Oct 2025 - May 2026) and compares against benchmarks.

Benchmarks:
  - Buy-and-hold BTC
  - Equal-weight (all 17 assets, rebalanced each step)

Metrics (all annualised where appropriate):
  - Total Return
  - Annualised Return
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Calmar Ratio
  - Win Rate (% of steps with positive return)

Outputs:
  - reports/backtest/performance_table.csv
  - reports/backtest/portfolio_values.csv
  - reports/backtest/equity_curve.png
  - reports/backtest/drawdown.png
  - reports/backtest/summary.txt
"""

import io
import os
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "strategy" / "rl"))

# ── PyTorch compatibility fix (ZipExtFile → BytesIO) ───────
import torch as _torch
if not getattr(_torch.load, "_bytesio_patched", False):
    _orig_load = _torch.load
    def _compat_load(f, *args, **kwargs):
        if not isinstance(f, (str, bytes, io.BytesIO)) and hasattr(f, "read"):
            f = io.BytesIO(f.read())
        return _orig_load(f, *args, **kwargs)
    _compat_load._bytesio_patched = True
    _torch.load = _compat_load

import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from strategy.rl.data_loader import prepare_training_data
from strategy.rl.fast_env import FastCryptoEnv

# ── Config ─────────────────────────────────────────────────
MODEL_PATH    = project_root / "models" / "hdrl_worldclass_v40" / "final_model.zip"
VECNORM_PATH  = project_root / "models" / "hdrl_worldclass_v40" / "vec_normalize.pkl"
_v5 = project_root / "data" / "processed" / "crypto_deep_v5.parquet"
DATA_PATH     = _v5 if _v5.exists() else project_root / "data" / "processed" / "crypto_deep_v4_sentiment.parquet"
REPORT_DIR    = project_root / "reports" / "backtest_v40"
INITIAL_CASH  = 100_000.0
TEST_FRACTION = 0.20          # last 20% = held-out test set
BARS_PER_YEAR = 365.25 * 6   # 4h bars per year (v27)
RISK_FREE_PER_BAR = 0.0       # assume 0 risk-free rate

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  CINNAMON-Q THESIS BACKTEST")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# 1.  Load model
# ══════════════════════════════════════════════════════════════
print("\n[1/6] Loading trained model...")
model = RecurrentPPO.load(str(MODEL_PATH), device="cpu")
obs_shape    = model.observation_space.shape
action_shape = model.action_space.shape
print(f"  Model obs shape  : {obs_shape}")
print(f"  Model action shape: {action_shape}")

# ══════════════════════════════════════════════════════════════
# 2.  Prepare full-dataset memory-mapped arrays
# ══════════════════════════════════════════════════════════════
print("\n[2/6] Preparing data arrays (this takes ~2 min for 1.5M rows)...")
temp_dir = Path(tempfile.mkdtemp(prefix="backtest_"))
try:
    data_cfg = prepare_training_data(DATA_PATH, temp_dir, resample_freq='4h')
    n_steps  = data_cfg["n_steps"]
    n_assets = data_cfg["n_assets"]
    assets   = data_cfg["asset_list"]
    timestamps = data_cfg["timestamps"]
    print(f"  Total steps : {n_steps:,}")
    print(f"  Assets ({n_assets}): {assets}")

    # ── Train / test split ─────────────────────────────────
    split_at   = int(n_steps * (1 - TEST_FRACTION))
    n_test     = n_steps - split_at
    train_end  = pd.Timestamp(timestamps[split_at - 1])
    test_start = pd.Timestamp(timestamps[split_at])
    test_end   = pd.Timestamp(timestamps[-1])
    days_test  = (test_end - test_start).total_seconds() / 86400

    print(f"\n  Train period  : up to {train_end.date()}")
    print(f"  Test period   : {test_start.date()} -> {test_end.date()} ({days_test:.0f} days, {n_test:,} steps)")

    # ── Build test-only data_config (slice npy arrays) ────
    print("\n[3/6] Slicing arrays to test period...")
    test_dir = temp_dir / "test"
    test_dir.mkdir()

    def _slice_and_save(src_path: str, name: str) -> str:
        arr = np.load(src_path, mmap_mode="r")
        sliced = np.array(arr[split_at:], dtype=np.float32)
        out = str(test_dir / name)
        np.save(out, sliced)
        return out

    test_cfg = {
        "returns_path"   : _slice_and_save(data_cfg["returns_path"],    "returns_array.npy"),
        "volatility_path": _slice_and_save(data_cfg["volatility_path"], "volatility_array.npy"),
        "commit_path"    : _slice_and_save(data_cfg["commit_path"],     "commit_array.npy"),
        "utility_path"   : _slice_and_save(data_cfg["utility_path"],    "utility_array.npy"),
        "price_path"     : _slice_and_save(data_cfg["price_path"],      "price_array.npy"),
        "corr_path"      : _slice_and_save(data_cfg["corr_path"],       "corr_array.npy"),
        "garch_path"     : _slice_and_save(data_cfg["garch_path"],      "garch_array.npy"),
        "rsi_path"       : _slice_and_save(data_cfg["rsi_path"],        "rsi_array.npy"),
        "macd_path"      : _slice_and_save(data_cfg["macd_path"],       "macd_array.npy"),
        "bbw_path"       : _slice_and_save(data_cfg["bbw_path"],        "bbw_array.npy"),
        "atr_path"       : _slice_and_save(data_cfg["atr_path"],        "atr_array.npy"),
        "sma_ratio_path" : _slice_and_save(data_cfg["sma_ratio_path"],  "sma_ratio_array.npy"),
        "asset_list"     : assets,
        "timestamps"     : timestamps[split_at:],
        "n_steps"        : n_test,
        "n_assets"       : n_assets,
        "has_ta_features": data_cfg.get("has_ta_features", False),
        "has_correlation": data_cfg.get("has_correlation", False),
        "has_garch"      : data_cfg.get("has_garch", False),
        "has_macro"          : data_cfg.get("has_macro", False),
        "macro_path"         : _slice_and_save(data_cfg["macro_path"], "macro_array.npy") if data_cfg.get("has_macro") else data_cfg.get("macro_path", ""),
        # v7: multi-timeframe returns + crypto regime
        "has_mtf"            : data_cfg.get("has_mtf", False),
        "ret_4h_path"        : _slice_and_save(data_cfg["ret_4h_path"],        "ret_4h_array.npy")        if data_cfg.get("has_mtf") else "",
        "ret_1d_path"        : _slice_and_save(data_cfg["ret_1d_path"],        "ret_1d_array.npy")        if data_cfg.get("has_mtf") else "",
        "ret_7d_path"        : _slice_and_save(data_cfg["ret_7d_path"],        "ret_7d_array.npy")        if data_cfg.get("has_mtf") else "",
        "crypto_regime_path" : _slice_and_save(data_cfg["crypto_regime_path"], "crypto_regime_array.npy") if data_cfg.get("has_mtf") else "",
        # v26: Gate 3 (Regime Uncertainty Gate) is inference-only — activate it here
        "training_mode"      : False,
        # bars_per_day: needed for correct borrow rate and Calmar cooldown in fast_env.
        # Without this the env defaults to 96 (15-min), giving 16× wrong borrow rate.
        "bars_per_day"       : data_cfg.get("bars_per_day", 6),
    }
    print(f"  TA features : {test_cfg['has_ta_features']}")
    print(f"  Correlation : {test_cfg['has_correlation']}")
    print(f"  GARCH       : {test_cfg['has_garch']}")
    print(f"  Macro       : {test_cfg['has_macro']}")

    # ══════════════════════════════════════════════════════
    # 4.  Load VecNormalize with shape-matched dummy env
    # ══════════════════════════════════════════════════════
    print("\n[4/6] Loading VecNormalize...")

    class _ShapeEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
            self.action_space      = spaces.Box(0.0, 1.0, shape=action_shape, dtype=np.float32)
        def reset(self, **kw): return self.observation_space.sample(), {}
        def step(self, a):     return self.observation_space.sample(), 0.0, True, False, {}

    dummy     = DummyVecEnv([_ShapeEnv])
    norm_env  = VecNormalize.load(str(VECNORM_PATH), dummy)
    norm_env.training   = False
    norm_env.norm_reward = False
    print(f"  VecNorm obs shape: {norm_env.observation_space.shape}  (matches: {norm_env.observation_space.shape == obs_shape})")

    # ══════════════════════════════════════════════════════
    # 5.  Run backtest — walk the test env step by step
    # ══════════════════════════════════════════════════════
    print("\n[5/6] Running walk-forward backtest on test period...")

    test_env = FastCryptoEnv(data_config=test_cfg, initial_cash=INITIAL_CASH)
    raw_obs, _ = test_env.reset()
    # CRITICAL: Force start at step 0 for a proper walk-forward backtest.
    # reset() randomises current_step for training diversity; we override that here.
    test_env.current_step = 0
    raw_obs = test_env._get_observation()

    # ── Obs-shape compatibility shim ───────────────────────
    # v5 model was trained with 160-feature obs (included 7-bar volatility at slot[1]
    # of each asset block).  v6+ env produces 146 features (7-bar vol removed).
    # If there is a mismatch, reconstruct the 160-d obs by re-inserting volatility.
    env_obs_dim   = test_env.observation_space.shape[0]
    model_obs_dim = obs_shape[0]
    _needs_v5_pad = (model_obs_dim == 160 and env_obs_dim == 146)
    if _needs_v5_pad:
        print(f"  [!!] Obs-shape shim active: padding env({env_obs_dim}) -> model({model_obs_dim})")
        print(f"    Inserting 7-bar volatility at slot[1] of each asset block (v5 layout)")

    def _pad_obs_to_v5(obs_146: np.ndarray) -> np.ndarray:
        """Reconstruct 160-dim v5 obs by inserting 7-bar vol at slot[1] per asset."""
        obs_160 = np.zeros(160, dtype=np.float32)
        n  = test_env.n_assets
        v6 = 10   # features per asset in v6: [ret, commit, util, corr, garch, RSI, MACD, BBW, ATR, SMA]
        v5 = 11   # features per asset in v5: [ret, vol7, commit, util, corr, garch, RSI, MACD, BBW, ATR, SMA]
        vol_row = test_env.volatility_array[test_env.current_step]  # shape (n_assets,)
        for i in range(n):
            s6, s5 = i * v6, i * v5
            obs_160[s5]          = obs_146[s6]            # returns
            obs_160[s5 + 1]      = vol_row[i]             # 7-bar vol  (re-inserted)
            obs_160[s5+2:s5+11]  = obs_146[s6+1:s6+10]   # commit … SMA
        obs_160[n * v5:]  = obs_146[n * v6:]              # portfolio(2) + macro(4)
        return obs_160

    # Manual normalisation: norm_env expects a (1, obs_dim) batch
    def normalise(obs_1d):
        if _needs_v5_pad:
            obs_1d = _pad_obs_to_v5(obs_1d)
        batch = obs_1d.reshape(1, -1).astype(np.float32)
        return norm_env.normalize_obs(batch)

    lstm_state    = None
    ep_start      = np.array([True])
    done          = False

    portfolio_values = [INITIAL_CASH]
    step_returns     = []
    allocations      = []     # store per-step weight vectors for analysis
    step_timestamps  = [timestamps[split_at]]

    step = 0
    while not done:
        norm_obs = normalise(raw_obs)
        action, lstm_state = model.predict(
            norm_obs, state=lstm_state, episode_start=ep_start, deterministic=True
        )
        ep_start = np.array([False])

        raw_obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        # Portfolio value after this step
        pv = float(test_env.portfolio_value)
        portfolio_values.append(pv)
        step_returns.append(float(reward))
        allocations.append(test_env.current_weights.copy())

        step += 1
        if step < len(timestamps) - split_at:
            step_timestamps.append(timestamps[split_at + step])

        if step % 2000 == 0:
            elapsed_days = step * 15 / 60 / 24
            pnl = (pv / INITIAL_CASH - 1) * 100
            print(f"  Step {step:5d}/{n_test:,}  day {elapsed_days:5.0f}  PV=${pv:,.0f}  P&L={pnl:+.1f}%")

    print(f"  Backtest complete: {step} steps run.")

    # ══════════════════════════════════════════════════════
    # 6.  Benchmarks & metrics
    # ══════════════════════════════════════════════════════
    print("\n[6/6] Computing metrics and benchmarks...")

    # ── Benchmarks from pre-resampled test arrays ─────────
    # Loading raw 15-min parquet then comparing to 4h model steps would cause a
    # length mismatch (21k 15-min bars vs 1.3k 4h bars), so we use the
    # already-resampled returns that the model also used.
    _test_rets = np.load(test_cfg["returns_path"], mmap_mode="r")  # (n_test, n_assets)

    btc_idx    = assets.index("BTC")
    btc_returns = _test_rets[:, btc_idx]
    btc_equity  = INITIAL_CASH * np.cumprod(1 + btc_returns)
    btc_equity  = np.concatenate([[INITIAL_CASH], btc_equity])

    eq_returns = _test_rets.mean(axis=1)
    eq_equity  = INITIAL_CASH * np.cumprod(1 + eq_returns)
    eq_equity  = np.concatenate([[INITIAL_CASH], eq_equity])

    # ── Metric helper ─────────────────────────────────────
    def compute_metrics(equity_curve: np.ndarray, label: str) -> dict:
        n       = len(equity_curve) - 1
        rets    = np.diff(equity_curve) / equity_curve[:-1]
        total_r = equity_curve[-1] / equity_curve[0] - 1
        ann_r   = (1 + total_r) ** (BARS_PER_YEAR / max(n, 1)) - 1

        mean_r  = np.mean(rets)
        std_r   = np.std(rets) if np.std(rets) > 0 else 1e-9
        sharpe  = (mean_r - RISK_FREE_PER_BAR) / std_r * np.sqrt(BARS_PER_YEAR)

        down_r  = rets[rets < 0]
        down_std = np.std(down_r) if len(down_r) > 0 and np.std(down_r) > 0 else 1e-9
        sortino = (mean_r - RISK_FREE_PER_BAR) / down_std * np.sqrt(BARS_PER_YEAR)

        peak = np.maximum.accumulate(equity_curve)
        dd   = (equity_curve - peak) / peak
        max_dd = float(np.min(dd))

        calmar = ann_r / abs(max_dd) if abs(max_dd) > 1e-9 else np.nan
        win_rate = float(np.mean(rets > 0))

        return {
            "Strategy"          : label,
            "Total Return (%)"  : round(total_r * 100, 2),
            "Ann. Return (%)"   : round(ann_r * 100, 2),
            "Sharpe Ratio"      : round(sharpe, 3),
            "Sortino Ratio"     : round(sortino, 3),
            "Max Drawdown (%)"  : round(max_dd * 100, 2),
            "Calmar Ratio"      : round(calmar, 3) if not np.isnan(calmar) else "N/A",
            "Win Rate (%)"      : round(win_rate * 100, 1),
            "Final Value ($)"   : round(equity_curve[-1], 2),
        }

    # Use the shorter of model vs price_pivot length for fair comparison
    n_common = min(len(portfolio_values), len(btc_equity))
    pv_arr  = np.array(portfolio_values[:n_common])
    btc_arr = np.array(btc_equity[:n_common])
    eq_arr  = np.array(eq_equity[:n_common])

    results = [
        compute_metrics(pv_arr,  "LSTM PPO (Ours)"),
        compute_metrics(btc_arr, "Buy-and-Hold BTC"),
        compute_metrics(eq_arr,  "Equal-Weight (17 assets)"),
    ]

    perf_df = pd.DataFrame(results).set_index("Strategy")

    # ── Print table ───────────────────────────────────────
    print()
    print("=" * 65)
    print("  PERFORMANCE RESULTS  (test period: {} to {})".format(
        test_start.date(), test_end.date()))
    print("=" * 65)
    print(perf_df.to_string())
    print("=" * 65)

    # ── Save CSV ──────────────────────────────────────────
    perf_df.to_csv(REPORT_DIR / "performance_table.csv")
    print(f"\n  Saved: {REPORT_DIR / 'performance_table.csv'}")

    # ── Save equity curves ────────────────────────────────
    min_len = min(len(pv_arr), len(btc_arr), len(eq_arr))
    ts_index = [pd.Timestamp(t) for t in timestamps[split_at : split_at + min_len]]
    if len(ts_index) > min_len:
        ts_index = ts_index[:min_len]
    elif len(ts_index) < min_len:
        # pad with the last timestamp
        ts_index += [ts_index[-1]] * (min_len - len(ts_index))

    curves_df = pd.DataFrame({
        "LSTM_PPO"          : pv_arr[:min_len],
        "BTC_BuyHold"       : btc_arr[:min_len],
        "EqualWeight"       : eq_arr[:min_len],
    }, index=ts_index[:min_len])
    curves_df.index.name = "timestamp"
    curves_df.to_csv(REPORT_DIR / "portfolio_values.csv")
    print(f"  Saved: {REPORT_DIR / 'portfolio_values.csv'}")

    # ── Save summary text ─────────────────────────────────
    summary_lines = [
        "CINNAMON-Q BACKTEST SUMMARY",
        "=" * 65,
        f"Test period : {test_start.date()} to {test_end.date()} ({days_test:.0f} days)",
        f"Steps       : {n_test:,}  (4h bars)",
        f"Assets      : {n_assets} ({', '.join(assets)})",
        f"Model       : RecurrentPPO LSTM, 500k training steps, Gemini LLM utility scores",
        "",
        perf_df.to_string(),
        "",
        "KEY FINDINGS:",
    ]
    our_row = perf_df.loc["LSTM PPO (Ours)"]
    btc_row = perf_df.loc["Buy-and-Hold BTC"]
    eq_row  = perf_df.loc["Equal-Weight (17 assets)"]
    summary_lines += [
        f"  Our total return   : {our_row['Total Return (%)']:+.2f}%",
        f"  BTC total return   : {btc_row['Total Return (%)']:+.2f}%",
        f"  EqWt total return  : {eq_row['Total Return (%)']:+.2f}%",
        "",
        f"  Our Sortino Ratio  : {our_row['Sortino Ratio']}",
        f"  BTC Sortino Ratio  : {btc_row['Sortino Ratio']}",
        "",
        f"  Our Max Drawdown   : {our_row['Max Drawdown (%)']:.2f}%",
        f"  BTC Max Drawdown   : {btc_row['Max Drawdown (%)']:.2f}%",
    ]
    summary_text = "\n".join(summary_lines)
    with open(REPORT_DIR / "summary.txt", "w") as f:
        f.write(summary_text)
    print(f"  Saved: {REPORT_DIR / 'summary.txt'}")

    # ── Plots ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        ts_plot = curves_df.index

        # Equity curve
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(ts_plot, curves_df["LSTM_PPO"],    label="LSTM PPO (Ours)",        linewidth=2,   color="#2196F3")
        ax.plot(ts_plot, curves_df["BTC_BuyHold"], label="Buy-and-Hold BTC",        linewidth=1.5, color="#FF9800", linestyle="--")
        ax.plot(ts_plot, curves_df["EqualWeight"], label="Equal-Weight (17 assets)", linewidth=1.5, color="#4CAF50", linestyle=":")
        ax.axhline(INITIAL_CASH, color="gray", linewidth=0.8, linestyle="-", alpha=0.5, label="Initial capital")
        ax.set_title("Portfolio Equity Curve — Test Period (Out-of-Sample)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (USD)")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "equity_curve.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {REPORT_DIR / 'equity_curve.png'}")

        # Drawdown chart
        def _drawdown(equity):
            peak = np.maximum.accumulate(equity)
            return (equity - peak) / peak * 100

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(ts_plot, _drawdown(pv_arr[:min_len]),  0, alpha=0.5, color="#2196F3", label="LSTM PPO")
        ax.fill_between(ts_plot, _drawdown(btc_arr[:min_len]), 0, alpha=0.3, color="#FF9800", label="BTC Buy-Hold")
        ax.fill_between(ts_plot, _drawdown(eq_arr[:min_len]),  0, alpha=0.3, color="#4CAF50", label="Equal-Weight")
        ax.set_title("Drawdown — Test Period (Out-of-Sample)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc="lower left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "drawdown.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {REPORT_DIR / 'drawdown.png'}")

        # Normalised returns (growth of $1)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(ts_plot, pv_arr[:min_len]  / INITIAL_CASH, label="LSTM PPO (Ours)",        linewidth=2,   color="#2196F3")
        ax.plot(ts_plot, btc_arr[:min_len] / INITIAL_CASH, label="Buy-and-Hold BTC",        linewidth=1.5, color="#FF9800", linestyle="--")
        ax.plot(ts_plot, eq_arr[:min_len]  / INITIAL_CASH, label="Equal-Weight (17 assets)", linewidth=1.5, color="#4CAF50", linestyle=":")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)
        ax.set_title("Growth of $1 — Test Period (Out-of-Sample)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalised Value")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=30)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(REPORT_DIR / "growth_of_1.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {REPORT_DIR / 'growth_of_1.png'}")

    except ImportError:
        print("  (matplotlib not available - skipping plots)")
    except Exception as plot_err:
        print(f"  (plot error - {plot_err})")

finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

print("\n" + "=" * 65)
print("  BACKTEST COMPLETE")
print(f"  Results in: {REPORT_DIR}")
print("=" * 65)
