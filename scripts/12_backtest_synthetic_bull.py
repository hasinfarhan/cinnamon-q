"""
Synthetic Bull Market Backtest
===============================
Takes the real bear-market test period and transforms every feature into
its bull-market equivalent, then runs the model through it.

Why this exists:
  The held-out test window (Jun 2025 - Jun 2026) is a -42% bear market.
  This script creates a coherent synthetic version of the same period
  where BTC rises +42%, timestamps shift forward one year (Jul 2026 -
  Jun 2027), and every directional feature is flipped to match.

Transformations applied per feature type:
  returns / MTF returns       x -1        (falling -> rising prices)
  prices                      reconstructed from inverted returns
  RSI                         100 - RSI   (oversold -> overbought)
  MACD                        x -1        (negative -> positive momentum)
  SMA ratio (price/SMA)       2 - ratio   (below SMA -> above SMA)
  BBW / ATR / GARCH / vol     unchanged   (same volatility magnitude)
  corr_btc                    unchanged   (correlation structure same)
  utility_score               1 - score   (low utility -> high utility)
  commit_array                unchanged   (on-chain activity not directional)
  crypto regime BTC returns   x -1        (negative -> positive returns)
  SMA200_norm                 1 - value   (below 200d -> above 200d)
  VIX                         mirrored around its mean (fear -> calm)
  macro_trend_score           1 - score   (bearish macro -> bullish macro)
  timestamps                  +366 days   (Jun 2025-Jun 2026 -> Jul 2026-Jun 2027)

Usage:
  python scripts/12_backtest_synthetic_bull.py
  python scripts/12_backtest_synthetic_bull.py --model v41  # override version
"""

import io
import os
import sys
import shutil
import tempfile
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "strategy" / "rl"))

# ── PyTorch compatibility fix ────────────────────────────────────────────────
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

# ── CLI override ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--model", default="v40", help="model version string, e.g. v40")
_args, _ = parser.parse_known_args()
VERSION = _args.model

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = project_root / "models" / f"hdrl_worldclass_{VERSION}" / "final_model.zip"
VECNORM_PATH = project_root / "models" / f"hdrl_worldclass_{VERSION}" / "vec_normalize.pkl"
_v5          = project_root / "data" / "processed" / "crypto_deep_v5.parquet"
DATA_PATH    = _v5 if _v5.exists() else project_root / "data" / "processed" / "crypto_deep_v4_sentiment.parquet"
REPORT_DIR   = project_root / "reports" / f"synthetic_bull_{VERSION}"
INITIAL_CASH  = 100_000.0
TEST_FRACTION = 0.20
BARS_PER_YEAR = 365.25 * 6
TIMESTAMP_SHIFT_DAYS = 366   # shift Jul 2026 -> Jun 2027

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print(f"  CINNAMON-Q SYNTHETIC BULL BACKTEST  [{VERSION}]")
print("=" * 65)
print(f"  Model     : {MODEL_PATH}")
print(f"  Report dir: {REPORT_DIR}")


# ════════════════════════════════════════════════════════════════════════════
# Helper: apply bull transformations to all arrays
# ════════════════════════════════════════════════════════════════════════════
def make_bull(arr: np.ndarray, kind: str) -> np.ndarray:
    """Return a bull-market version of arr based on the feature kind."""
    a = arr.astype(np.float32).copy()
    if kind == "returns":
        return -a
    elif kind == "rsi":
        return np.clip(100.0 - a, 0.0, 100.0)
    elif kind == "macd":
        return -a
    elif kind == "sma_ratio":
        # price/SMA ratio: bear < 1.0, bull > 1.0 → mirror around 1.0
        return np.clip(2.0 - a, 0.5, 2.0)
    elif kind == "utility":
        return np.clip(1.0 - a, 0.0, 1.0)
    elif kind == "crypto_regime":
        # cols: [BTC_1d, BTC_7d, Mkt_7d, SMA200_norm]
        out = a.copy()
        out[:, 0] = np.clip(-a[:, 0], -1.0, 1.0)   # BTC_1d return  → invert
        out[:, 1] = np.clip(-a[:, 1], -1.0, 1.0)   # BTC_7d return  → invert
        out[:, 2] = np.clip(-a[:, 2], -1.0, 1.0)   # Mkt_7d return  → invert
        out[:, 3] = np.clip(1.0 - a[:, 3], 0.0, 1.0)  # SMA200_norm → above SMA
        return out
    elif kind == "macro":
        # cols: [VIX, DXY, TNX, macro_trend_score]
        out = a.copy()
        vix_mean = float(np.mean(a[:, 0]))
        out[:, 0] = np.clip(2.0 * vix_mean - a[:, 0], 0.0, None)  # VIX → calmer
        # DXY (col 1) and TNX (col 2): keep — complex macro relationship
        out[:, 3] = np.clip(1.0 - a[:, 3], 0.0, 1.0)   # macro_score → bullish
        return out
    elif kind == "prices":
        # Reconstruct rising price series from inverted returns
        # We use the original prices as starting point
        pass  # handled separately below
    else:
        # BBW, ATR, GARCH, volatility, corr, commit → unchanged
        return a


def reconstruct_bull_prices(prices: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Reconstruct a synthetic bull price series from inverted returns."""
    bull_ret = -returns.astype(np.float32)
    bull_prices = np.zeros_like(prices, dtype=np.float32)
    bull_prices[0] = prices[0]
    for t in range(1, len(bull_prices)):
        bull_prices[t] = bull_prices[t - 1] * (1.0 + bull_ret[t])
    return bull_prices


# ════════════════════════════════════════════════════════════════════════════
# 1. Load model
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading trained model...")
model        = RecurrentPPO.load(str(MODEL_PATH), device="cpu")
obs_shape    = model.observation_space.shape
action_shape = model.action_space.shape
print(f"  Model obs shape   : {obs_shape}")
print(f"  Model action shape: {action_shape}")

# ════════════════════════════════════════════════════════════════════════════
# 2. Prepare real data arrays then apply bull transformations
# ════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Preparing data and applying bull transformations...")
temp_dir = Path(tempfile.mkdtemp(prefix="synth_bull_"))
try:
    data_cfg   = prepare_training_data(DATA_PATH, temp_dir, resample_freq='4h')
    n_steps    = data_cfg["n_steps"]
    n_assets   = data_cfg["n_assets"]
    assets     = data_cfg["asset_list"]
    timestamps = data_cfg["timestamps"]

    split_at  = int(n_steps * (1 - TEST_FRACTION))
    n_test    = n_steps - split_at
    real_start = pd.Timestamp(timestamps[split_at])
    real_end   = pd.Timestamp(timestamps[-1])
    days_test  = (real_end - real_start).total_seconds() / 86400

    # Shift timestamps forward by one year
    synth_timestamps = np.array([
        (pd.Timestamp(ts) + pd.Timedelta(days=TIMESTAMP_SHIFT_DAYS)).isoformat()
        for ts in timestamps[split_at:]
    ])
    synth_start = pd.Timestamp(str(synth_timestamps[0]))
    synth_end   = pd.Timestamp(str(synth_timestamps[-1]))

    print(f"  Real test period   : {real_start.date()} -> {real_end.date()} ({days_test:.0f} days)")
    print(f"  Synthetic period   : {synth_start.date()} -> {synth_end.date()} (shifted +{TIMESTAMP_SHIFT_DAYS}d)")

    bull_dir = temp_dir / "bull"
    bull_dir.mkdir()

    def _transform_and_save(src_path: str, name: str, kind: str) -> str:
        arr    = np.array(np.load(src_path, mmap_mode="r")[split_at:], dtype=np.float32)
        transformed = make_bull(arr, kind)
        out = str(bull_dir / name)
        np.save(out, transformed)
        return out

    # Special case: prices need the returns array first
    _returns_orig = np.array(np.load(data_cfg["returns_path"], mmap_mode="r")[split_at:], dtype=np.float32)
    _prices_orig  = np.array(np.load(data_cfg["price_path"],   mmap_mode="r")[split_at:], dtype=np.float32)
    _bull_prices  = reconstruct_bull_prices(_prices_orig, _returns_orig)
    _bull_returns = -_returns_orig
    np.save(str(bull_dir / "returns_array.npy"), _bull_returns)
    np.save(str(bull_dir / "price_array.npy"),   _bull_prices)

    btc_idx = assets.index("BTC")
    btc_real_ret   = float(np.prod(1 + _returns_orig[:, btc_idx]) - 1) * 100
    btc_synth_ret  = float(np.prod(1 + _bull_returns[:, btc_idx]) - 1) * 100
    print(f"  BTC real return    : {btc_real_ret:+.1f}%")
    print(f"  BTC synthetic bull : {btc_synth_ret:+.1f}%")

    test_cfg = {
        "returns_path"   : str(bull_dir / "returns_array.npy"),
        "price_path"     : str(bull_dir / "price_array.npy"),
        "volatility_path": _transform_and_save(data_cfg["volatility_path"], "volatility_array.npy", "unchanged"),
        "commit_path"    : _transform_and_save(data_cfg["commit_path"],     "commit_array.npy",     "unchanged"),
        "utility_path"   : _transform_and_save(data_cfg["utility_path"],    "utility_array.npy",    "utility"),
        "corr_path"      : _transform_and_save(data_cfg["corr_path"],       "corr_array.npy",       "unchanged"),
        "garch_path"     : _transform_and_save(data_cfg["garch_path"],      "garch_array.npy",      "unchanged"),
        "rsi_path"       : _transform_and_save(data_cfg["rsi_path"],        "rsi_array.npy",        "rsi"),
        "macd_path"      : _transform_and_save(data_cfg["macd_path"],       "macd_array.npy",       "macd"),
        "bbw_path"       : _transform_and_save(data_cfg["bbw_path"],        "bbw_array.npy",        "unchanged"),
        "atr_path"       : _transform_and_save(data_cfg["atr_path"],        "atr_array.npy",        "unchanged"),
        "sma_ratio_path" : _transform_and_save(data_cfg["sma_ratio_path"],  "sma_ratio_array.npy",  "sma_ratio"),
        "asset_list"     : assets,
        "timestamps"     : synth_timestamps,
        "n_steps"        : n_test,
        "n_assets"       : n_assets,
        "has_ta_features": data_cfg.get("has_ta_features", False),
        "has_correlation": data_cfg.get("has_correlation", False),
        "has_garch"      : data_cfg.get("has_garch", False),
        "has_macro"      : data_cfg.get("has_macro", False),
        "macro_path"     : _transform_and_save(data_cfg["macro_path"], "macro_array.npy", "macro")
                           if data_cfg.get("has_macro") else data_cfg.get("macro_path", ""),
        "has_mtf"        : data_cfg.get("has_mtf", False),
        "training_mode"  : False,
        "bars_per_day"   : data_cfg.get("bars_per_day", 6),
    }
    if data_cfg.get("has_mtf"):
        test_cfg["ret_4h_path"]        = _transform_and_save(data_cfg["ret_4h_path"],        "ret_4h_array.npy",        "returns")
        test_cfg["ret_1d_path"]        = _transform_and_save(data_cfg["ret_1d_path"],        "ret_1d_array.npy",        "returns")
        test_cfg["ret_7d_path"]        = _transform_and_save(data_cfg["ret_7d_path"],        "ret_7d_array.npy",        "returns")
        test_cfg["crypto_regime_path"] = _transform_and_save(data_cfg["crypto_regime_path"], "crypto_regime_array.npy", "crypto_regime")
    else:
        test_cfg["ret_4h_path"] = test_cfg["ret_1d_path"] = test_cfg["ret_7d_path"] = test_cfg["crypto_regime_path"] = ""

    print("\n[3/6] Bull feature transformations applied:")
    print("  returns/MTF    : x-1 (falling->rising)")
    print("  RSI            : 100-RSI (oversold->overbought)")
    print("  MACD           : x-1 (negative->positive momentum)")
    print("  SMA ratio      : 2-ratio (below->above SMA)")
    print("  utility_score  : 1-score (low->high utility)")
    print("  SMA200_norm    : 1-value (below->above 200d MA)")
    print("  VIX            : mirrored around mean (fear->calm)")
    print("  macro_score    : 1-score (bearish->bullish macro)")
    print("  timestamps     : +366 days")
    print("  BBW/ATR/GARCH/ : unchanged (same volatility structure)")
    print("  corr_btc       : unchanged (correlation structure same)")
    print("  commit         : unchanged (on-chain not directional)")

    # ════════════════════════════════════════════════════════════════════════
    # 4. Load VecNormalize
    # ════════════════════════════════════════════════════════════════════════
    print("\n[4/6] Loading VecNormalize...")

    class _ShapeEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
            self.action_space      = spaces.Box(-1., 1., shape=action_shape, dtype=np.float32)
        def reset(self, **kw): return self.observation_space.sample(), {}
        def step(self, a):     return self.observation_space.sample(), 0., True, False, {}

    dummy    = DummyVecEnv([_ShapeEnv])
    norm_env = VecNormalize.load(str(VECNORM_PATH), dummy)
    norm_env.training    = False
    norm_env.norm_reward = False
    print(f"  VecNorm obs shape: {norm_env.observation_space.shape}  (matches: {norm_env.observation_space.shape == obs_shape})")

    def normalise(obs_1d):
        return norm_env.normalize_obs(obs_1d.reshape(1, -1).astype(np.float32))

    # ════════════════════════════════════════════════════════════════════════
    # 5. Run walk-forward backtest
    # ════════════════════════════════════════════════════════════════════════
    print("\n[5/6] Running synthetic bull walk-forward backtest...")

    test_env = FastCryptoEnv(data_config=test_cfg, initial_cash=INITIAL_CASH)
    raw_obs, _ = test_env.reset()
    test_env.current_step = 0
    raw_obs = test_env._get_observation()

    lstm_state       = None
    ep_start         = np.array([True])
    done             = False
    portfolio_values = [INITIAL_CASH]
    step_returns     = []
    allocations      = []
    step_timestamps  = [synth_timestamps[0]]

    step = 0
    while not done:
        norm_obs = normalise(raw_obs)
        action, lstm_state = model.predict(
            norm_obs, state=lstm_state, episode_start=ep_start, deterministic=True)
        ep_start = np.array([False])

        raw_obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        pv = float(test_env.portfolio_value)
        portfolio_values.append(pv)
        step_returns.append(float(reward))
        allocations.append(test_env.current_weights.copy())

        step += 1
        if step < len(synth_timestamps):
            step_timestamps.append(synth_timestamps[step])

        if step % 2000 == 0:
            pnl = (pv / INITIAL_CASH - 1) * 100
            print(f"  Step {step:5d}/{n_test:,}  PV=${pv:,.0f}  P&L={pnl:+.1f}%")

    print(f"  Backtest complete: {step} steps run.")

    # ════════════════════════════════════════════════════════════════════════
    # 6. Metrics
    # ════════════════════════════════════════════════════════════════════════
    print("\n[6/6] Computing metrics...")

    bull_rets  = _bull_returns
    btc_equity = INITIAL_CASH * np.cumprod(1 + bull_rets[:, btc_idx])
    btc_equity = np.concatenate([[INITIAL_CASH], btc_equity])
    eq_equity  = INITIAL_CASH * np.cumprod(1 + bull_rets.mean(axis=1))
    eq_equity  = np.concatenate([[INITIAL_CASH], eq_equity])

    def compute_metrics(equity_curve: np.ndarray, label: str) -> dict:
        n    = len(equity_curve) - 1
        rets = np.diff(equity_curve) / equity_curve[:-1]
        ann_factor = BARS_PER_YEAR / n

        total_ret = (equity_curve[-1] / equity_curve[0] - 1) * 100
        ann_ret   = ((equity_curve[-1] / equity_curve[0]) ** ann_factor - 1) * 100
        excess    = rets - RISK_FREE_PER_BAR
        sharpe    = float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(BARS_PER_YEAR))
        downside  = rets[rets < 0]
        sortino   = float(np.mean(excess) / (np.std(downside) + 1e-9) * np.sqrt(BARS_PER_YEAR))
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdowns   = (equity_curve - rolling_max) / rolling_max
        max_dd      = float(drawdowns.min()) * 100
        calmar      = ann_ret / (abs(max_dd) + 1e-9)
        win_rate    = float(np.mean(rets > 0)) * 100
        return {
            "Total Return (%)": round(total_ret, 2),
            "Ann. Return (%)":  round(ann_ret, 2),
            "Sharpe Ratio":     round(sharpe, 3),
            "Sortino Ratio":    round(sortino, 3),
            "Max Drawdown (%)": round(max_dd, 2),
            "Calmar Ratio":     round(calmar, 3),
            "Win Rate (%)":     round(win_rate, 1),
            "Final Value ($)":  round(equity_curve[-1], 2),
        }

    RISK_FREE_PER_BAR = 0.0
    results = {
        f"LSTM PPO [{VERSION}]":       compute_metrics(np.array(portfolio_values), "model"),
        "Buy-and-Hold BTC (bull)":     compute_metrics(btc_equity, "btc"),
        "Equal-Weight (bull)":         compute_metrics(eq_equity, "eq"),
    }
    df = pd.DataFrame(results).T

    print("\n" + "=" * 65)
    print(f"  SYNTHETIC BULL RESULTS  ({synth_start.date()} to {synth_end.date()})")
    print(f"  [Derived from real bear period: {real_start.date()} to {real_end.date()}]")
    print("=" * 65)
    print(df.to_string())
    print("=" * 65)

    # Save
    df.to_csv(REPORT_DIR / "performance_table.csv")

    pv_df = pd.DataFrame({
        "timestamp":        step_timestamps[:len(portfolio_values)],
        "portfolio_value":  portfolio_values,
        "btc_equity":       btc_equity[:len(portfolio_values)],
    })
    pv_df.to_csv(REPORT_DIR / "portfolio_values.csv", index=False)

    # Equity curve plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ts = pd.to_datetime(step_timestamps[:len(portfolio_values)])
        axes[0].plot(ts, portfolio_values, label=f"LSTM PPO [{VERSION}]", linewidth=2)
        axes[0].plot(ts, btc_equity[:len(ts)], label="BTC buy-and-hold (bull)", linewidth=1.5, alpha=0.8)
        axes[0].plot(ts, eq_equity[:len(ts)], label="Equal-weight (bull)", linewidth=1.2, alpha=0.6)
        axes[0].set_title(f"Synthetic Bull Market Equity Curve [{VERSION}]")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        pv_arr = np.array(portfolio_values)
        dd = (pv_arr - np.maximum.accumulate(pv_arr)) / np.maximum.accumulate(pv_arr) * 100
        axes[1].fill_between(ts, dd, 0, alpha=0.4, color="red")
        axes[1].set_title("Drawdown (%)")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(REPORT_DIR / "equity_curve.png", dpi=150)
        plt.close()
        print(f"  Saved: {REPORT_DIR / 'equity_curve.png'}")
    except Exception as e:
        print(f"  [plot skipped: {e}]")

    print(f"  Saved: {REPORT_DIR / 'performance_table.csv'}")
    print(f"  Saved: {REPORT_DIR / 'portfolio_values.csv'}")
    print(f"\n  Reports in: {REPORT_DIR}")
    print("\n" + "=" * 65)
    print("  SYNTHETIC BULL BACKTEST COMPLETE")
    print("=" * 65)

finally:
    shutil.rmtree(temp_dir, ignore_errors=True)
