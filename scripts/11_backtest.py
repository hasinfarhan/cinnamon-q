"""
Walk-forward backtest of a trained model on the held-out test year, with benchmarks.

    python scripts/11_backtest.py                              # final model (seed 7)
    python scripts/11_backtest.py --model models/v50/seed_42   # second seed

The test window starts at the first bar after the model's last training bar (recorded in
holdout.json) and runs to the end of the data. Scaling ranges are the training-period ones.
The model acts once per 4-hour bar on information available at that bar; the rule-based
gates run in evaluation mode.

Benchmarks (computed on the same bar-aligned 4-hour returns, without transaction costs):
  * buy-and-hold BTC
  * equal weight over the listed coins, rebalanced every bar

Metrics: total and annualized return, Sharpe and Sortino ratios (annualized, risk-free
rate 0; Sortino uses the downside deviation around 0 over all bars, thesis formula [2]),
maximum drawdown from the all-time peak, Calmar ratio and the share of positive bars.

Output (reports/<name>/): performance_table.csv, portfolio_values.csv, diagnostics.csv,
summary.txt, equity_curve.png, drawdown.png.
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# SB3 zip members are file-like objects that some torch builds cannot read directly
if not getattr(torch.load, "_bytesio_patched", False):
    _orig_load = torch.load
    def _load(f, *a, **kw):
        if not isinstance(f, (str, bytes, io.BytesIO)) and hasattr(f, "read"):
            f = io.BytesIO(f.read())
        return _orig_load(f, *a, **kw)
    _load._bytesio_patched = True
    torch.load = _load

import gymnasium as gym                                       # noqa: E402
from gymnasium import spaces                                  # noqa: E402
from sb3_contrib import RecurrentPPO                          # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize   # noqa: E402

from hdrl.data import prepare_data, slice_data                # noqa: E402
from hdrl.env import TradingEnv                               # noqa: E402

INITIAL_CASH = 100_000.0
BARS_PER_YEAR = 365.25 * 6


def metrics(equity: np.ndarray, label: str) -> dict:
    n = len(equity) - 1
    rets = np.diff(equity) / equity[:-1]
    total = equity[-1] / equity[0] - 1
    ann = (1 + total) ** (BARS_PER_YEAR / max(n, 1)) - 1
    mean_r = np.mean(rets)
    std = np.std(rets) if np.std(rets) > 0 else 1e-9
    down = float(np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2)))
    down = down if down > 0 else 1e-9
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - peak) / peak))
    calmar = ann / abs(max_dd) if abs(max_dd) > 1e-9 else np.nan
    return {
        "Strategy": label,
        "Total Return (%)": round(total * 100, 2),
        "Ann. Return (%)": round(ann * 100, 2),
        "Sharpe Ratio": round(mean_r / std * np.sqrt(BARS_PER_YEAR), 3),
        "Sortino Ratio": round(mean_r / down * np.sqrt(BARS_PER_YEAR), 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Calmar Ratio": round(calmar, 3) if not np.isnan(calmar) else "N/A",
        "Win Rate (%)": round(float(np.mean(rets > 0)) * 100, 1),
        "Final Value ($)": round(equity[-1], 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "v50" / "seed_7"))
    ap.add_argument("--data", default=None, help="dataset path (default: the file named in holdout.json)")
    ap.add_argument("--out", default=None, help="report folder (default reports/test_year_<model folder>)")
    args = ap.parse_args()

    model_dir = Path(args.model).resolve()
    model_label = model_dir.relative_to(ROOT).as_posix() if model_dir.is_relative_to(ROOT) else model_dir.name
    holdout = json.loads((model_dir / "holdout.json").read_text())
    data_path = Path(args.data) if args.data else ROOT / "data" / "processed" / holdout["data_file"]
    out = Path(args.out) if args.out else ROOT / "reports" / f"test_year_{model_dir.name}"
    out.mkdir(parents=True, exist_ok=True)

    model = RecurrentPPO.load(str(model_dir / "final_model.zip"), device="cpu")
    obs_shape, act_shape = model.observation_space.shape, model.action_space.shape

    data = prepare_data(data_path, fit_end=holdout["train_end"], clip_pct=holdout["clip_pct"])
    split_at, end = int(holdout["split_at"]), data["n_steps"]
    ts = data["timestamps"]
    assert str(pd.Timestamp(ts[split_at - 1])) == holdout["train_end"]
    assert str(pd.Timestamp(ts[split_at])) == holdout["test_start"]
    test = slice_data(data, split_at, end)
    assets = data["asset_list"]
    print(f"Model trained on data up to {holdout['train_end']}; "
          f"test window {pd.Timestamp(ts[split_at])} -> {pd.Timestamp(ts[end - 1])} ({end - split_at:,} bars)")

    class _Shape(gym.Env):     # carries the spaces for VecNormalize.load
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
            self.action_space = spaces.Box(0.0, 1.0, shape=act_shape, dtype=np.float32)
        def reset(self, **kw): return self.observation_space.sample(), {}
        def step(self, a): return self.observation_space.sample(), 0.0, True, False, {}

    norm = VecNormalize.load(str(model_dir / "vec_normalize.pkl"), DummyVecEnv([_Shape]))
    norm.training, norm.norm_reward = False, False

    env = TradingEnv(test, training=False, initial_cash=INITIAL_CASH)
    env.reset()
    env.current_step = 0                       # walk forward from the first test bar
    obs = env._get_observation()

    state, start, done = None, np.array([True]), False
    values, diag = [INITIAL_CASH], []
    while not done:
        action, state = model.predict(norm.normalize_obs(obs.reshape(1, -1).astype(np.float32)),
                                      state=state, episode_start=start, deterministic=True)
        start = np.array([False])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        values.append(float(env.portfolio_value))
        if "long_exposure" in info:
            t = env.current_step
            w = np.asarray(env.prev_weights[:-1], dtype=np.float64)       # weights held over this bar
            prev = np.where(env.price[t - 1] > 1e-8, env.price[t - 1], 1.0)
            r = np.where(env.price[t - 1] > 1e-8, env.price[t] / prev - 1.0, 0.0)
            diag.append({
                "timestamp": str(env.timestamps[t - 1]),
                "net_return": info["portfolio_return"], "gross_return": info["gross_return"],
                "long_exposure": info["long_exposure"], "short_exposure": info["short_exposure"],
                "cash_weight": info["cash_weight"], "turnover": info["turnover"],
                "transaction_cost": info["transaction_cost"], "short_borrow_cost": info["short_borrow_cost"],
                "long_pnl": float(np.sum(np.maximum(w, 0.0) * r)),
                "short_pnl": float(np.sum(np.minimum(w, 0.0) * r)),
                "drawdown_vs_breaker_peak": info["current_drawdown"],
                "breaker_active": info["breaker_active"],
                "btc_trend": env.btc_trend(t - 1),
                **info["reward_components"], "reward": float(reward),
            })

    # Benchmarks on the same bars: the strategy's k-th return runs from bar k to bar k+1,
    # which is row k+1 of the return array (each row holds the return into that bar).
    rets = test["returns"]
    btc = INITIAL_CASH * np.concatenate([[1.0], np.cumprod(1 + rets[1:, assets.index("BTC")])])
    av = test["avail"]
    ew_r = (rets[1:] * av[:-1]).sum(axis=1) / np.maximum(av[:-1].sum(axis=1), 1)
    ew = INITIAL_CASH * np.concatenate([[1.0], np.cumprod(1 + ew_r)])

    n = min(len(values), len(btc))
    pv, btc, ew = np.array(values[:n]), btc[:n], ew[:n]
    table = pd.DataFrame([metrics(pv, "v50 (ours)"), metrics(btc, "Buy-and-Hold BTC"),
                          metrics(ew, "Equal-Weight (14 assets)")]).set_index("Strategy")
    print("\n" + table.to_string())

    table.to_csv(out / "performance_table.csv")
    idx = pd.DatetimeIndex([pd.Timestamp(t) for t in ts[split_at:split_at + n]], name="timestamp")
    curves = pd.DataFrame({"v50": pv, "BTC_BuyHold": btc, "EqualWeight": ew}, index=idx)
    curves.to_csv(out / "portfolio_values.csv")
    pd.DataFrame(diag).to_csv(out / "diagnostics.csv", index_label="step")
    (out / "summary.txt").write_text(
        f"Test window : {idx[0]} -> {idx[-1]} ({n - 1:,} four-hour returns)\n"
        f"Model       : {model_label}, trained on data up to {holdout['train_end']} (seed {holdout['seed']})\n"
        f"Data        : {data_path.name}\n"
        f"Benchmarks  : before transaction costs; the model pays 0.1% per unit of turnover "
        f"and 3% p.a. on short positions\n\n{table.to_string()}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(idx, pv, label="v50", lw=2)
        ax.plot(idx, btc, label="Buy-and-hold BTC", lw=1.3, ls="--")
        ax.plot(idx, ew, label="Equal weight (14 assets)", lw=1.3, ls=":")
        ax.axhline(INITIAL_CASH, color="gray", lw=0.8, alpha=0.5)
        ax.set_ylabel("Portfolio value (USD)"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        ax.set_title("Test year: portfolio value")
        fig.tight_layout(); fig.savefig(out / "equity_curve.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 4))
        for y, lab in [(pv, "v50"), (btc, "Buy-and-hold BTC"), (ew, "Equal weight")]:
            pk = np.maximum.accumulate(y)
            ax.plot(idx, (y - pk) / pk * 100, label=lab)
        ax.set_ylabel("Drawdown (%)"); ax.legend(loc="lower left"); ax.grid(alpha=0.3)
        ax.set_title("Test year: drawdown from the running peak")
        fig.tight_layout(); fig.savefig(out / "drawdown.png", dpi=150); plt.close(fig)
    except ImportError:
        pass
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
