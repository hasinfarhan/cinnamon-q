"""
360-degree diagnostic of a trained model on chosen windows (no training, no reward).

For every step: Brain 1 conviction scores (alphas), the solver's raw weights before
scaling, the scaled pre-gate weights, the final weights, and which gate changed them.
Counterfactual: the same alphas pushed through the solver with symmetric 80/80 caps.
Input sensitivity: every 12th step, one input group is replaced with its value from a
random other step (LSTM state held fixed) and the change in alphas is measured.

Env vars: CQ_MODEL_DIR, CQ_DATA_PATH, CQ_WINDOWS ("name|start|end;name|start|end"), CQ_OUT
"""
import io, json, os, sys, tempfile, shutil
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "strategy" / "rl"))
import torch as _torch
if not getattr(_torch.load, "_bytesio_patched", False):
    _orig = _torch.load
    def _c(f, *a, **kw):
        if not isinstance(f, (str, bytes, io.BytesIO)) and hasattr(f, "read"):
            f = io.BytesIO(f.read())
        return _orig(f, *a, **kw)
    _c._bytesio_patched = True
    _torch.load = _c
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from strategy.rl.data_loader import prepare_training_data
from strategy.rl.fast_env import FastCryptoEnv

MODEL_DIR = Path(os.environ["CQ_MODEL_DIR"])
DATA_PATH = Path(os.environ["CQ_DATA_PATH"])
WINDOWS = [w.split("|") for w in os.environ["CQ_WINDOWS"].split(";")]
OUT = Path(os.environ["CQ_OUT"])
HOLDOUT = json.loads((MODEL_DIR / "holdout.json").read_text()) if (MODEL_DIR / "holdout.json").exists() else None
FIT_END = HOLDOUT["train_end"] if HOLDOUT else None

model = RecurrentPPO.load(str(MODEL_DIR / "final_model.zip"), device="cpu")
obs_shape, act_shape = model.observation_space.shape, model.action_space.shape

class _Shape(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=act_shape, dtype=np.float32)
    def reset(self, **kw): return self.observation_space.sample(), {}
    def step(self, a): return self.observation_space.sample(), 0.0, True, False, {}
norm_env = VecNormalize.load(str(MODEL_DIR / "vec_normalize.pkl"), DummyVecEnv([_Shape]))
norm_env.training, norm_env.norm_reward = False, False
normalise = lambda o: norm_env.normalize_obs(o.reshape(1, -1).astype(np.float32))

N = 14
PER = ["returns", "commit", "sentiment", "corr_btc", "volatility(ewma)", "RSI", "MACD", "BB_width",
       "ATR", "SMA_ratio(50h)", "ret_4h", "ret_1d", "ret_7d"]
GROUPS = {name: [a * 13 + k for a in range(N)] for k, name in enumerate(PER)}
GROUPS.update({"portfolio value+cash": [182, 183], "own positions": list(range(184, 198)),
               "macro (VIX,DXY,TNX,score)": list(range(198, 202)),
               "BTC 1d/7d + market 7d": [202, 203, 204], "BTC vs 200-day SMA": [205],
               "time of day/week/year": list(range(206, 212))})

def scale(raw_w, alpha, max_long, max_short):
    """Replica of fast_env._construct_portfolio step 3, with parametric caps."""
    conv = float(np.mean(np.abs(alpha)))
    longs, shorts = np.clip(raw_w, 0, None), np.clip(raw_w, None, 0)
    ls, ss = longs.sum(), np.abs(shorts).sum()
    if ls > 1e-8: longs = longs / ls * (max_long * conv)
    if ss > 1e-8: shorts = shorts / ss * (max_short * conv)
    return np.clip(longs, 0, 0.10) + np.clip(shorts, -0.10, 0)

def raw_solver(env, alpha):
    """Replica of fast_env._construct_portfolio steps 1-2 (raw Sortino weights)."""
    t = env.current_step
    hist = env.returns_array[max(0, t - env.bars_per_day * 20):t].astype(np.float64)
    if len(hist) < 10:
        return alpha / np.clip(env.volatility_array[t].astype(np.float64), 1e-8, None)
    d = np.minimum(hist, 0.0); sc = d.T @ d / len(hist)
    sc += max(1e-4, 0.10 * float(np.diag(sc).mean())) * np.eye(N)
    return np.linalg.inv(sc) @ alpha

tmp = Path(tempfile.mkdtemp(prefix="diag360_"))
results = {"model": MODEL_DIR.name, "train_end": FIT_END, "windows": {}}
try:
    cfg = prepare_training_data(DATA_PATH, tmp, resample_freq="4h", fit_end=FIT_END, **((HOLDOUT or {}).get("loader_options") or {}))
    ts_all = pd.to_datetime(pd.Series(cfg["timestamps"]))
    assets = cfg["asset_list"]
    for wname, ws, we in WINDOWS:
        s = int((ts_all >= pd.Timestamp(ws)).idxmax()); e = int((ts_all <= pd.Timestamp(we)).sum())
        wdir = tmp / wname; wdir.mkdir()
        wcfg = {}
        for k, v in cfg.items():
            if isinstance(v, str) and v.endswith(".npy"):
                out = str(wdir / Path(v).name); np.save(out, np.array(np.load(v, mmap_mode="r")[s:e])); wcfg[k] = out
            else:
                wcfg[k] = v
        wcfg.update({"timestamps": cfg["timestamps"][s:e], "n_steps": e - s, "training_mode": False})
        if HOLDOUT and HOLDOUT.get("env_options"): wcfg.update(HOLDOUT["env_options"])
        env = FastCryptoEnv(data_config=wcfg, initial_cash=100_000.0)
        env.reset(); env.current_step = 0
        obs = env._get_observation()
        state, start = None, np.array([True])
        rec = {k: [] for k in ["alpha", "raw", "pre", "final", "sym", "obs", "state", "sig", "gate"]}
        replica_err = 0.0
        while True:
            rec["obs"].append(obs.copy()); rec["state"].append(state)
            action, state = model.predict(normalise(obs), state=state, episode_start=start, deterministic=True)
            start = np.array([False])
            _raw = np.asarray(action, dtype=np.float64).flatten()[:N]
            a = np.tanh(_raw) if getattr(env, "squash_actions", False) else np.clip(_raw, -1, 1)
            full_pre = env._construct_portfolio(np.asarray(action, dtype=np.float32).flatten()[:N])
            raw = raw_solver(env, a); pre = full_pre[:N].astype(np.float64)
            replica_err = max(replica_err, float(np.abs(scale(raw, a, 0.80, 0.60) - pre).max()))
            # v44 no-trade band (if enabled) acts between the solver and the gates
            banded = (env._apply_no_trade_band(full_pre)[:N].astype(np.float64)
                      if getattr(env, "no_trade_band", 0.0) > 0 else pre)
            sig = float(env.crypto_regime_array[env.current_step, 3]) - 0.5
            obs, _, term, trunc, info = env.step(action)
            fin = np.asarray(env.current_weights[:N], dtype=np.float64)
            gate = "none"
            if np.abs(fin - banded).max() > 1e-6:
                gate = ("calmar" if env.calmar_gate_active else
                        "regime_band" if abs(sig) <= 0.025 else "macro_or_band")
            for k, v in zip(["alpha", "raw", "pre", "final", "sym", "sig", "gate", "banded"],
                            [a, raw, pre, fin, scale(raw, a, 0.80, 0.80), sig, gate, banded]):
                rec.setdefault(k, []).append(v)
            if term or trunc or "long_exposure" not in info:
                break
        A, R, P, F, Y = (np.array(rec[k]) for k in ["alpha", "raw", "pre", "final", "sym"])
        sig = np.array(rec["sig"]); bear = sig < -0.025; T = len(A)
        net = lambda W: W.sum(axis=1)
        mixed = ((R > 0).any(1) & (R < 0).any(1))
        w = {"steps": T, "solver_replica_max_err": replica_err,
             "pct_bars_bear_signal": float(bear.mean() * 100),
             "alpha_pct_negative_all": float((A < 0).mean() * 100),
             "alpha_pct_negative_in_bear": float((A[bear] < 0).mean() * 100) if bear.any() else None,
             "alpha_mean_in_bear": float(A[bear].mean()) if bear.any() else None,
             "alpha_mean_in_bull": float(A[sig > 0.025].mean()) if (sig > 0.025).any() else None,
             "mean_abs_alpha": float(np.abs(A).mean()),
             "pct_steps_majority_alphas_negative": float(((A < 0).sum(1) > N / 2).mean() * 100),
             "pct_steps_all_alphas_negative": float((A < 0).all(1).mean() * 100),
             "pct_steps_raw_weights_mixed_sign": float(mixed.mean() * 100),
             "pct_steps_raw_all_negative": float((R <= 0).all(1).mean() * 100),
             "pct_steps_alpha_neg_but_raw_pos": float(((A < 0) & (R > 0)).mean() * 100),
             "pct_net_short_pre_gate": float((net(P) < 0).mean() * 100),
             "pct_net_short_final": float((net(F) < 0).mean() * 100),
             "pct_net_short_if_symmetric_80_80_caps": float((net(Y) < 0).mean() * 100),
             "pct_net_short_if_symmetric_in_bear": float((net(Y)[bear] < 0).mean() * 100) if bear.any() else None,
             "mean_net_pre_gate": float(net(P).mean()), "mean_net_final": float(net(F).mean()),
             "gate_changes": {g: float((np.array(rec["gate"]) == g).mean() * 100) for g in ["calmar", "regime_band", "macro_or_band"]},
             "pct_coin_steps_band_held": float((np.abs(np.array(rec["banded"]) - P) > 1e-6).mean() * 100),
             "per_asset_pct_negative_alpha": {assets[i]: float((A[:, i] < 0).mean() * 100) for i in range(N)}}
        # input sensitivity: replace one group with another step's values, LSTM state fixed
        rng = np.random.default_rng(0); O = np.array(rec["obs"]); sens = {g: [] for g in GROUPS}
        sens_net = {g: [] for g in GROUPS}; check = 0.0
        # v45+: the env squashes actions with tanh; older models are clipped
        _sq = (lambda x: np.tanh(x)) if getattr(env, "squash_actions", False) else (lambda x: np.clip(x, -1, 1))
        for t in range(12, T, 12):
            st, base_a = rec["state"][t], A[t]
            a0, _ = model.predict(normalise(O[t]), state=st, episode_start=np.array([False]), deterministic=True)
            check = max(check, float(np.abs(_sq(np.asarray(a0).flatten()[:N]) - base_a).max()))
            for g, idx in GROUPS.items():
                j = int(rng.integers(0, T)); o = O[t].copy(); o[idx] = O[j][idx]
                ap, _ = model.predict(normalise(o), state=st, episode_start=np.array([False]), deterministic=True)
                ap = _sq(np.asarray(ap).flatten()[:N])
                sens[g].append(float(np.abs(ap - base_a).mean())); sens_net[g].append(float(abs(ap.mean() - base_a.mean())))
        w["sensitivity_replay_check_max_err"] = check
        w["input_sensitivity_mean_abs_alpha_change"] = dict(sorted(((g, float(np.mean(v))) for g, v in sens.items()), key=lambda x: -x[1]))
        w["input_sensitivity_net_alpha_change"] = {g: float(np.mean(sens_net[g])) for g in w["input_sensitivity_mean_abs_alpha_change"]}
        results["windows"][wname] = w
        print(f"[done] {wname}: {T} steps")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
OUT.write_text(json.dumps(results, indent=2))
print(f"saved {OUT}")
