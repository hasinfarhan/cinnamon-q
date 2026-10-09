"""
Brain 1 / Brain 2 trace. For each step of each window, records:
  Brain 1: the 128-d latent feeding the output layer, the raw (pre-clip) mean action,
           the clipped alpha, and the LSTM hidden state
  Brain 2: raw Sortino weights (inv(semicov) @ alpha), scaled pre-gate weights, final weights
  Ground truth: forward returns of every coin over 1 bar, 1 day (6 bars), 1 week (42 bars),
                and the BTC 200-day regime signal the reward uses
Saves one .npz per window for offline analysis.

Env vars: CQ_MODEL_DIR, CQ_DATA_PATH, CQ_WINDOWS ("name|start|end;..."), CQ_OUT_DIR
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

MODEL_DIR = Path(os.environ["CQ_MODEL_DIR"]); DATA_PATH = Path(os.environ["CQ_DATA_PATH"])
WINDOWS = [w.split("|") for w in os.environ["CQ_WINDOWS"].split(";")]
OUT_DIR = Path(os.environ["CQ_OUT_DIR"]); OUT_DIR.mkdir(parents=True, exist_ok=True)
HOLDOUT = json.loads((MODEL_DIR / "holdout.json").read_text()) if (MODEL_DIR / "holdout.json").exists() else None
FIT_END = HOLDOUT["train_end"] if HOLDOUT else None
N = 14

model = RecurrentPPO.load(str(MODEL_DIR / os.environ.get("CQ_MODEL_FILE", "final_model.zip")), device="cpu")   # v48: checkpoint choice
obs_shape, act_shape = model.observation_space.shape, model.action_space.shape
class _Shape(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=act_shape, dtype=np.float32)
    def reset(self, **kw): return self.observation_space.sample(), {}
    def step(self, a): return self.observation_space.sample(), 0.0, True, False, {}
norm_env = VecNormalize.load(str(MODEL_DIR / "vec_normalize.pkl"), DummyVecEnv([_Shape]))
norm_env.training, norm_env.norm_reward = False, False

captured = {}
model.policy.action_net.register_forward_hook(
    lambda mod, inp, out: captured.update(latent=inp[0].detach().numpy().reshape(-1).copy(),
                                          mean=out.detach().numpy().reshape(-1).copy()))
W_out = model.policy.action_net.weight.detach().numpy(); b_out = model.policy.action_net.bias.detach().numpy()
np.savez(OUT_DIR / "action_net.npz", W=W_out, b=b_out, log_std=model.policy.log_std.detach().numpy())

def raw_solver(env, alpha):
    """Replica of fast_env._construct_portfolio steps 1-2, over listed coins only (v46)."""
    t = env.current_step
    av = env._avail(); idx = np.flatnonzero(av); alpha = np.where(av, alpha, 0.0)
    hist = env.returns_array[max(0, t - env.bars_per_day * 20):t].astype(np.float64)[:, idx]
    if len(hist) < 10:
        return alpha / np.clip(env.volatility_array[t].astype(np.float64), 1e-8, None)
    d = np.minimum(hist, 0.0); sc = d.T @ d / len(hist)
    sc += max(1e-4, 0.10 * float(np.diag(sc).mean())) * np.eye(len(idx))
    raw = np.zeros(N); raw[idx] = np.linalg.inv(sc) @ alpha[idx]
    return raw

tmp = Path(tempfile.mkdtemp(prefix="diagbrain_"))
try:
    cfg = prepare_training_data(DATA_PATH, tmp, resample_freq="4h", fit_end=FIT_END, **((HOLDOUT or {}).get("loader_options") or {}))
    ts_all = pd.to_datetime(pd.Series(cfg["timestamps"]))
    price_full = np.load(cfg["price_path"], mmap_mode="r")
    for wname, ws, we in WINDOWS:
        s = int((ts_all >= pd.Timestamp(ws)).idxmax()); e = int((ts_all <= pd.Timestamp(we)).sum())
        wdir = tmp / wname; wdir.mkdir(); wcfg = {}
        for k, v in cfg.items():
            if isinstance(v, str) and v.endswith(".npy"):
                o = str(wdir / Path(v).name); np.save(o, np.array(np.load(v, mmap_mode="r")[s:e])); wcfg[k] = o
            else:
                wcfg[k] = v
        wcfg.update({"timestamps": cfg["timestamps"][s:e], "n_steps": e - s, "training_mode": False})
        if HOLDOUT and HOLDOUT.get("env_options"): wcfg.update(HOLDOUT["env_options"])
        env = FastCryptoEnv(data_config=wcfg, initial_cash=100_000.0)
        env.reset(); env.current_step = 0
        obs = env._get_observation(); state, start = None, np.array([True])
        rec = {k: [] for k in ["t", "latent", "mean", "alpha", "hidden", "raw", "pre", "banded", "final", "sig", "avail"]}
        while True:
            t = env.current_step
            action, state = model.predict(norm_env.normalize_obs(obs.reshape(1, -1).astype(np.float32)),
                                          state=state, episode_start=start, deterministic=True)
            start = np.array([False])
            _raw = np.asarray(action, dtype=np.float64).flatten()[:N]
            a = np.tanh(_raw) if getattr(env, "squash_actions", False) else np.clip(_raw, -1, 1)
            rec["t"].append(s + t); rec["latent"].append(captured["latent"]); rec["mean"].append(captured["mean"])
            rec["alpha"].append(a); rec["hidden"].append(np.asarray(state[0]).reshape(-1).copy())
            rec["raw"].append(raw_solver(env, a))
            full_pre = env._construct_portfolio(a.astype(np.float32))   # same squashed alpha env.step uses
            rec["pre"].append(full_pre[:N])
            rec["banded"].append(env._apply_no_trade_band(full_pre)[:N] if getattr(env, "no_trade_band", 0.0) > 0 else full_pre[:N])
            rec["sig"].append(float(env.crypto_regime_array[t, 3]) - 0.5)
            rec["avail"].append(env._avail().copy())   # v46: which coins were listed at this bar
            obs, _, term, trunc, info = env.step(action)
            rec["final"].append(np.asarray(env.current_weights[:N], dtype=np.float64))
            if term or trunc or "long_exposure" not in info:
                break
        T_idx = np.array(rec["t"])
        def fwd(h):
            j = np.minimum(T_idx + h, len(price_full) - 1)
            p0 = np.asarray(price_full[T_idx], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):   # unlisted coins (price 0) -> NaN
                return np.where(p0 > 0, np.asarray(price_full[j], dtype=np.float64) / p0 - 1.0, np.nan)
        np.savez(OUT_DIR / f"{wname}.npz", **{k: np.array(v) for k, v in rec.items()},
                 fwd_1bar=fwd(1), fwd_1day=fwd(6), fwd_1week=fwd(42),
                 timestamps=np.array([str(x) for x in ts_all.iloc[T_idx]]), assets=np.array(cfg["asset_list"]))
        print(f"[done] {wname}: {len(T_idx)} steps")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
