"""
Reward ranking check (training data only).

Runs fixed strategies and variants of a trained model through the environment in TRAINING
mode over the training period, recording per bar: net return r, drawdown, the trend signal s,
the mean position and mean signal over listed coins, and the env's own reward. Any reward
formula built from these can then be scored offline, and its ranking of the strategies
compared with their money results.

Env vars: CQ_MODEL_DIR, CQ_DATA_PATH, CQ_OUT (.parquet), CQ_LABEL (prefix for model policies)
"""
import io, json, os, sys, tempfile, shutil
from pathlib import Path
import numpy as np, pandas as pd
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src")); sys.path.insert(0, str(project_root / "src" / "strategy" / "rl"))
import torch as _torch
_orig = _torch.load
def _c(f, *a, **kw):
    if not isinstance(f, (str, bytes, io.BytesIO)) and hasattr(f, "read"):
        f = io.BytesIO(f.read())
    return _orig(f, *a, **kw)
_torch.load = _c
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from strategy.rl.data_loader import prepare_training_data
from strategy.rl.fast_env import FastCryptoEnv

MD = Path(os.environ["CQ_MODEL_DIR"]); HO = json.loads((MD / "holdout.json").read_text()); N = 14
LABEL = os.environ.get("CQ_LABEL", MD.name); ONLY_MODEL = os.environ.get("CQ_ONLY_MODEL") == "1"
model = RecurrentPPO.load(str(MD / "final_model.zip"), device="cpu")
class _S(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=model.observation_space.shape, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=model.action_space.shape, dtype=np.float32)
    def reset(self, **kw): return self.observation_space.sample(), {}
    def step(self, a): return self.observation_space.sample(), 0.0, True, False, {}
nv = VecNormalize.load(str(MD / "vec_normalize.pkl"), DummyVecEnv([_S])); nv.training, nv.norm_reward = False, False

tmp = Path(tempfile.mkdtemp(prefix="rank_")); frames = []
try:
    cfg = prepare_training_data(Path(os.environ["CQ_DATA_PATH"]), tmp, resample_freq="4h", fit_end=HO["train_end"],
                                **(HO.get("loader_options") or {}))
    e = HO["split_at"]
    for k, v in list(cfg.items()):
        if isinstance(v, str) and v.endswith(".npy"):
            o = str(tmp / ("tr_" + Path(v).name)); np.save(o, np.array(np.load(v, mmap_mode="r")[:e])); cfg[k] = o
    cfg.update({"timestamps": cfg["timestamps"][:e], "n_steps": e, "training_mode": True}); cfg.update(HO["env_options"])
    cfg["reward_mode"] = "v40"   # record the v40 reward as a cross-check of the offline formula
    ts = pd.to_datetime(pd.Series(cfg["timestamps"])); assert ts.iloc[-1] <= pd.Timestamp(HO["train_end"])
    reg = np.load(cfg["crypto_regime_path"])[:, 3] - 0.5
    start_bar = int(HO["env_options"].get("warmup_bars", 200))
    tdir = lambda t: 0.0 if abs(reg[t]) <= 0.025 else float(np.sign(reg[t]))
    fixed = {"cash": lambda t: np.zeros(N), "always long": lambda t: np.ones(N),
             "trend rule +-1": lambda t: np.full(N, tdir(t)), "trend rule +-0.5": lambda t: np.full(N, 0.5 * tdir(t)),
             "long in bull, cash in bear": lambda t: np.full(N, max(tdir(t), 0.0))}
    def model_variant(kind):
        def f(a, t):
            if kind == "as trained": return a
            if kind == "x2": return np.clip(2.0 * a, -1, 1)
            if kind == "no stray coins": return np.where(a * tdir(t) < 0, 0.0, a)   # drop signals against a clear trend
            if kind == "own size, trend sign": return np.abs(a) * tdir(t)
        return f
    policies = ([] if ONLY_MODEL else [(n, f, False) for n, f in fixed.items()]) + \
               [(f"{LABEL} {k}", model_variant(k), True) for k in ["as trained", "no stray coins", "x2", "own size, trend sign"]]
    for name, pol, uses_model in policies:
        env = FastCryptoEnv(data_config=cfg, initial_cash=100_000.0); env.reset(); env.current_step = start_bar
        env.portfolio_value = env.peak_equity = 100_000.0
        obs = env._get_observation(); state, startf = None, np.array([True]); rows = []
        while True:
            t = env.current_step
            if uses_model:
                act, state = model.predict(nv.normalize_obs(obs.reshape(1, -1).astype(np.float32)), state=state,
                                           episode_start=startf, deterministic=True)
                startf = np.array([False]); a = pol(np.tanh(np.asarray(act, dtype=np.float64).flatten()[:N]), t)
            else:
                a = pol(t)
            a = np.clip(a, -0.9999, 0.9999)
            obs, r_env, term, trunc, info = env.step(np.arctanh(a).astype(np.float32))
            if term or trunc or "long_exposure" not in info:
                break
            m = env._avail(t); w = env.current_weights[:N].astype(np.float64)
            rows.append((t, float(info["portfolio_return"]), float(info["current_drawdown"]), float(reg[t]),
                         float(w[m].mean()), float(a[m].mean()), float(r_env), float(info["portfolio_value"]),
                         float(np.clip(w, 0, None).sum()), float(-np.clip(w, None, 0).sum())))
        df = pd.DataFrame(rows, columns=["t", "r", "dd", "s", "mean_w", "mean_a", "reward_v40_env", "pv", "long", "short"])
        df["timestamp"] = ts.iloc[df.t.values].values; df["policy"] = name; frames.append(df)
        print(f"[done] {name}: {len(df)} bars, return {(df.pv.iloc[-1] / 1e5 - 1) * 100:+.1f}%")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
pd.concat(frames, ignore_index=True).to_parquet(os.environ["CQ_OUT"], index=False)
