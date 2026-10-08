"""
Does the training reward want a more confident Brain 1? (training data only)

For each window, the trained model trades as one fresh episode in TRAINING mode (the exact
reward and gates it learned under), with its signals multiplied by k (1 = as trained).
Reports total reward, its components, return, drawdown and breaker firings per k.
If larger k earns more reward, the reward wanted confidence and learning fell short;
if larger k earns less, the reward itself discourages confidence.

Env vars: CQ_MODEL_DIR, CQ_DATA_PATH, CQ_WINDOWS ("name|start|end;..."), CQ_OUT
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
model = RecurrentPPO.load(str(MD / "final_model.zip"), device="cpu")
class _S(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=model.observation_space.shape, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=model.action_space.shape, dtype=np.float32)
    def reset(self, **kw): return self.observation_space.sample(), {}
    def step(self, a): return self.observation_space.sample(), 0.0, True, False, {}
nv = VecNormalize.load(str(MD / "vec_normalize.pkl"), DummyVecEnv([_S])); nv.training, nv.norm_reward = False, False
WINDOWS = [w.split("|") for w in os.environ["CQ_WINDOWS"].split(";")]
KS = [1.0, 2.0, 3.0, 5.0]
tmp = Path(tempfile.mkdtemp(prefix="scale_")); out = {}
try:
    cfg = prepare_training_data(Path(os.environ["CQ_DATA_PATH"]), tmp, resample_freq="4h", fit_end=HO["train_end"],
                                **(HO.get("loader_options") or {}))
    ts = pd.to_datetime(pd.Series(cfg["timestamps"]))
    for wname, ws, we in WINDOWS:
        s = int((ts >= pd.Timestamp(ws)).idxmax()); e = int((ts <= pd.Timestamp(we)).sum())
        assert pd.Timestamp(ts.iloc[e - 1]) <= pd.Timestamp(HO["train_end"]), "window must be inside the training data"
        wcfg = {}
        for k, v in cfg.items():
            if isinstance(v, str) and v.endswith(".npy"):
                o = str(tmp / f"{wname}_{Path(v).name}"); np.save(o, np.array(np.load(v, mmap_mode="r")[s:e])); wcfg[k] = o
            else:
                wcfg[k] = v
        wcfg.update({"timestamps": cfg["timestamps"][s:e], "n_steps": e - s, "training_mode": True}); wcfg.update(HO["env_options"])
        out[wname] = {}
        for K in KS:
            env = FastCryptoEnv(data_config=wcfg, initial_cash=100_000.0); env.reset(); env.current_step = 0
            env.portfolio_value = env.peak_equity = 100_000.0
            obs = env._get_observation(); state, start = None, np.array([True]); R = []
            while True:
                t = env.current_step
                act, state = model.predict(nv.normalize_obs(obs.reshape(1, -1).astype(np.float32)), state=state,
                                           episode_start=start, deterministic=True)
                start = np.array([False])
                a = np.clip(np.tanh(np.asarray(act, dtype=np.float64).flatten()[:N]) * K, -0.9999, 0.9999)
                obs, r, term, trunc, info = env.step(np.arctanh(a).astype(np.float32))
                if term or trunc or "long_exposure" not in info:
                    break
                m = env._avail(t); sg = float(env.crypto_regime_array[t, 3]) - 0.5; d = np.sign(sg) if abs(sg) > 0.025 else 0.0
                w = env.current_weights[:N].astype(np.float64)
                c6 = abs(sg) * (np.mean(np.maximum(w[m] * d, 0)) + np.mean(np.minimum(w[m] * d, 0))) * 2.0 if d else 0.0
                c4 = abs(sg) * float(np.mean(a[m] * d)) * 0.005 if d else 0.0
                c2 = -float(info["current_drawdown"]) * 0.005
                R.append(dict(reward=r, C1=r - c6 - c4 - c2, C2=c2, C6=c6, pv=info["portfolio_value"],
                              net=float(w.sum()), short=float(-np.clip(w, None, 0).sum()), long=float(np.clip(w, 0, None).sum()),
                              gate=bool(env.calmar_gate_active), conv=float(np.abs(a[m]).mean())))
            R = pd.DataFrame(R); pv = R.pv
            fires = int((R.gate.astype(int).diff() == 1).sum() + int(R.gate.iloc[0]))
            out[wname][K] = {"reward": float(R.reward.sum()), "C1_profit": float(R.C1.sum()), "C2_drawdown": float(R.C2.sum()),
                             "C6_trend": float(R.C6.sum()), "return_pct": float((pv.iloc[-1] / 1e5 - 1) * 100),
                             "max_dd_pct": float(((pv / pv.cummax()) - 1).min() * 100), "breaker_fired": fires,
                             "cash_by_breaker_pct": float(R.gate.mean() * 100), "avg_conviction": float(R.conv.mean()),
                             "avg_long": float(R.long.mean()), "avg_short": float(R.short.mean())}
            print(f"[done] {wname} k={K}: reward {R.reward.sum():.2f} return {out[wname][K]['return_pct']:+.1f}%")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
Path(os.environ["CQ_OUT"]).write_text(json.dumps(out, indent=1))
