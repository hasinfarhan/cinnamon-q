"""Same policy, same window (training data only), training_mode True vs False.
Isolates what the inference-only rules (Gate 3 regime band, B re-entry) change.
Env vars: CQ_MODEL_DIR, CQ_DATA_PATH, CQ_WINDOW ("start|end")"""
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
ws, we = os.environ["CQ_WINDOW"].split("|")
tmp = Path(tempfile.mkdtemp(prefix="modes_"))
try:
    cfg = prepare_training_data(Path(os.environ["CQ_DATA_PATH"]), tmp, resample_freq="4h", fit_end=HO["train_end"], **(HO.get("loader_options") or {}))
    ts = pd.to_datetime(pd.Series(cfg["timestamps"]))
    s = int((ts >= pd.Timestamp(ws)).idxmax()); e = int((ts <= pd.Timestamp(we)).sum())
    assert e <= HO["split_at"], "window must lie inside the training data"
    wcfg = {}
    for k, v in cfg.items():
        if isinstance(v, str) and v.endswith(".npy"):
            o = str(tmp / ("w_" + Path(v).name)); np.save(o, np.array(np.load(v, mmap_mode="r")[s:e])); wcfg[k] = o
        else:
            wcfg[k] = v
    wcfg.update({"timestamps": cfg["timestamps"][s:e], "n_steps": e - s}); wcfg.update(HO["env_options"])
    for label, policy in [("v45", "model"), ("always long", "long")]:
        for mode in [True, False]:
            env = FastCryptoEnv(data_config={**wcfg, "training_mode": mode}, initial_cash=100_000.0)
            env.reset(); env.current_step = 0; env.portfolio_value = env.peak_equity = 100_000.0
            obs = env._get_observation(); state, start = None, np.array([True]); R = []
            while True:
                if policy == "model":
                    act, state = model.predict(nv.normalize_obs(obs.reshape(1, -1).astype(np.float32)), state=state,
                                               episode_start=start, deterministic=True); start = np.array([False])
                else:
                    act = np.full(N, 5.0, dtype=np.float32)
                sig = float(env.crypto_regime_array[env.current_step, 3]) - 0.5
                obs, r, term, trunc, info = env.step(act)
                if term or trunc or "long_exposure" not in info:
                    break
                R.append((info["portfolio_value"], info["long_exposure"] - info["short_exposure"], info["cash_weight"],
                          bool(env.calmar_gate_active), abs(sig) <= 0.025, info["transaction_cost"]))
            R = pd.DataFrame(R, columns=["pv", "net", "cash", "gate", "band", "fee"])
            print(f"{label:12s} training_mode={str(mode):5s}: return {(R.pv.iloc[-1] / 1e5 - 1) * 100:+7.1f}% | maxDD "
                  f"{((R.pv / R.pv.cummax()) - 1).min() * 100:6.1f}% | avg net {R.net.mean():.2f} | 100% cash {(R.cash > 0.999).mean() * 100:4.1f}% "
                  f"| gate1 {R.gate.mean() * 100:4.1f}% | dead-zone bars {R.band.mean() * 100:4.1f}% | fees {R.fee.sum() * 100:.2f}pp")
            rr = R.pv.pct_change().fillna(R.pv.iloc[0] / 1e5 - 1)
            print(f"{'':12s}   compounded return on dead-zone bars {(np.prod(1 + rr[R.band]) - 1) * 100:+.1f}% | other bars "
                  f"{(np.prod(1 + rr[~R.band]) - 1) * 100:+.1f}% | bars after first dead-zone bar: net {R.net[R.band.cummax()].mean():.2f}"
                  f" vs before {R.net[~R.band.cummax()].mean():.2f}")
            if os.environ.get("CQ_DUMP"): R.assign(ret=rr).to_csv(f"{os.environ['CQ_DUMP']}_{label.replace(' ', '_')}_{mode}.csv")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
