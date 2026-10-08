"""Sign pattern of Brain 1 across training checkpoints, replayed on one fixed window.
Note: checkpoints have no saved normalization stats, so the final vec_normalize.pkl is used
for all of them (approximate for early checkpoints; fine for the sign pattern question).
Env vars: CQ_MODEL_DIR, CQ_DATA_PATH, CQ_WINDOW ("start|end"), CQ_CKPT_STEPS ("10240,..."), CQ_OUT"""
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

MD = Path(os.environ["CQ_MODEL_DIR"]); HO = json.loads((MD / "holdout.json").read_text())
ws, we = os.environ["CQ_WINDOW"].split("|"); steps = [int(x) for x in os.environ["CQ_CKPT_STEPS"].split(",")]
final = RecurrentPPO.load(str(MD / "final_model.zip"), device="cpu")
class _S(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=final.observation_space.shape, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=final.action_space.shape, dtype=np.float32)
    def reset(self, **kw): return self.observation_space.sample(), {}
    def step(self, a): return self.observation_space.sample(), 0.0, True, False, {}
nv = VecNormalize.load(str(MD / "vec_normalize.pkl"), DummyVecEnv([_S])); nv.training = False
tmp = Path(tempfile.mkdtemp(prefix="ckpt_")); out = {}
try:
    cfg = prepare_training_data(Path(os.environ["CQ_DATA_PATH"]), tmp, resample_freq="4h", fit_end=HO["train_end"], **(HO.get("loader_options") or {}))
    ts = pd.to_datetime(pd.Series(cfg["timestamps"]))
    s = int((ts >= pd.Timestamp(ws)).idxmax()); e = int((ts <= pd.Timestamp(we)).sum())
    wcfg = {}
    for k, v in cfg.items():
        if isinstance(v, str) and v.endswith(".npy"):
            o = str(tmp / ("w_" + Path(v).name)); np.save(o, np.array(np.load(v, mmap_mode="r")[s:e])); wcfg[k] = o
        else:
            wcfg[k] = v
    wcfg.update({"timestamps": cfg["timestamps"][s:e], "n_steps": e - s, "training_mode": False})
    wcfg.update(HO.get("env_options") or {})   # v45: squash_actions etc. must match training
    assets = cfg["asset_list"]
    for st in steps:
        f = MD / ("final_model.zip" if st == 0 else f"hdrl_crypto_lstm_{st}_steps.zip")
        m = final if st == 0 else RecurrentPPO.load(str(f), device="cpu")
        env = FastCryptoEnv(data_config=wcfg, initial_cash=100_000.0); env.reset(); env.current_step = 0
        obs = env._get_observation(); state, start, A = None, np.array([True]), []
        while True:
            a, state = m.predict(nv.normalize_obs(obs.reshape(1, -1).astype(np.float32)), state=state, episode_start=start, deterministic=True)
            start = np.array([False]); _r = np.asarray(a).flatten()[:14]; A.append(np.tanh(_r) if getattr(env, "squash_actions", False) else np.clip(_r, -1, 1))
            obs, _, term, trunc, info = env.step(a)
            if term or trunc or "long_exposure" not in info:
                break
        A = np.array(A)
        out[st if st else "final"] = {"mean_abs": float(np.abs(A).mean()), "mean": float(A.mean()),
                                      "pct_all_negative": float((A < 0).mean() * 100),
                                      "pct_negative": {assets[i]: float((A[:, i] < 0).mean() * 100) for i in range(14)}}
        print(f"[done] checkpoint {st or 'final'}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
Path(os.environ["CQ_OUT"]).write_text(json.dumps(out, indent=1))
