"""
Reward landscape on the TRAINING data (no test-year bars are loaded).

Runs fixed rule-based policies and the trained model through the environment in
training mode (same reward and gates the agent learned under) as one long episode,
and reports reward components, return and exposure per period. If a rule that shorts
in bear markets earns more reward than the trained model, the reward supports shorting
and the failure is in learning; if not, the reward itself does not pay for shorting.

Env vars: CQ_MODEL_DIR, CQ_DATA_PATH, CQ_OUT
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
atanh = lambda a: np.arctanh(np.clip(a, -0.9999, 0.9999))   # env applies tanh (squash_actions)

tmp = Path(tempfile.mkdtemp(prefix="cf_")); out = {}
try:
    cfg = prepare_training_data(Path(os.environ["CQ_DATA_PATH"]), tmp, resample_freq="4h", fit_end=HO["train_end"], **(HO.get("loader_options") or {}))
    e = HO["split_at"]
    for k, v in list(cfg.items()):
        if isinstance(v, str) and v.endswith(".npy"):
            o = str(tmp / ("tr_" + Path(v).name)); np.save(o, np.array(np.load(v, mmap_mode="r")[:e])); cfg[k] = o
    cfg.update({"timestamps": cfg["timestamps"][:e], "n_steps": e, "training_mode": True}); cfg.update(HO["env_options"])
    ts = pd.to_datetime(pd.Series(cfg["timestamps"]))
    assert ts.iloc[-1] <= pd.Timestamp(HO["train_end"])
    reg = np.load(cfg["crypto_regime_path"])[:, 3] - 0.5

    def rdir(t, s=None):
        s = reg[t] if s is None else s
        return 0.0 if abs(s) <= 0.025 else np.sign(s)
    policies = {
        "cash":                    lambda t, a: np.zeros(N),
        "always long (+1)":        lambda t, a: np.ones(N),
        "regime rule +-1":         lambda t, a: np.full(N, rdir(t)),
        "regime rule +-0.5":       lambda t, a: np.full(N, 0.5 * rdir(t)),
        "long in bull, cash bear": lambda t, a: np.full(N, max(rdir(t), 0.0)),
        "v45 as trained":          None,
        "v45 size, regime sign":   "mirror",
    }
    for name, pol in policies.items():
        env = FastCryptoEnv(data_config=cfg, initial_cash=100_000.0); env.reset(); env.current_step = int(HO["env_options"].get("warmup_bars", 200))
        env.portfolio_value = env.peak_equity = 100_000.0
        obs = env._get_observation(); state, start = None, np.array([True]); rows = []
        while True:
            t = env.current_step
            if pol is None or pol == "mirror":
                act, state = model.predict(nv.normalize_obs(obs.reshape(1, -1).astype(np.float32)), state=state,
                                           episode_start=start, deterministic=True)
                start = np.array([False]); a = np.tanh(np.asarray(act, dtype=np.float64).flatten()[:N])
                if pol == "mirror": a = np.abs(a) * rdir(t)
            else:
                a = pol(t, None)
            obs, r, term, trunc, info = env.step(atanh(a).astype(np.float32))
            if term or trunc or "long_exposure" not in info:
                break
            w = env.current_weights[:N]; s = reg[t]; d = rdir(t)
            m = env._avail(t)   # v46: C4/C6 average over listed coins, as in the env
            c6 = abs(s) * float(np.mean(np.maximum(w[m] * d, 0))) * 2.0 + abs(s) * float(np.mean(np.minimum(w[m] * d, 0))) * 2.0 if d else 0.0
            c4 = abs(s) * float(np.mean(a[m] * d)) * 0.005 if d else 0.0
            dd = max(0.0, (env.peak_equity - env.portfolio_value) / env.peak_equity); c2 = -dd * 0.005
            rows.append(dict(t=ts.iloc[t], reward=r, C6=c6, C4=c4, C2=c2, C1=r - c6 - c4 - c2, ret=info.get("net_return", np.nan),
                             net=float(w.sum()), gate=bool(env.calmar_gate_active), pv=env.portfolio_value, bear=d < 0))
        df = pd.DataFrame(rows); df["ret"] = df.pv.pct_change().fillna(df.pv.iloc[0] / 1e5 - 1)
        df["period"] = df.t.dt.year.astype(str); df.loc[df.t < "2022-01-01", "period"] = "2021H2"; df.loc[df.t >= "2025-01-01", "period"] = "2025H1"
        g = df.groupby("period").agg(reward=("reward", "sum"), C1=("C1", "sum"), C2=("C2", "sum"), C6=("C6", "sum"),
                                     ret=("ret", lambda x: (np.prod(1 + x) - 1) * 100), net=("net", "mean"),
                                     gate=("gate", lambda x: x.mean() * 100))
        b = df[df.bear]
        out[name] = {"total_reward": float(df.reward.sum()), "total_return_pct": float((df.pv.iloc[-1] / 1e5 - 1) * 100),
                     "maxdd_pct": float(((df.pv / df.pv.cummax()) - 1).min() * 100),
                     "bear_bars_reward": float(b.reward.sum()), "bear_bars_C1": float(b.C1.sum()), "bear_bars_C6": float(b.C6.sum()),
                     "by_period": g.round(3).to_dict(orient="index")}
        print(f"[done] {name}: reward {df.reward.sum():.2f}, return {out[name]['total_return_pct']:+.1f}%")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
Path(os.environ["CQ_OUT"]).write_text(json.dumps(out, indent=1, default=str))
