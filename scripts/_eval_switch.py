"""
Regime switch: run two trained models side by side and execute one of them per bar.

  BTC more than 2.5% BELOW its 200-day SMA -> the BEAR model's decision (v50)
  otherwise                                -> the BULL model's decision (v46)

Both models see every bar and keep their own LSTM state, so the one not in charge stays
current. The environment uses the bear model's settings (226 inputs; the bull model gets
the first 212, which are identical to v46's inputs). Brain 2, limits, gates and rule G are
the environment's. CQ_SWITCH_MODE = switch | bull | bear ("bull"/"bear" run one model only
and must reproduce that model's own backtest).

Env vars: CQ_BULL_DIR, CQ_BEAR_DIR, CQ_DATA_PATH, CQ_EVAL_START, CQ_EVAL_END (omit both for
the test year), CQ_SWITCH_MODE, CQ_OUT (.csv of per-bar diagnostics)
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

BULL, BEAR = Path(os.environ["CQ_BULL_DIR"]), Path(os.environ["CQ_BEAR_DIR"])
MODE = os.environ.get("CQ_SWITCH_MODE", "switch"); N = 14
HB, HR = (json.loads((d / "holdout.json").read_text()) for d in (BULL, BEAR))
assert HB["train_end"] == HR["train_end"] and HB["data_file"] == HR["data_file"] and HB["loader_options"] == HR["loader_options"]

def load(d):
    m = RecurrentPPO.load(str(d / "final_model.zip"), device="cpu")
    shape = m.observation_space.shape
    class _S(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=m.action_space.shape, dtype=np.float32)
        def reset(self, **kw): return self.observation_space.sample(), {}
        def step(self, a): return self.observation_space.sample(), 0.0, True, False, {}
    v = VecNormalize.load(str(d / "vec_normalize.pkl"), DummyVecEnv([_S])); v.training, v.norm_reward = False, False
    return m, v, shape[0]
mb, vb, nb = load(BULL); mr, vr, nr = load(BEAR)

tmp = Path(tempfile.mkdtemp(prefix="switch_"))
try:
    cfg = prepare_training_data(Path(os.environ["CQ_DATA_PATH"]), tmp, resample_freq="4h", fit_end=HR["train_end"], **HR["loader_options"])
    ts = pd.to_datetime(pd.Series(cfg["timestamps"]))
    if os.environ.get("CQ_EVAL_START"):
        s = int((ts >= pd.Timestamp(os.environ["CQ_EVAL_START"])).idxmax()); e = int((ts <= pd.Timestamp(os.environ["CQ_EVAL_END"])).sum())
        label = f"IN-SAMPLE {ts.iloc[s]} -> {ts.iloc[e - 1]}"
    else:
        s, e = HR["split_at"], len(ts); label = f"TEST YEAR {ts.iloc[s]} -> {ts.iloc[e - 1]}"
        assert str(ts.iloc[s - 1]) == HR["train_end"]
    wcfg = {}
    for k, v in cfg.items():
        if isinstance(v, str) and v.endswith(".npy"):
            o = str(tmp / ("w_" + Path(v).name)); np.save(o, np.array(np.load(v, mmap_mode="r")[s:e], dtype=np.float32)); wcfg[k] = o   # float32, as 11_backtest slices
        else:
            wcfg[k] = v
    wcfg.update({"timestamps": cfg["timestamps"][s:e], "n_steps": e - s, "training_mode": False}); wcfg.update(HR["env_options"])
    env = FastCryptoEnv(data_config=wcfg, initial_cash=100_000.0); env.reset(); env.current_step = 0
    assert env.observation_space.shape[0] == nr and nb <= nr
    obs = env._get_observation(); stb = str_ = None; first = np.array([True]); rows = []
    while True:
        t = env.current_step; sig = float(env.crypto_regime_array[t, 3]) - 0.5
        ab, stb = mb.predict(vb.normalize_obs(obs[:nb].reshape(1, -1).astype(np.float32)), state=stb, episode_start=first, deterministic=True)
        ar, str_ = mr.predict(vr.normalize_obs(obs.reshape(1, -1).astype(np.float32)), state=str_, episode_start=first, deterministic=True)
        first = np.array([False])
        use_bear = (MODE == "bear") or (MODE == "switch" and sig < -0.025)
        obs, _, term, trunc, info = env.step(ar if use_bear else ab)
        if "long_exposure" not in info:          # terminal bookkeeping step: no trade happened
            break
        rows.append({"timestamp": ts.iloc[s + t], "net_return": info["portfolio_return"], "long": info["long_exposure"],
                     "short": info["short_exposure"], "fee": info["transaction_cost"], "bear_model": use_bear, "sig": sig,
                     "gate": bool(env.calmar_gate_active), "pv": info["portfolio_value"]})
        if term or trunc:                         # record the last traded bar before stopping
            break
finally:
    shutil.rmtree(tmp, ignore_errors=True)
d = pd.DataFrame(rows); d.to_csv(os.environ["CQ_OUT"], index=False)
pv = np.concatenate([[100_000.0], d.pv.values]); r = np.diff(pv) / pv[:-1]
down = r[r < 0].std()
print(f"[{MODE}] {label}: final ${pv[-1]:,.2f} | return {(pv[-1] / 1e5 - 1) * 100:+.2f}% | Sharpe {r.mean() / r.std() * np.sqrt(2190):.3f} | "
      f"Sortino {r.mean() / down * np.sqrt(2190):.3f} | MaxDD {((pv / np.maximum.accumulate(pv)) - 1).min() * 100:.2f}% | "
      f"bear model in charge {d.bear_model.mean() * 100:.0f}% of bars | switches {int((d.bear_model.astype(int).diff().abs() == 1).sum())} | fees {d.fee.sum() * 100:.2f} pp")
