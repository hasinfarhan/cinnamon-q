"""Diagnose v6 model allocation behaviour — what is it actually doing?"""
import io, sys, tempfile
import numpy as np, pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "strategy" / "rl"))

import torch as _torch
if not getattr(_torch.load, "_bytesio_patched", False):
    _orig = _torch.load
    def _compat(f, *a, **kw):
        if not isinstance(f, (str, bytes, io.BytesIO)) and hasattr(f, "read"):
            f = io.BytesIO(f.read())
        return _orig(f, *a, **kw)
    _compat._bytesio_patched = True
    _torch.load = _compat

import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from strategy.rl.data_loader import prepare_training_data
from strategy.rl.fast_env import FastCryptoEnv

MODEL_PATH   = project_root / "models" / "hdrl_worldclass_v6" / "final_model.zip"
VECNORM_PATH = project_root / "models" / "hdrl_worldclass_v6" / "vec_normalize.pkl"
DATA_PATH    = project_root / "data" / "processed" / "crypto_deep_v4_sentiment.parquet"
TEST_FRACTION = 0.20
INITIAL_CASH  = 100_000.0

print("Loading model...")
model = RecurrentPPO.load(str(MODEL_PATH), device="cpu")
obs_shape    = model.observation_space.shape
action_shape = model.action_space.shape
print(f"  obs_shape   : {obs_shape}")
print(f"  action_shape: {action_shape}")

print("Preparing data...")
temp_dir = Path(tempfile.mkdtemp())
data_cfg  = prepare_training_data(DATA_PATH, temp_dir)
n_steps   = data_cfg["n_steps"]
split_at  = int(n_steps * (1 - TEST_FRACTION))
n_test    = n_steps - split_at
assets    = data_cfg["asset_list"]
timestamps = data_cfg["timestamps"]

test_dir = temp_dir / "test"; test_dir.mkdir()
def _slice(src, name):
    arr = np.load(src, mmap_mode="r")
    out = str(test_dir / name); np.save(out, np.array(arr[split_at:], dtype=np.float32)); return out

test_cfg = {
    "returns_path"   : _slice(data_cfg["returns_path"],    "returns_array.npy"),
    "volatility_path": _slice(data_cfg["volatility_path"], "volatility_array.npy"),
    "commit_path"    : _slice(data_cfg["commit_path"],     "commit_array.npy"),
    "utility_path"   : _slice(data_cfg["utility_path"],    "utility_array.npy"),
    "price_path"     : _slice(data_cfg["price_path"],      "price_array.npy"),
    "corr_path"      : _slice(data_cfg["corr_path"],       "corr_array.npy"),
    "garch_path"     : _slice(data_cfg["garch_path"],      "garch_array.npy"),
    "rsi_path"       : _slice(data_cfg["rsi_path"],        "rsi_array.npy"),
    "macd_path"      : _slice(data_cfg["macd_path"],       "macd_array.npy"),
    "bbw_path"       : _slice(data_cfg["bbw_path"],        "bbw_array.npy"),
    "atr_path"       : _slice(data_cfg["atr_path"],        "atr_array.npy"),
    "sma_ratio_path" : _slice(data_cfg["sma_ratio_path"],  "sma_ratio_array.npy"),
    "macro_path"     : _slice(data_cfg["macro_path"],      "macro_array.npy"),
    "asset_list": assets, "timestamps": timestamps[split_at:],
    "n_steps": n_test, "n_assets": data_cfg["n_assets"],
    "has_ta_features": True, "has_correlation": True,
    "has_garch": True, "has_macro": True,
}

class _ShapeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=action_shape, dtype=np.float32)
    def reset(self, **kw): return self.observation_space.sample(), {}
    def step(self, a):     return self.observation_space.sample(), 0.0, True, False, {}

dummy    = DummyVecEnv([_ShapeEnv])
norm_env = VecNormalize.load(str(VECNORM_PATH), dummy)
norm_env.training = False; norm_env.norm_reward = False

test_env = FastCryptoEnv(data_config=test_cfg, initial_cash=INITIAL_CASH)
test_env.reset(); test_env.current_step = 0
raw_obs = test_env._get_observation()

def normalise(o):
    return norm_env.normalize_obs(o.reshape(1, -1).astype(np.float32))

print(f"\nRunning {n_test:,} steps...")
lstm_state = None; ep_start = np.array([True]); done = False

long_exp_list, short_exp_list, cash_list = [], [], []
asset_weights = []  # (n_test, n_assets)

step = 0
while not done:
    norm_obs = normalise(raw_obs)
    action, lstm_state = model.predict(norm_obs, state=lstm_state, episode_start=ep_start, deterministic=True)
    ep_start = np.array([False])
    raw_obs, _, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

    long_exp_list.append(info['long_exposure'])
    short_exp_list.append(info['short_exposure'])
    cash_list.append(info['cash_weight'])
    asset_weights.append(test_env.current_weights[:-1].copy())
    step += 1

long_arr  = np.array(long_exp_list)
short_arr = np.array(short_exp_list)
cash_arr  = np.array(cash_list)
w_arr     = np.array(asset_weights)  # (steps, n_assets)

print(f"\n{'='*60}")
print(f"  V6 ALLOCATION DIAGNOSIS  ({step:,} steps)")
print(f"{'='*60}")
print(f"\n--- Average Exposure ---")
print(f"  Long  exposure : {long_arr.mean():.3f}  (min {long_arr.min():.3f} max {long_arr.max():.3f})")
print(f"  Short exposure : {short_arr.mean():.3f}  (min {short_arr.min():.3f} max {short_arr.max():.3f})")
print(f"  Cash  weight   : {cash_arr.mean():.3f}  (min {cash_arr.min():.3f} max {cash_arr.max():.3f})")

print(f"\n--- Time in Each Regime ---")
print(f"  Mostly cash (cash>0.7)  : {(cash_arr>0.7).mean()*100:.1f}% of steps")
print(f"  Mostly long (long>0.5)  : {(long_arr>0.5).mean()*100:.1f}% of steps")
print(f"  Has shorts (short>0.05) : {(short_arr>0.05).mean()*100:.1f}% of steps")
print(f"  Net short (short>long)  : {(short_arr>long_arr).mean()*100:.1f}% of steps")

print(f"\n--- Per-Asset Average Weight ---")
for i, a in enumerate(assets):
    avg_w = w_arr[:, i].mean()
    max_long  = w_arr[:, i].max()
    max_short = w_arr[:, i].min()
    pct_short = (w_arr[:, i] < -0.01).mean() * 100
    print(f"  {a:6s}: avg={avg_w:+.4f}  max_long={max_long:+.4f}  max_short={max_short:+.4f}  short_pct={pct_short:.1f}%")

print(f"\n--- Short Usage Summary ---")
any_short = (w_arr < -0.01).any(axis=1)
print(f"  Steps with ANY short position : {any_short.mean()*100:.1f}%")
print(f"  Max total short exposure ever : {short_arr.max():.3f}")
print(f"  Avg short when shorting active: {short_arr[short_arr>0.01].mean():.3f}" if (short_arr>0.01).any() else "  No shorts taken")

print(f"\n--- Macro Response Check ---")
macro = np.load(test_cfg["macro_path"])  # (n_test, 4) — vix, dxy, tnx, macro_score
macro_score = macro[:len(cash_arr), 3]
high_risk_off = macro_score < 0.3
if high_risk_off.any():
    print(f"  Steps with macro_score<0.3 (risk-off): {high_risk_off.mean()*100:.1f}%")
    print(f"  Avg cash in risk-off periods : {cash_arr[high_risk_off].mean():.3f}")
    print(f"  Avg cash in normal periods   : {cash_arr[~high_risk_off].mean():.3f}")
else:
    print(f"  No strong risk-off periods detected (macro_score always ≥0.3)")

print(f"\n{'='*60}")
