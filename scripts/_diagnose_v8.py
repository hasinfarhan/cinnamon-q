"""Diagnose v7 model allocation behaviour."""
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

MODEL_PATH   = project_root / "models" / "hdrl_worldclass_v8" / "final_model.zip"
VECNORM_PATH = project_root / "models" / "hdrl_worldclass_v8" / "vec_normalize.pkl"
DATA_PATH    = project_root / "data" / "processed" / "crypto_deep_v4_sentiment.parquet"
TEST_FRACTION = 0.20
INITIAL_CASH  = 100_000.0

print("Loading model...")
model = RecurrentPPO.load(str(MODEL_PATH), device="cpu")
obs_shape = model.observation_space.shape
print(f"  obs_shape: {obs_shape}")

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
    out = str(test_dir / name)
    np.save(out, np.array(arr[split_at:], dtype=np.float32))
    return out

test_cfg = {
    "returns_path"       : _slice(data_cfg["returns_path"],        "returns_array.npy"),
    "volatility_path"    : _slice(data_cfg["volatility_path"],     "volatility_array.npy"),
    "commit_path"        : _slice(data_cfg["commit_path"],         "commit_array.npy"),
    "utility_path"       : _slice(data_cfg["utility_path"],        "utility_array.npy"),
    "price_path"         : _slice(data_cfg["price_path"],          "price_array.npy"),
    "corr_path"          : _slice(data_cfg["corr_path"],           "corr_array.npy"),
    "garch_path"         : _slice(data_cfg["garch_path"],          "garch_array.npy"),
    "rsi_path"           : _slice(data_cfg["rsi_path"],            "rsi_array.npy"),
    "macd_path"          : _slice(data_cfg["macd_path"],           "macd_array.npy"),
    "bbw_path"           : _slice(data_cfg["bbw_path"],            "bbw_array.npy"),
    "atr_path"           : _slice(data_cfg["atr_path"],            "atr_array.npy"),
    "sma_ratio_path"     : _slice(data_cfg["sma_ratio_path"],      "sma_ratio_array.npy"),
    "macro_path"         : _slice(data_cfg["macro_path"],          "macro_array.npy"),
    "ret_4h_path"        : _slice(data_cfg["ret_4h_path"],         "ret_4h_array.npy"),
    "ret_1d_path"        : _slice(data_cfg["ret_1d_path"],         "ret_1d_array.npy"),
    "ret_7d_path"        : _slice(data_cfg["ret_7d_path"],         "ret_7d_array.npy"),
    "crypto_regime_path" : _slice(data_cfg["crypto_regime_path"],  "crypto_regime_array.npy"),
    "asset_list": assets, "timestamps": timestamps[split_at:],
    "n_steps": n_test, "n_assets": data_cfg["n_assets"],
    "has_ta_features": True, "has_correlation": True, "has_garch": True,
    "has_macro": True, "has_mtf": True,
}

class _ShapeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(15,), dtype=np.float32)
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

long_list, short_list, cash_list, weights_list = [], [], [], []
while not done:
    action, lstm_state = model.predict(normalise(raw_obs), state=lstm_state,
                                       episode_start=ep_start, deterministic=True)
    ep_start = np.array([False])
    raw_obs, _, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    long_list.append(info['long_exposure'])
    short_list.append(info['short_exposure'])
    cash_list.append(info['cash_weight'])
    weights_list.append(test_env.current_weights[:-1].copy())

long_arr  = np.array(long_list)
short_arr = np.array(short_list)
cash_arr  = np.array(cash_list)
w_arr     = np.array(weights_list)

print(f"\n{'='*60}")
print(f"  V8 ALLOCATION DIAGNOSIS  ({len(long_arr):,} steps)")
print(f"{'='*60}")
print(f"\n--- Average Exposure ---")
print(f"  Long  : {long_arr.mean():.3f}  (min {long_arr.min():.3f}  max {long_arr.max():.3f})")
print(f"  Short : {short_arr.mean():.3f}  (min {short_arr.min():.3f}  max {short_arr.max():.3f})")
print(f"  Cash  : {cash_arr.mean():.3f}  (min {cash_arr.min():.3f}  max {cash_arr.max():.3f})")

print(f"\n--- Time in Each Regime ---")
print(f"  Mostly cash  (>70%)   : {(cash_arr>0.7).mean()*100:.1f}%")
print(f"  Mostly long  (>50%)   : {(long_arr>0.5).mean()*100:.1f}%")
print(f"  Has shorts   (>5%)    : {(short_arr>0.05).mean()*100:.1f}%")
print(f"  Net short (short>long): {(short_arr>long_arr).mean()*100:.1f}%")

print(f"\n--- Dynamic Range (std of allocation over time — higher = more dynamic) ---")
for i, a in enumerate(assets):
    std = w_arr[:, i].std()
    avg = w_arr[:, i].mean()
    tag = "DYNAMIC" if std > 0.02 else "static"
    print(f"  {a:6s}: avg={avg:+.4f}  std={std:.4f}  [{tag}]")

print(f"\n--- Short Usage ---")
print(f"  Steps with any short  : {(short_arr>0.01).mean()*100:.1f}%")
print(f"  Max total short ever  : {short_arr.max():.3f}")
print(f"  Net short steps       : {(short_arr>long_arr).mean()*100:.1f}%")

crypto = np.load(test_cfg["crypto_regime_path"])
btc_7d = crypto[:len(cash_arr), 1]
bear   = btc_7d < -0.3
bull   = btc_7d >  0.3
print(f"\n--- Regime Response (BTC 7d return signal) ---")
print(f"  Bear steps (btc_7d < -0.3): {bear.mean()*100:.1f}%")
if bear.any():
    print(f"    cash={cash_arr[bear].mean():.3f}  short={short_arr[bear].mean():.3f}  long={long_arr[bear].mean():.3f}")
print(f"  Bull steps (btc_7d >  0.3): {bull.mean()*100:.1f}%")
if bull.any():
    print(f"    cash={cash_arr[bull].mean():.3f}  short={short_arr[bull].mean():.3f}  long={long_arr[bull].mean():.3f}")
neutral = ~bear & ~bull
print(f"  Neutral steps             : {neutral.mean()*100:.1f}%")
if neutral.any():
    print(f"    cash={cash_arr[neutral].mean():.3f}  short={short_arr[neutral].mean():.3f}  long={long_arr[neutral].mean():.3f}")
print(f"{'='*60}")
