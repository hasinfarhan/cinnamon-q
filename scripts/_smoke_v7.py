"""Smoke test for v7 changes."""
import sys, tempfile, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'strategy' / 'rl'))
from data_loader import prepare_training_data
from fast_env import FastCryptoEnv

cfg = prepare_training_data(
    Path(__file__).parent.parent / 'data/processed/crypto_deep_v4_sentiment.parquet',
    Path(tempfile.mkdtemp())
)

print(f"\nhas_mtf          : {cfg.get('has_mtf')}")
print(f"has_macro        : {cfg.get('has_macro')}")
print(f"crypto_regime_path exists: {Path(cfg['crypto_regime_path']).exists()}")

env = FastCryptoEnv(data_config=cfg)
print(f"\nObs shape : {env.observation_space.shape}  (expected 192)")
print(f"Action low : {env.action_space.low[:3]} ... {env.action_space.high[:3]}")

obs, _ = env.reset()
env.current_step = 0
obs = env._get_observation()

# Mixed long/short action
action = np.zeros(env.n_assets + 1, dtype=np.float32)
btc_idx = env.assets.index('BTC')
eth_idx = env.assets.index('ETH')
action[btc_idx]  =  0.5
action[eth_idx]  = -0.3
action[-1]       =  0.2
_, r, _, _, info = env.step(action)

print(f"\nStep result:")
print(f"  reward        : {r:.6f}")
print(f"  long_exposure : {info['long_exposure']:.3f}  (should ~0.5)")
print(f"  short_exposure: {info['short_exposure']:.3f}  (should ~0.3)")
print(f"  cash_weight   : {info['cash_weight']:.3f}   (should ~0.2)")

assert env.observation_space.shape == (192,), f"Wrong obs shape: {env.observation_space.shape}"
assert abs(info['long_exposure'] - 0.5) < 0.01
assert abs(info['short_exposure'] - 0.3) < 0.01
assert abs(info['cash_weight'] - 0.2) < 0.01
print("\nSMOKE TEST PASSED")
