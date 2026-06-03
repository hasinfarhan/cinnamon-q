"""Smoke test for v8 rolling Sharpe reward."""
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
env = FastCryptoEnv(data_config=cfg)
assert env.observation_space.shape == (192,), f"Wrong obs: {env.observation_space.shape}"

obs, _ = env.reset()
env.current_step = 0

print("Running 200 steps to warm up rolling Sharpe buffer...")
rewards, sharpes = [], []
for i in range(200):
    action = np.random.uniform(-1, 1, env.n_assets + 1).astype(np.float32)
    action[-1] = abs(action[-1])  # cash must be positive
    _, r, term, trunc, info = env.step(action)
    rewards.append(r)
    sharpes.append(info.get('rolling_sharpe', 0.0))
    if term or trunc:
        break

print(f"  Return buffer length at step 200: {len(env.return_buffer)}  (should be 96 — capped)")
print(f"  reward_type : {info['reward_type']}")
print(f"  reward mean : {np.mean(rewards):.6f}")
print(f"  rolling_sharpe (last step): {info['rolling_sharpe']:.4f}")
print(f"  rolling_sharpe range: [{min(sharpes):.3f}, {max(sharpes):.3f}]")

# Verify buffer is capped at 96
assert len(env.return_buffer) == 96, f"Buffer not capped: {len(env.return_buffer)}"

# Verify rolling_sharpe is non-zero after warmup
assert sharpes[-1] != 0.0, "rolling_sharpe should be non-zero after 20+ steps"

# Verify reward_type changed
assert info['reward_type'] == 'asymmetric_rolling_sharpe'

# Test reset clears buffer
env.reset()
assert len(env.return_buffer) == 0, "Buffer not cleared on reset"

print("\nSMOKE TEST PASSED — rolling Sharpe reward is working")
