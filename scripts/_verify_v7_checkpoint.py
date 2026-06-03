"""Verify v7 checkpoint: no rolling Sharpe, correct model paths, reward_type correct."""
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
obs, _ = env.reset()

action = np.array([0.05]*7 + [-0.04]*7 + [0.02], dtype=np.float32)
_, r, _, _, info = env.step(action)

print("=== V7 CHECKPOINT VERIFICATION ===")
print()

# 1. Obs shape
assert obs.shape == (192,), f"FAIL obs shape: {obs.shape}"
print(f"[PASS] obs shape = {obs.shape}")

# 2. No return_buffer
assert not hasattr(env, 'return_buffer'), "FAIL: return_buffer still present (v8 leftover)"
print("[PASS] no return_buffer (v8 rolling Sharpe removed)")

# 3. No _sharpe_window
assert not hasattr(env, '_sharpe_window'), "FAIL: _sharpe_window still present (v8 leftover)"
print("[PASS] no _sharpe_window")

# 4. reward_type is v7
assert info['reward_type'] == 'asymmetric_drawdown_vix', f"FAIL reward_type: {info['reward_type']}"
print(f"[PASS] reward_type = '{info['reward_type']}'")

# 5. No rolling_sharpe in info
assert 'rolling_sharpe' not in info, f"FAIL: rolling_sharpe still in info dict"
print("[PASS] no rolling_sharpe key in info dict")

# 6. Training script points to v7
train_script = Path(__file__).parent / '10_train_hdrl_optimized.py'
content = train_script.read_text(encoding='utf-8', errors='ignore')
assert 'hdrl_worldclass_v7' in content, "FAIL: training script not pointing to v7"
assert 'hdrl_worldclass_v8' not in content, "FAIL: training script still has v8 reference"
print("[PASS] training script points to hdrl_worldclass_v7")

# 7. Backtest script points to v7
backtest_script = Path(__file__).parent / '11_backtest.py'
content = backtest_script.read_text(encoding='utf-8', errors='ignore')
assert 'hdrl_worldclass_v7' in content, "FAIL: backtest script not pointing to v7"
assert 'hdrl_worldclass_v8' not in content, "FAIL: backtest script still has v8 reference"
assert 'backtest_v7' in content, "FAIL: backtest report dir not v7"
print("[PASS] backtest script points to hdrl_worldclass_v7 / backtest_v7")

print()
print("ALL CHECKS PASSED — v7 checkpoint is clean and correct")
