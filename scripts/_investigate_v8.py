"""Deep investigation: why is v8 static and worse than v7?"""
import sys, tempfile, os, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'strategy' / 'rl'))
from data_loader import prepare_training_data
from fast_env import FastCryptoEnv

cfg = prepare_training_data(
    Path(__file__).parent.parent / 'data/processed/crypto_deep_v4_sentiment.parquet',
    Path(tempfile.mkdtemp())
)
n = cfg['n_steps']
split = int(n * 0.80)

td = tempfile.mkdtemp()
def sl(src, name):
    arr = np.load(src, mmap_mode='r')
    out = os.path.join(td, name)
    np.save(out, np.array(arr[split:], dtype=np.float32))
    return out

keys = ['returns_path','volatility_path','commit_path','utility_path','price_path',
        'corr_path','garch_path','rsi_path','macd_path','bbw_path','atr_path',
        'sma_ratio_path','macro_path','ret_4h_path','ret_1d_path','ret_7d_path','crypto_regime_path']
test_cfg = {k: sl(cfg[k], k.replace('_path','') + '.npy') for k in keys}
test_cfg.update({k: cfg[k] for k in ['asset_list','timestamps','n_assets']})
test_cfg['timestamps'] = cfg['timestamps'][split:]
test_cfg['n_steps'] = n - split
test_cfg['has_ta_features'] = True
test_cfg['has_correlation'] = True
test_cfg['has_garch'] = True
test_cfg['has_macro'] = True
test_cfg['has_mtf'] = True

env = FastCryptoEnv(data_config=test_cfg, initial_cash=100000.0)
env.reset()

# ── Test 1: static allocation (mimic v8 behavior) ────────────────────────────
static_rewards, static_sharpes = [], []
for i in range(500):
    # Exact same fixed allocation v8 uses: 7 long at +0.0667, 7 short at -0.0667
    action = np.array([0.0667]*7 + [-0.0667]*7 + [0.067], dtype=np.float32)
    _, r, term, trunc, info = env.step(action)
    static_rewards.append(r)
    static_sharpes.append(info.get('rolling_sharpe', 0.0))
    if term or trunc:
        break

print('=== TEST 1: Static v8-like allocation (500 steps) ===')
print(f'Total reward   : mean={np.mean(static_rewards):.6f}  std={np.std(static_rewards):.6f}')
print(f'Rolling Sharpe : mean={np.mean(static_sharpes):.4f}  std={np.std(static_sharpes):.4f}')
print(f'Sharpe contrib : mean={np.mean(static_sharpes)*0.002:.6f}  (weight=0.002)')
pos = sum(1 for s in static_sharpes if s > 0)
print(f'Sharpe positive: {pos}/500 ({pos/5:.1f}%)  — if mostly positive, static is REWARDED')

# ── Test 2: all-cash allocation ───────────────────────────────────────────────
env.reset()
cash_rewards, cash_sharpes = [], []
for i in range(500):
    action = np.array([0.0]*14 + [1.0], dtype=np.float32)
    _, r, term, trunc, info = env.step(action)
    cash_rewards.append(r)
    cash_sharpes.append(info.get('rolling_sharpe', 0.0))
    if term or trunc:
        break

print()
print('=== TEST 2: All-cash allocation (500 steps) ===')
print(f'Total reward   : mean={np.mean(cash_rewards):.6f}  std={np.std(cash_rewards):.6f}')
print(f'Rolling Sharpe : mean={np.mean(cash_sharpes):.4f}  std={np.std(cash_sharpes):.4f}')
print(f'Sharpe contrib : mean={np.mean(cash_sharpes)*0.002:.6f}')

# ── Test 3: Does rolling sharpe actually differentiate? ───────────────────────
print()
print('=== TEST 3: Reward comparison ===')
print(f'Static total reward mean : {np.mean(static_rewards):.6f}')
print(f'Cash   total reward mean : {np.mean(cash_rewards):.6f}')
print(f'Difference (cash - static): {np.mean(cash_rewards) - np.mean(static_rewards):.6f}')
print(f'  → Positive means cash is rewarded MORE (model should prefer cash in bear market)')
print(f'  → Negative means static is rewarded MORE (perverse incentive)')

# ── Test 4: Inspect the raw rolling Sharpe values during training window ──────
print()
print('=== TEST 4: Rolling Sharpe behaviour across whole test period ===')
env.reset()
all_sharpes_train = []
for i in range(21018):
    action = np.array([0.0667]*7 + [-0.0667]*7 + [0.067], dtype=np.float32)
    _, r, term, trunc, info = env.step(action)
    all_sharpes_train.append(info.get('rolling_sharpe', 0.0))
    if term or trunc:
        break

arr = np.array(all_sharpes_train)
print(f'Rolling Sharpe full run: mean={arr.mean():.4f}  std={arr.std():.4f}')
print(f'Positive steps: {(arr>0).mean()*100:.1f}%')
print(f'Negative steps: {(arr<0).mean()*100:.1f}%')
print(f'Range: [{arr.min():.4f}, {arr.max():.4f}]')

# Load crypto regime to check bear periods
crypto = np.load(test_cfg['crypto_regime_path'])
btc_7d = crypto[:len(arr), 1]
bear = btc_7d < -0.3
bull = btc_7d > 0.3
if bear.any():
    print(f'Bear periods: sharpe mean={arr[bear].mean():.4f}  (negative = good punishment signal)')
if bull.any():
    print(f'Bull periods: sharpe mean={arr[bull].mean():.4f}')
neutral = ~bear & ~bull
if neutral.any():
    print(f'Neutral     : sharpe mean={arr[neutral].mean():.4f}')
