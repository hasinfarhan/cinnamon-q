"""Deep investigation: exactly where and why v10 loses money."""
import io, sys, tempfile, os
import numpy as np
import pandas as pd
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

MODEL_PATH   = project_root / "models" / "hdrl_worldclass_v14" / "final_model.zip"
VECNORM_PATH = project_root / "models" / "hdrl_worldclass_v14" / "vec_normalize.pkl"
DATA_PATH    = project_root / "data" / "processed" / "crypto_deep_v4_sentiment.parquet"

print("Loading model and data...")
model    = RecurrentPPO.load(str(MODEL_PATH), device="cpu")
obs_shape = model.observation_space.shape

temp_dir = Path(tempfile.mkdtemp())
data_cfg  = prepare_training_data(DATA_PATH, temp_dir)
n_steps   = data_cfg["n_steps"]
split_at  = int(n_steps * 0.80)
assets    = data_cfg["asset_list"]

td = str(temp_dir / "test"); os.makedirs(td, exist_ok=True)
def sl(src, name):
    arr = np.load(src, mmap_mode="r")
    out = os.path.join(td, name)
    np.save(out, np.array(arr[split_at:], dtype=np.float32))
    return out

keys = ['returns_path','volatility_path','commit_path','utility_path','price_path',
        'corr_path','garch_path','rsi_path','macd_path','bbw_path','atr_path',
        'sma_ratio_path','macro_path','ret_4h_path','ret_1d_path','ret_7d_path','crypto_regime_path']
test_cfg = {k: sl(data_cfg[k], k.replace('_path','')+'.npy') for k in keys}
test_cfg.update({k: data_cfg[k] for k in ['asset_list','n_assets']})
test_cfg['timestamps']       = data_cfg['timestamps'][split_at:]
test_cfg['n_steps']          = n_steps - split_at
test_cfg['has_ta_features']  = True
test_cfg['has_correlation']  = True
test_cfg['has_garch']        = True
test_cfg['has_macro']        = True
test_cfg['has_mtf']          = True

class _ShapeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
    def reset(self, **kw): return self.observation_space.sample(), {}
    def step(self, a):     return self.observation_space.sample(), 0.0, True, False, {}

dummy    = DummyVecEnv([_ShapeEnv])
norm_env = VecNormalize.load(str(VECNORM_PATH), dummy)
norm_env.training = False; norm_env.norm_reward = False

test_env = FastCryptoEnv(data_config=test_cfg, initial_cash=100_000.0)
test_env.reset(); test_env.current_step = 0
raw_obs = test_env._get_observation()

def normalise(o):
    return norm_env.normalize_obs(o.reshape(1,-1).astype(np.float32))

print(f"Running full test period ({test_cfg['n_steps']:,} steps)...")
lstm_state = None; ep_start = np.array([True])

records = []
n_test  = test_cfg['n_steps']

for i in range(n_test):
    step_idx = test_env.current_step
    action, lstm_state = model.predict(normalise(raw_obs), state=lstm_state,
                                       episode_start=ep_start, deterministic=True)
    ep_start = np.array([False])
    raw_obs, _, term, trunc, info = test_env.step(action[0])

    # Load raw market returns for this step
    mkt_returns = test_env.returns_array[step_idx]
    btc_7d      = test_env.crypto_regime_array[step_idx, 1]
    btc_1d      = test_env.crypto_regime_array[step_idx, 0]

    records.append({
        'step'          : i,
        'pv'            : info['portfolio_value'],
        'pnl_pct'       : (info['portfolio_value'] / 100_000 - 1) * 100,
        'step_return'   : info['portfolio_return'],
        'long_exp'      : info['long_exposure'],
        'short_exp'     : info['short_exposure'],
        'cash'          : info['cash_weight'],
        'calmar_active' : info['calmar_gate_active'],
        'macro_active'  : info['macro_gate_active'],
        'btc_7d'        : float(btc_7d),
        'btc_1d'        : float(btc_1d),
        'btc_return'    : float(mkt_returns[assets.index('BTC')]),
        **{f'alpha_{a}': float(action[0][j]) for j, a in enumerate(assets)},
        **{f'w_{a}': float(test_env.current_weights[j]) for j, a in enumerate(assets)},
    })
    if term or trunc: break

df = pd.DataFrame(records)
timestamps = test_cfg['timestamps'][:len(df)]
df['date'] = pd.to_datetime(timestamps)

print(f"\n{'='*65}")
print(f"  V14 DEEP INVESTIGATION  ({len(df):,} steps)")
print(f"{'='*65}")

# ── 1. Timeline of losses ─────────────────────────────────────────────────────
print(f"\n--- 1. LOSS TIMELINE (weekly P&L) ---")
df['week'] = (df['step'] // (7 * 96)).astype(int)
weekly = df.groupby('week').agg(
    start_date=('date','first'),
    pnl_start=('pnl_pct','first'),
    pnl_end=('pnl_pct','last'),
    avg_long=('long_exp','mean'),
    avg_short=('short_exp','mean'),
    avg_cash=('cash','mean'),
    calmar_pct=('calmar_active','mean'),
    btc_7d_avg=('btc_7d','mean'),
).reset_index()
weekly['weekly_pnl'] = weekly['pnl_end'] - weekly['pnl_start']
for _, row in weekly.iterrows():
    gate = " [GATE]" if row['calmar_pct'] > 0.5 else ""
    regime = "BEAR" if row['btc_7d_avg'] < -0.3 else ("BULL" if row['btc_7d_avg'] > 0.3 else "neutral")
    print(f"  Week {int(row['week']):2d} ({str(row['start_date'])[:10]}) "
          f"P&L: {row['weekly_pnl']:+.2f}%  "
          f"L={row['avg_long']:.2f} S={row['avg_short']:.2f} C={row['avg_cash']:.2f}  "
          f"[{regime}]{gate}")

# ── 2. Is the model responding to regime? ─────────────────────────────────────
print(f"\n--- 2. DOES BRAIN 1 RESPOND TO REGIME? ---")
bear   = df[df['btc_7d'] < -0.3]
bull   = df[df['btc_7d'] >  0.3]
neutral= df[(df['btc_7d'] >= -0.3) & (df['btc_7d'] <= 0.3)]

for label, subset in [('Bear (btc_7d<-0.3)', bear),
                       ('Neutral', neutral),
                       ('Bull (btc_7d>0.3)', bull)]:
    if len(subset) == 0: continue
    btc_alpha = subset['alpha_BTC'].mean()
    sol_alpha = subset['alpha_SOL'].mean()
    cash_avg  = subset['cash'].mean()
    step_ret  = subset['step_return'].mean() * 10000
    print(f"  {label:25s}: BTC_alpha={btc_alpha:+.3f}  SOL_alpha={sol_alpha:+.3f}  "
          f"cash={cash_avg:.2f}  step_ret={step_ret:+.2f}bps")

# ── 3. P&L attribution by long vs short ───────────────────────────────────────
print(f"\n--- 3. P&L ATTRIBUTION ---")
active = df[~df['calmar_active']]
gate_on= df[df['calmar_active']]
print(f"  Steps trading (gate OFF): {len(active):,}  ({len(active)/len(df)*100:.1f}%)")
print(f"  Steps in cash (gate ON) : {len(gate_on):,}  ({len(gate_on)/len(df)*100:.1f}%)")
if len(active) > 0:
    print(f"  Return while trading    : {active['step_return'].sum()*100:+.2f}%")
    print(f"  Return while in cash    : {gate_on['step_return'].sum()*100:+.2f}%")

# Per-asset P&L proxy (weight × market return)
print(f"\n--- 4. ESTIMATED P&L CONTRIBUTION PER ASSET (while trading) ---")
active_idx = df[~df['calmar_active']].index
pnl_contrib = {}
for a in assets:
    w_col = f'w_{a}'
    if w_col in df.columns:
        # P&L = weight × asset_return (approx)
        mkt_ret = test_env.returns_array[:len(df)]
        a_idx   = assets.index(a)
        asset_rets = mkt_ret[active_idx.tolist(), a_idx]
        weights_   = df.loc[active_idx, w_col].values
        contrib    = float((weights_ * asset_rets).sum() * 100)
        pnl_contrib[a] = contrib

sorted_contrib = sorted(pnl_contrib.items(), key=lambda x: x[1])
for a, contrib in sorted_contrib:
    avg_w = df.loc[active_idx, f'w_{a}'].mean()
    tag = "LONG" if avg_w > 0.01 else ("SHORT" if avg_w < -0.01 else "flat")
    bar = "█" * int(abs(contrib) * 2) if abs(contrib) < 20 else "█" * 40
    sign = "+" if contrib >= 0 else ""
    print(f"  {a:6s} [{tag:5s}] avg_w={avg_w:+.3f}  contrib={sign}{contrib:.3f}%  {bar}")

# ── 5. Root cause: what regime is it at the start of the test period? ─────────
print(f"\n--- 5. TEST PERIOD START CONDITIONS ---")
print(f"  First 5 days BTC 7d signal: {df['btc_7d'].head(480).mean():.3f}")
print(f"  First 5 days BTC 1d signal: {df['btc_1d'].head(480).mean():.3f}")
print(f"  BTC return first 5 days   : {df['btc_return'].head(480).sum()*100:.2f}%")
print(f"  Model avg cash first 5d   : {df['cash'].head(480).mean():.3f}")
print(f"  Model avg BTC alpha 1st 5d: {df['alpha_BTC'].head(480).mean():+.3f}")

print(f"\n--- 6. KEY FINDING SUMMARY ---")
print(f"  Total return   : {df['pnl_pct'].iloc[-1]:+.2f}%")
print(f"  Max drawdown   : {((df['pv'].cummax() - df['pv']) / df['pv'].cummax()).max()*100:.2f}%")
print(f"  Brain 1 dynamic: YES (std>0 confirmed in prior diagnosis)")
print(f"  Gate fired     : {gate_on['calmar_active'].any()} ({len(gate_on):,} steps in cash)")
print(f"  Biggest loser  : {sorted_contrib[0][0]} ({sorted_contrib[0][1]:+.3f}%)")
print(f"  Biggest winner : {sorted_contrib[-1][0]} ({sorted_contrib[-1][1]:+.3f}%)")
