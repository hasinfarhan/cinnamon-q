"""
v39 Deep Diagnostic: Alpha conviction, Brain 2 positions, reward components.
Run after backtest. Prints per-asset stats and reward component breakdown.
"""
import sys, os, tempfile, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces
from strategy.rl.fast_env import FastCryptoEnv

# Use same pipeline as backtest
from strategy.rl.data_loader import prepare_training_data

MODEL_PATH   = project_root / "models" / "hdrl_worldclass_v40" / "final_model.zip"
VECNORM_PATH = project_root / "models" / "hdrl_worldclass_v40" / "vec_normalize.pkl"
DATA_PATH    = project_root / "data" / "processed" / "crypto_deep_v5.parquet"
TEST_FRACTION = 0.20
INITIAL_CASH  = 100_000.0

print("=" * 70)
print("  v39 DEEP DIAGNOSTIC")
print("=" * 70)

# -- Load model -------------------------------------------------------------
model = RecurrentPPO.load(str(MODEL_PATH), device="cpu")
assets = ['AAVE', 'AVAX', 'BAT', 'BCH', 'BTC', 'CRV', 'ETH', 'GRT',
          'LINK', 'LTC', 'SOL', 'SUSHI', 'UNI', 'YFI']
n_assets = len(assets)
btc_idx  = assets.index('BTC')

# -- Prepare data -----------------------------------------------------------
temp_dir = tempfile.mkdtemp(prefix="diag_v39_")
data_cfg  = prepare_training_data(DATA_PATH, temp_dir, resample_freq='4h')
n_steps   = data_cfg["n_steps"]
split_at  = int(n_steps * (1 - TEST_FRACTION))
timestamps = data_cfg["timestamps"]

test_cfg = {k: v for k, v in data_cfg.items()}
test_cfg.update({
    "timestamps" : timestamps[split_at:],
    "n_steps"    : n_steps - split_at,
    "bars_per_day": data_cfg.get("bars_per_day", 6),
    "training_mode": False,
})
# slice npy arrays to test window
for key in ["prices_path","returns_path","volatility_path","rsi_path","macd_path",
            "bbw_path","atr_path","sma_ratio_path","corr_btc_path","garch_path",
            "macro_path","crypto_regime_path"]:
    if key in test_cfg:
        arr = np.load(test_cfg[key], mmap_mode='r')
        sliced = arr[split_at:]
        sliced_path = os.path.join(temp_dir, f"test_{key}.npy")
        np.save(sliced_path, sliced)
        test_cfg[key] = sliced_path

# -- VecNormalize -----------------------------------------------------------
obs_shape    = model.observation_space.shape
action_shape = model.action_space.shape

class _ShapeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space      = spaces.Box(-1., 1., shape=action_shape, dtype=np.float32)
    def reset(self, **kw): return self.observation_space.sample(), {}
    def step(self, a):     return self.observation_space.sample(), 0., True, False, {}

dummy    = DummyVecEnv([_ShapeEnv])
norm_env = VecNormalize.load(str(VECNORM_PATH), dummy)
norm_env.training    = False
norm_env.norm_reward = False

def normalise(obs):
    o = np.array(obs, dtype=np.float32).reshape(1, -1)
    return norm_env.normalize_obs(o)[0]

# -- Run test period, record everything ------------------------------------
test_env = FastCryptoEnv(data_config=test_cfg, initial_cash=INITIAL_CASH)
raw_obs, _ = test_env.reset()
test_env.current_step = 0
raw_obs = test_env._get_observation()

lstm_state = None
ep_start   = np.array([True])
done       = False

alphas_all   = []   # Brain 1 outputs  [n_steps x 14]
weights_all  = []   # Brain 2 weights  [n_steps x 15]  (14 assets + cash)
sma10_all    = []
sma50_all    = []
btc_return   = []   # actual BTC 4h return at each step (sign of market)
pv_all       = []
step_ts      = []

n_test = test_cfg["n_steps"]
_btc_returns = np.load(test_cfg["returns_path"], mmap_mode='r')[:, btc_idx]

step = 0
while not done:
    norm_obs = normalise(raw_obs)
    action, lstm_state = model.predict(
        norm_obs, state=lstm_state, episode_start=ep_start, deterministic=True)
    ep_start = np.array([False])

    # Record Brain 1 alpha BEFORE env.step changes current_step
    t = test_env.current_step
    alphas_all.append(action.copy())
    sma10_all.append(float(test_env.sma10_signal[t]) if t < len(test_env.sma10_signal) else 0.)
    sma50_all.append(float(test_env.sma50_signal[t]) if t < len(test_env.sma50_signal) else 0.)
    btc_return.append(float(_btc_returns[t]) if t < len(_btc_returns) else 0.)

    raw_obs, _, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated

    weights_all.append(test_env.current_weights.copy())
    pv_all.append(float(test_env.portfolio_value))
    if split_at + step < len(timestamps):
        step_ts.append(timestamps[split_at + step])
    step += 1

alphas  = np.array(alphas_all)    # (T, 14)
weights = np.array(weights_all)   # (T, 15)
sma10   = np.array(sma10_all)     # (T,)
sma50   = np.array(sma50_all)     # (T,)
btc_ret = np.array(btc_return)    # (T,)
T = len(alphas)

print(f"\n  Test steps: {T}  ({T//6} days)")

# ══════════════════════════════════════════════════════════════════════════
# 1. BRAIN 1 — ALPHA STATS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "-"*70)
print("  BRAIN 1 — ALPHA SCORES (Brain 1 raw outputs, range ±1)")
print("-"*70)
print(f"  {'Asset':<8}  {'Mean':>7}  {'Std':>6}  {'Mean|α|':>8}  {'%Long':>6}  {'%Short':>7}  {'%Neutral':>9}")
print(f"  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*9}")
for i, a in enumerate(assets):
    col    = alphas[:, i]
    plong  = float(np.mean(col >  0.05)) * 100
    pshort = float(np.mean(col < -0.05)) * 100
    pneut  = 100 - plong - pshort
    print(f"  {a:<8}  {np.mean(col):>+7.3f}  {np.std(col):>6.3f}  "
          f"{np.mean(np.abs(col)):>8.3f}  {plong:>5.1f}%  {pshort:>6.1f}%  {pneut:>8.1f}%")

mean_conviction = float(np.mean(np.abs(alphas)))
print(f"\n  Overall mean |alpha| (conviction): {mean_conviction:.3f}  "
      f"(v38 was ~0.146, >0.30 is healthy)")

# ══════════════════════════════════════════════════════════════════════════
# 2. BRAIN 2 — PORTFOLIO WEIGHTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "-"*70)
print("  BRAIN 2 — PORTFOLIO WEIGHTS (after Sortino solver)")
print("-"*70)
asset_w = weights[:, :n_assets]
cash_w  = weights[:, n_assets] if weights.shape[1] > n_assets else 1 - np.abs(asset_w).sum(axis=1)

long_book  = np.sum(np.maximum(asset_w, 0), axis=1)   # total long exposure per step
short_book = np.sum(np.abs(np.minimum(asset_w, 0)), axis=1)  # total short exposure

print(f"  Avg cash weight  : {np.mean(cash_w):>+.3f}  (v38 was 0.79)")
print(f"  Avg long book    : {np.mean(long_book):>.3f}  (max 0.80)")
print(f"  Avg short book   : {np.mean(short_book):>.3f}  (max 0.60)")
print(f"  Avg gross exposure: {np.mean(long_book + short_book):>.3f}")
print(f"  Avg net exposure  : {np.mean(long_book - short_book):>+.3f}  (neg = net short)")

# Per-asset avg weight
print(f"\n  {'Asset':<8}  {'AvgWeight':>10}  {'%Steps Long':>12}  {'%Steps Short':>13}")
print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*13}")
for i, a in enumerate(assets):
    col = asset_w[:, i]
    print(f"  {a:<8}  {np.mean(col):>+10.4f}  "
          f"{np.mean(col > 0.001)*100:>11.1f}%  {np.mean(col < -0.001)*100:>12.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# 3. DIRECTIONAL ACCURACY — is the model correctly bearish?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "-"*70)
print("  DIRECTIONAL ACCURACY — alpha direction vs actual BTC move")
print("-"*70)
# When BTC falls (btc_ret < 0), a SHORT (negative alpha) is correct
btc_alpha = alphas[:, btc_idx]
bear_steps = btc_ret < 0
bull_steps = btc_ret > 0

if bear_steps.sum() > 0:
    btc_correct_in_bear = float(np.mean(btc_alpha[bear_steps] < 0)) * 100
    print(f"  BTC falling bars ({bear_steps.sum()}):  BTC alpha negative (correct): {btc_correct_in_bear:.1f}%")
if bull_steps.sum() > 0:
    btc_correct_in_bull = float(np.mean(btc_alpha[bull_steps] > 0)) * 100
    print(f"  BTC rising bars  ({bull_steps.sum()}):  BTC alpha positive (correct): {btc_correct_in_bull:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# 4. REGIME SIGNAL vs ALPHA DIRECTION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "-"*70)
print("  REGIME SIGNALS vs ALPHA ALIGNMENT")
print("-"*70)
bear_sma10 = sma10 < -0.025
bull_sma10 = sma10 >  0.025
bear_sma50 = sma50 < -0.025
bull_sma50 = sma50 >  0.025

print(f"  SMA10 bear steps: {bear_sma10.sum()} ({bear_sma10.mean()*100:.1f}%)  "
      f"bull steps: {bull_sma10.sum()} ({bull_sma10.mean()*100:.1f}%)  "
      f"neutral: {(~bear_sma10&~bull_sma10).sum()}")
print(f"  SMA50 bear steps: {bear_sma50.sum()} ({bear_sma50.mean()*100:.1f}%)  "
      f"bull steps: {bull_sma50.sum()} ({bull_sma50.mean()*100:.1f}%)  "
      f"neutral: {(~bear_sma50&~bull_sma50).sum()}")

# Mean BTC alpha when SMA10 is bearish vs bullish
if bear_sma10.sum() > 0:
    print(f"\n  Mean BTC alpha when SMA10 bearish : {np.mean(btc_alpha[bear_sma10]):>+.3f}  (should be negative)")
if bull_sma10.sum() > 0:
    print(f"  Mean BTC alpha when SMA10 bullish : {np.mean(btc_alpha[bull_sma10]):>+.3f}  (should be positive)")
if bear_sma50.sum() > 0:
    print(f"  Mean BTC alpha when SMA50 bearish : {np.mean(btc_alpha[bear_sma50]):>+.3f}  (should be negative)")
if bull_sma50.sum() > 0:
    print(f"  Mean BTC alpha when SMA50 bullish : {np.mean(btc_alpha[bull_sma50]):>+.3f}  (should be positive)")

# ══════════════════════════════════════════════════════════════════════════
# 5. REWARD COMPONENTS — approximate values from test period
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "-"*70)
print("  REWARD COMPONENTS — estimated magnitudes during test period")
print("-"*70)

# Reconstruct approximate reward components
returns_arr = np.load(test_cfg["returns_path"], mmap_mode='r')  # (n_test, n_assets)
fee_rate = 0.001

# Per-step net portfolio return (approx: sum of weight_i * return_i)
net_returns = []
fees_paid   = []
peak_eq     = INITIAL_CASH
c1_values   = []
c4_values   = []
c6_values   = []

prev_w = np.zeros(n_assets)
for t in range(min(T, len(returns_arr))):
    aw   = asset_w[t]                 # current asset weights
    rets = returns_arr[t]             # asset returns this step

    r_raw  = float(np.dot(aw, rets))
    turnover = float(np.sum(np.abs(aw - prev_w)))
    fee   = turnover * fee_rate
    r     = r_raw - fee
    net_returns.append(r)
    fees_paid.append(fee)
    prev_w = aw.copy()

    # C1
    pv_t = pv_all[t]
    if pv_t < peak_eq:
        dd = (peak_eq - pv_t) / peak_eq
    else:
        peak_eq = pv_t
        dd = 0.0
    c1 = r * 0.35 if (r < 0 and dd > 0.005) else 0.0  # extra 0.35 from 1.35x multiplier
    c1_values.append(c1)

    # C4 (approximate — uses sma10 vs alpha direction)
    sig4 = sma10[t]
    if abs(sig4) > 0.025:
        alph = alphas[t]
        per_asset4 = alph * np.sign(sig4)
        c4 = abs(sig4) * (np.mean(np.maximum(per_asset4,0)) + np.mean(np.minimum(per_asset4,0))) * 0.005
    else:
        c4 = 0.0
    c4_values.append(c4)

    # C6 (approximate — uses sma50 vs actual weights)
    sig6 = sma50[t]
    if abs(sig6) > 0.025:
        per_asset6 = aw * np.sign(sig6)
        c6 = abs(sig6) * (np.mean(np.maximum(per_asset6,0)) + np.mean(np.minimum(per_asset6,0))) * 2.0
    else:
        c6 = 0.0
    c6_values.append(c6)

net_returns = np.array(net_returns)
fees_paid   = np.array(fees_paid)
c1_values   = np.array(c1_values)
c4_values   = np.array(c4_values)
c6_values   = np.array(c6_values)

print(f"  Component            Mean/step       Total (annualised)")
print(f"  {'-'*55}")
print(f"  Base P&L (gross)  : {np.mean(net_returns+fees_paid):>+10.6f}    {np.sum(net_returns+fees_paid)*100:>+8.2f}%")
print(f"  Fees drag         : {np.mean(-fees_paid):>+10.6f}    {np.sum(-fees_paid)*100:>+8.2f}%")
print(f"  Net P&L           : {np.mean(net_returns):>+10.6f}    {np.sum(net_returns)*100:>+8.2f}%")
print(f"  C1 (loss amplify) : {np.mean(c1_values):>+10.6f}    {np.sum(c1_values)*100:>+8.2f}%")
print(f"  C4 (SMA10 alpha)  : {np.mean(c4_values):>+10.6f}    {np.sum(c4_values)*100:>+8.2f}%")
print(f"  C6 (SMA50 weight) : {np.mean(c6_values):>+10.6f}    {np.sum(c6_values)*100:>+8.2f}%")

total_fees_pct = np.sum(fees_paid) * 100
avg_daily_turnover = np.mean(fees_paid) * 6 * 100 / 0.1  # turnover as % of portfolio per day
print(f"\n  Total fee drag    : {total_fees_pct:>+.2f}% (v38 was -12.2%)")
print(f"  Avg daily turnover: {avg_daily_turnover:.1f}% of portfolio")

print("\n" + "-"*70)
print("  CONVICTION OVER TIME (monthly mean |alpha|)")
print("-"*70)
if step_ts:
    ts_pd = pd.to_datetime(step_ts[:T])
    df = pd.DataFrame({
        'conviction': np.mean(np.abs(alphas), axis=1),
        'btc_alpha' : btc_alpha,
        'cash'      : cash_w[:T],
        'sma10'     : sma10[:T],
        'sma50'     : sma50[:T],
    }, index=ts_pd)
    monthly = df.resample('ME').mean()
    print(f"  {'Month':<10}  {'Conviction':>10}  {'BTC alpha':>10}  {'Cash':>6}  {'SMA10':>7}  {'SMA50':>7}")
    for idx, row in monthly.iterrows():
        print(f"  {str(idx.date()):<10}  {row['conviction']:>10.3f}  "
              f"{row['btc_alpha']:>+10.3f}  {row['cash']:>6.3f}  "
              f"{row['sma10']:>+7.3f}  {row['sma50']:>+7.3f}")

print("\n" + "=" * 70)
print("  DIAGNOSTIC COMPLETE")
print("=" * 70)
