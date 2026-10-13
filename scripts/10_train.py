"""
Train the signal-generation layer (recurrent PPO) on the training period.

    python scripts/10_train.py --seed 7
    python scripts/10_train.py --seed 42

The training data end on TRAIN_END; every time-indexed array is physically truncated
there before any environment is built, and the scaling ranges of the inputs are fitted
on the same rows. Output: models/<name>/final_model.zip, vec_normalize.pkl and
holdout.json (the split and settings used by scripts/11_backtest.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hdrl.data import prepare_data, slice_data   # noqa: E402
from hdrl.env import TradingEnv                   # noqa: E402

DATA_FILE = "crypto_deep_v6.parquet"
TRAIN_END = "2025-06-24 20:00:00"   # last training bar; the test year starts at the next bar
CLIP_PCT = 0.5                      # scaling ranges = 0.5th / 99.5th percentiles
HYPER = dict(learning_rate=3e-5, n_steps=512, batch_size=512, n_epochs=4, gamma=0.99,
             gae_lambda=0.95, clip_range=0.2, ent_coef=0.015, vf_coef=0.5, max_grad_norm=0.5)
LOG_STD_INIT = -1.0                 # initial exploration noise std exp(-1) = 0.37
N_ENVS = 4


def load_training_data(data_path: Path):
    data = prepare_data(data_path, fit_end=TRAIN_END, clip_pct=CLIP_PCT)
    ts = pd.to_datetime(pd.Series(data['timestamps']))
    split_at = int((ts <= pd.Timestamp(TRAIN_END)).sum())
    assert str(pd.Timestamp(data['timestamps'][split_at - 1])) == TRAIN_END
    train = slice_data(data, 0, split_at)
    assert pd.Timestamp(train['timestamps'][-1]) < pd.Timestamp(data['timestamps'][split_at])
    return data, train, split_at


def build_model(train: dict, seed: int, tensorboard_dir=None):
    env = make_vec_env(lambda: TradingEnv(train, training=True), n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=True)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=torch.nn.ReLU,
                         lstm_hidden_size=128, n_lstm_layers=1, enable_critic_lstm=False,
                         log_std_init=LOG_STD_INIT)
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, device="cpu", policy_kwargs=policy_kwargs,
                         seed=seed, tensorboard_log=tensorboard_dir, **HYPER)
    return model, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=1_300_000)
    ap.add_argument("--name", default=None, help="output folder under models/ (default v50_s<seed>)")
    ap.add_argument("--checkpoint-every", type=int, default=100_000)
    args = ap.parse_args()

    model_dir = ROOT / "models" / (args.name or f"v50_s{args.seed}")
    model_dir.mkdir(parents=True, exist_ok=True)

    data, train, split_at = load_training_data(ROOT / "data" / "processed" / DATA_FILE)
    print(f"Training bars: {train['timestamps'][0]} -> {train['timestamps'][-1]} ({split_at:,}); "
          f"held out from {data['timestamps'][split_at]} ({data['n_steps'] - split_at:,} bars)")
    (model_dir / "holdout.json").write_text(json.dumps({
        "train_end": str(pd.Timestamp(data['timestamps'][split_at - 1])),
        "test_start": str(pd.Timestamp(data['timestamps'][split_at])),
        "n_steps_full": int(data['n_steps']), "split_at": split_at,
        "data_file": DATA_FILE, "clip_pct": CLIP_PCT, "seed": args.seed, "total_steps": args.steps,
        "hyperparameters": {**HYPER, "log_std_init": LOG_STD_INIT, "n_envs": N_ENVS},
    }, indent=2))

    try:
        import tensorboard  # noqa: F401
        tb = str(ROOT / "logs" / model_dir.name)
    except ImportError:
        tb = None
    model, env = build_model(train, args.seed, tb)
    ckpt = CheckpointCallback(save_freq=args.checkpoint_every, save_path=str(model_dir), name_prefix="ckpt")

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.steps, callback=[ckpt], progress_bar=False)
    finally:
        model.save(str(model_dir / "final_model"))
        env.save(str(model_dir / "vec_normalize.pkl"))
        print(f"Saved {model_dir} after {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
