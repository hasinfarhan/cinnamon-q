"""Compare v7 vs v8 training dynamics in detail."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def read_tb(path):
    ea = EventAccumulator(path); ea.Reload()
    return {tag: np.array([(e.step, e.value) for e in ea.Scalars(tag)]) for tag in ea.Tags()['scalars']}

v7 = read_tb(r'logs/hdrl_training_worldclass_v7/tensorboard/RecurrentPPO_2')
v8 = read_tb(r'logs/hdrl_training_worldclass_v8/tensorboard/RecurrentPPO_1')

print('=== ENTROPY COLLAPSE SPEED (more negative = more deterministic/collapsed) ===')
for label, d in [('V7', v7), ('V8', v8)]:
    ent = d['train/entropy_loss'][:, 1]
    steps = d['train/entropy_loss'][:, 0]
    chunks = np.array_split(ent, 5)
    chunk_steps = np.array_split(steps, 5)
    parts = [f"step{int(chunk_steps[i].mean()/1000)}k={chunks[i].mean():.1f}" for i in range(5)]
    print(f'  {label}: {" | ".join(parts)}')

print()
print('=== EXPLAINED VARIANCE JOURNEY (1.0=perfect value fn, 0.0=random guess) ===')
for label, d in [('V7', v7), ('V8', v8)]:
    ev = d['train/explained_variance'][:, 1]
    chunks = np.array_split(ev, 5)
    parts = [f"{c.mean():.3f}" for c in chunks]
    print(f'  {label}: {" -> ".join(parts)}')

print()
print('=== EPISODE REWARD MEAN ===')
for label, d in [('V7', v7), ('V8', v8)]:
    rew = d['rollout/ep_rew_mean'][:, 1]
    chunks = np.array_split(rew, 5)
    parts = [f"{c.mean():.0f}" for c in chunks]
    print(f'  {label}: {" -> ".join(parts)}')

print()
print('=== POLICY GRADIENT LOSS (should be non-zero to show learning) ===')
for label, d in [('V7', v7), ('V8', v8)]:
    pg = d['train/policy_gradient_loss'][:, 1]
    chunks = np.array_split(pg, 5)
    parts = [f"{c.mean():.5f}" for c in chunks]
    print(f'  {label}: {" -> ".join(parts)}')

print()
print('=== KEY FINDING SUMMARY ===')
v7_ev_end = v7['train/explained_variance'][:, 1][-100:].mean()
v8_ev_end = v8['train/explained_variance'][:, 1][-100:].mean()
v7_ent_end = v7['train/entropy_loss'][:, 1][-100:].mean()
v8_ent_end = v8['train/entropy_loss'][:, 1][-100:].mean()
v7_rew_end = v7['rollout/ep_rew_mean'][:, 1][-100:].mean()
v8_rew_end = v8['rollout/ep_rew_mean'][:, 1][-100:].mean()
print(f'  Explained Variance (end): V7={v7_ev_end:.3f}  V8={v8_ev_end:.3f}  <- V7 value fn is MUCH better')
print(f'  Entropy Loss (end)      : V7={v7_ent_end:.2f}  V8={v8_ent_end:.2f}  <- both collapsed equally')
print(f'  Ep Reward Mean (end)    : V7={v7_rew_end:.1f}  V8={v8_rew_end:.1f}')
