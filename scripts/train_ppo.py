#!/usr/bin/env python3
"""Train a PPO policy through real Gymnasium/MuJoCo interactions."""

import argparse

from robot_learning.reinforcement_learning.ppo import train_ppo

p = argparse.ArgumentParser()
p.add_argument("--timesteps", type=int, default=20000)
p.add_argument("--n-steps", type=int, default=256)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--checkpoint-prefix", default="experiments/checkpoints/ppo_policy")
a = p.parse_args()
path = train_ppo(a.checkpoint_prefix, timesteps=a.timesteps, seed=a.seed, n_steps=a.n_steps, batch_size=a.batch_size)
print(f"PPO checkpoint: {path}; actual_timesteps={a.timesteps}; device=cpu")
