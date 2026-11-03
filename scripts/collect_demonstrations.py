#!/usr/bin/env python3
"""Collect scripted expert demonstrations from MuJoCo Fetch."""

import argparse

from robot_learning.demonstrations.collect import collect_demonstrations

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=40)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--max-steps", type=int, default=50)
p.add_argument("--output", default="experiments/data/expert_demos.npz")
a = p.parse_args()
data = collect_demonstrations(a.episodes, a.seed, a.output, a.max_steps)
print(
    f"generated episodes={data.n_episodes} transitions={data.n_transitions} successes={int(data.episode_success.sum())}/{data.n_episodes} path={a.output}"
)
