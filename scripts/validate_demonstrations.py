#!/usr/bin/env python3
"""Validate dataset schema and episode boundaries."""

import argparse

from robot_learning.demonstrations.dataset import DemonstrationDataset

p = argparse.ArgumentParser()
p.add_argument("dataset", nargs="?", default="experiments/data/expert_demos.npz")
a = p.parse_args()
d = DemonstrationDataset.load(a.dataset)
print(
    f"VALID schema=1 episodes={d.n_episodes} transitions={d.n_transitions} expert_success_rate={d.episode_success.mean():.3f} obs_dim={d.observations.shape[1]} action_dim={d.actions.shape[1]}"
)
