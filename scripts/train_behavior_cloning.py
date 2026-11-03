#!/usr/bin/env python3
"""Train a PyTorch behavior cloning policy."""

import argparse
import json
from dataclasses import asdict

from robot_learning.demonstrations.dataset import DemonstrationDataset
from robot_learning.imitation_learning.behavior_cloning import train_behavior_cloning

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="experiments/data/expert_demos.npz")
p.add_argument("--checkpoint", default="experiments/checkpoints/bc_policy.pt")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch-size", type=int, default=128)
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()
r = train_behavior_cloning(
    DemonstrationDataset.load(a.dataset), a.checkpoint, epochs=a.epochs, batch_size=a.batch_size, seed=a.seed
)
print(json.dumps({**asdict(r), "checkpoint": a.checkpoint}, indent=2))
