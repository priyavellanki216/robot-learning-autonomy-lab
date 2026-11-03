#!/usr/bin/env python3
"""Plot actual evaluation JSON; refuses to invent data."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--input", default="experiments/results/evaluation.json")
p.add_argument("--output", default="experiments/plots/policy_comparison.png")
a = p.parse_args()
data = json.loads(Path(a.input).read_text())
if not data:
    raise SystemExit("No measured results found")
labels = [x["policy"] for x in data]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, key, title in zip(
    axes, ["success_rate", "mean_reward", "mean_episode_length"], ["Success rate", "Mean return", "Mean episode length"]
):
    ax.bar(labels, [x[key] for x in data], color=["#3478b8", "#d9822b", "#4b9b68"][: len(data)])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=160)
print(a.output)
