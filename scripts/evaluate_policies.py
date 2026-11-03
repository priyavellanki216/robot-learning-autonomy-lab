#!/usr/bin/env python3
"""Evaluate learned policies and expert baseline on identical seeds."""

import argparse

from robot_learning.controllers.scripted_expert import ScriptedPickPlaceExpert
from robot_learning.evaluation.evaluate import evaluate_policy, save_evaluations
from robot_learning.imitation_learning.behavior_cloning import BehaviorCloningPolicy
from robot_learning.reinforcement_learning.ppo import load_ppo_policy

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--seed-start", type=int, default=1000)
p.add_argument("--max-steps", type=int, default=50)
p.add_argument("--observation-noise-std", type=float, default=0.0)
p.add_argument("--bc-checkpoint", default="experiments/checkpoints/bc_policy.pt")
p.add_argument("--ppo-checkpoint", default="experiments/checkpoints/ppo_policy.zip")
p.add_argument("--output", default="experiments/results/evaluation.json")
a = p.parse_args()
seeds = list(range(a.seed_start, a.seed_start + a.episodes))
policies = [("scripted_baseline", ScriptedPickPlaceExpert())]
try:
    policies.append(("behavior_cloning", BehaviorCloningPolicy(a.bc_checkpoint)))
except FileNotFoundError:
    print("Skipping BC: checkpoint is absent")
try:
    policies.append(("ppo", load_ppo_policy(a.ppo_checkpoint)))
except FileNotFoundError:
    print("Skipping PPO: checkpoint is absent")
results = []
for name, policy in policies:
    if hasattr(policy, "reset"):
        policy.reset()
    result = evaluate_policy(
        policy,
        seeds,
        label=name,
        max_steps=a.max_steps,
        observation_noise_std=a.observation_noise_std,
    )
    results.append(result)
    print(name, "success_rate=", result["success_rate"], "mean_reward=", result["mean_reward"])
save_evaluations(results, a.output)
print("saved", a.output)
