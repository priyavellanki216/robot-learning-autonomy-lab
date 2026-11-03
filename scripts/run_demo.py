#!/usr/bin/env python3
"""Run one policy episode with optional human rendering."""

import argparse

from robot_learning.controllers.scripted_expert import ScriptedPickPlaceExpert
from robot_learning.environments.fetch_pick_place import make_env
from robot_learning.imitation_learning.behavior_cloning import BehaviorCloningPolicy
from robot_learning.reinforcement_learning.ppo import load_ppo_policy
from robot_learning.safety.action_validator import safe_stop_action, validate_action

p = argparse.ArgumentParser()
p.add_argument("--policy", choices=["expert", "bc", "ppo"], default="expert")
p.add_argument("--checkpoint", default=None)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--render", action="store_true")
a = p.parse_args()
if a.policy == "expert":
    policy = ScriptedPickPlaceExpert()
elif a.policy == "bc":
    policy = BehaviorCloningPolicy(a.checkpoint or "experiments/checkpoints/bc_policy.pt")
else:
    policy = load_ppo_policy(a.checkpoint or "experiments/checkpoints/ppo_policy.zip")
env = make_env(render_mode="human" if a.render else None)
obs, _ = env.reset(seed=a.seed)
success = False
try:
    for step in range(1, 51):
        try:
            action = validate_action(
                policy.act(obs)
                if hasattr(policy, "act")
                else policy.predict(obs)
                if hasattr(policy, "predict")
                else policy(obs)
            )
        except (ValueError, FloatingPointError):
            action = safe_stop_action()
        obs, reward, term, trunc, info = env.step(action)
        success |= bool(info.get("is_success", False))
        if success or term or trunc:
            break
    print(f"policy={a.policy} seed={a.seed} success={success} steps={step} reward={reward:.3f}")
finally:
    env.close()
