"""Common-seed evaluation for scripted and learned policies."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from robot_learning.environments.fetch_pick_place import make_env
from robot_learning.safety.action_validator import InvalidActionError, safe_stop_action, validate_action

Policy = Callable[[np.ndarray], np.ndarray]


def evaluate_policy(
    policy: Policy,
    seeds: list[int],
    *,
    label: str,
    max_steps: int = 50,
    observation_noise_std: float = 0.0,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    if observation_noise_std < 0:
        raise ValueError("observation_noise_std cannot be negative")
    env = make_env()
    rows: list[dict[str, object]] = []
    try:
        for seed in seeds:
            if hasattr(policy, "reset"):
                policy.reset()  # type: ignore[attr-defined]
            obs, _ = env.reset(seed=seed)
            noise_rng = np.random.default_rng(seed + 10_000)
            episode_reward = 0.0
            success = False
            started = time.perf_counter()
            invalid_actions = 0
            initial_obj_z = float(obs[5])
            grasped = False
            dropped = False
            for step in range(1, max_steps + 1):
                try:
                    policy_obs = obs
                    if observation_noise_std:
                        policy_obs = obs + noise_rng.normal(0.0, observation_noise_std, size=obs.shape).astype(
                            np.float32
                        )
                    if hasattr(policy, "act"):
                        raw_action = policy.act(policy_obs)  # type: ignore[attr-defined]
                    elif hasattr(policy, "predict"):
                        raw_action = policy.predict(policy_obs)  # type: ignore[attr-defined]
                    else:
                        raw_action = policy(policy_obs)
                    action = validate_action(raw_action)
                except (InvalidActionError, ValueError, FloatingPointError):
                    invalid_actions += 1
                    action = safe_stop_action()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                success = success or bool(info.get("is_success", False))
                if obs[5] > initial_obj_z + 0.04:
                    grasped = True
                if grasped and obs[5] < initial_obj_z + 0.015 and not success:
                    dropped = True
                if success or terminated or truncated:
                    break
            rows.append(
                {
                    "seed": seed,
                    "success": success,
                    "steps": step,
                    "elapsed_seconds": time.perf_counter() - started,
                    "return": episode_reward,
                    "grasped": grasped,
                    "dropped": dropped,
                    "invalid_actions": invalid_actions,
                    "failure_category": "success"
                    if success
                    else ("invalid_action" if invalid_actions else "drop" if dropped else "not_placed"),
                }
            )
    finally:
        env.close()
    result: dict[str, object] = {
        "policy": label,
        "observation_noise_std": observation_noise_std,
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_completion_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])),
        "mean_episode_length": float(np.mean([r["steps"] for r in rows])),
        "grasp_success_rate": float(np.mean([r["grasped"] for r in rows])),
        "object_drop_rate": float(np.mean([r["dropped"] for r in rows])),
        "mean_reward": float(np.mean([r["return"] for r in rows])),
        "invalid_action_episodes": int(sum(int(r["invalid_actions"] > 0) for r in rows)),
        "failure_categories": {
            category: sum(r["failure_category"] == category for r in rows)
            for category in sorted({str(r["failure_category"]) for r in rows})
        },
        "episodes_detail": rows,
    }
    return result


def save_evaluations(results: list[dict[str, object]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
