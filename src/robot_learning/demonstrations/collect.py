"""Generate expert demonstrations from live MuJoCo episodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_learning.controllers.scripted_expert import ScriptedPickPlaceExpert
from robot_learning.demonstrations.dataset import DemonstrationDataset
from robot_learning.environments.fetch_pick_place import make_env


def collect_demonstrations(episodes: int, seed: int, output: str | Path, max_steps: int = 50) -> DemonstrationDataset:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    obs_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    offsets = [0]
    outcomes: list[bool] = []
    seeds: list[int] = []
    env = make_env()
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            observation, _ = env.reset(seed=episode_seed)
            expert = ScriptedPickPlaceExpert()
            success = False
            for _ in range(max_steps):
                action = expert.act(observation)
                obs_rows.append(observation.copy())
                action_rows.append(action.copy())
                observation, _, terminated, truncated, info = env.step(action)
                success = success or bool(info.get("is_success", False))
                if success or terminated or truncated:
                    break
            if len(obs_rows) == offsets[-1]:
                raise RuntimeError(f"episode {episode} generated no transitions")
            offsets.append(len(obs_rows))
            outcomes.append(success)
            seeds.append(episode_seed)
    finally:
        env.close()
    dataset = DemonstrationDataset(
        np.stack(obs_rows), np.stack(action_rows), np.asarray(offsets), np.asarray(outcomes), np.asarray(seeds)
    )
    dataset.save(output)
    return dataset
