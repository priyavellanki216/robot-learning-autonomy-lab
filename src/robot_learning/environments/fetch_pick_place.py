"""Gymnasium-compatible Fetch pick-and-place task adapter."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from robot_learning.simulation.mujoco_compat import patch_gymnasium_robotics_joint_helpers


class FetchPickPlaceEnv(gym.Wrapper):
    """Flatten Fetch's goal dictionary to 28 float features.

    Feature order is Fetch observation (25 values) followed by desired goal XYZ.
    The underlying environment retains genuine MuJoCo contact and grasp physics.
    """

    def __init__(self, *, render_mode: str | None = None, reward_type: str = "dense") -> None:
        import gymnasium_robotics  # noqa: F401 -- registers Fetch task IDs

        patch_gymnasium_robotics_joint_helpers()
        base = gym.make("FetchPickAndPlace-v4", reward_type=reward_type, render_mode=render_mode)
        super().__init__(base)
        low = np.concatenate([base.observation_space["observation"].low, base.observation_space["desired_goal"].low])
        high = np.concatenate([base.observation_space["observation"].high, base.observation_space["desired_goal"].high])
        self.observation_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)
        self.action_space = base.action_space

    @staticmethod
    def flatten(observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate((observation["observation"], observation["desired_goal"])).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_count = 0
        observation, info = self.env.reset(seed=seed, options=options)
        return self.flatten(observation), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        info = dict(info)
        info["is_success"] = bool(info.get("is_success", False))
        info["episode_step"] = self._step_count
        return self.flatten(observation), float(reward), bool(terminated), bool(truncated), info


def make_env(*, render_mode: str | None = None, reward_type: str = "dense") -> FetchPickPlaceEnv:
    """Construct the project's standard simulation environment."""
    return FetchPickPlaceEnv(render_mode=render_mode, reward_type=reward_type)
