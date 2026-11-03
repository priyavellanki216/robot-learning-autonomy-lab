"""Deterministic waypoint expert for generating real simulator demonstrations."""

from __future__ import annotations

import numpy as np


class ScriptedPickPlaceExpert:
    """A simple Cartesian finite-state controller, not a learned policy."""

    def __init__(self, action_scale: float = 0.05) -> None:
        self.action_scale = float(action_scale)
        self.reset()

    def reset(self) -> None:
        self.phase = "approach"
        self.phase_steps = 0
        self.initial_object_z: float | None = None

    def _move(self, grip: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        delta = np.clip((target - grip) / self.action_scale, -1.0, 1.0)
        return np.asarray([delta[0], delta[1], delta[2], gripper], dtype=np.float32)

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (28,) or not np.isfinite(obs).all():
            raise ValueError(f"Expected finite 28D observation, received {obs.shape}")
        grip, obj, goal = obs[:3], obs[3:6], obs[25:28]
        if self.initial_object_z is None:
            self.initial_object_z = float(obj[2])
        self.phase_steps += 1

        if self.phase == "approach":
            target = obj + np.array([0.0, 0.0, 0.12])
            if np.linalg.norm(grip[:2] - obj[:2]) < 0.025 and abs(grip[2] - target[2]) < 0.025:
                self.phase = "lower"
                self.phase_steps = 0
            return self._move(grip, target, 1.0)

        if self.phase == "lower":
            target = obj + np.array([0.0, 0.0, 0.005])
            if np.linalg.norm(grip - target) < 0.018:
                self.phase = "close"
                self.phase_steps = 0
            return self._move(grip, target, 1.0)

        if self.phase == "close":
            self.phase = "lift" if self.phase_steps >= 12 else "close"
            if self.phase != "close":
                self.phase_steps = 0
            return self._move(grip, grip, -1.0)

        if self.phase == "lift":
            target = obj + np.array([0.0, 0.0, 0.16])
            lifted = obj[2] > float(self.initial_object_z) + 0.045
            if lifted or self.phase_steps >= 30:
                self.phase = "transport"
                self.phase_steps = 0
            return self._move(grip, target, -1.0)

        if self.phase == "transport":
            # Track the measured object-to-gripper offset while carrying.
            target_grip = grip + (goal - obj) + np.array([0.0, 0.0, 0.10])
            if np.linalg.norm(obj[:2] - goal[:2]) < 0.035 or self.phase_steps >= 60:
                self.phase = "lower_place"
                self.phase_steps = 0
            return self._move(grip, target_grip, -1.0)

        if self.phase == "lower_place":
            target = goal + np.array([0.0, 0.0, 0.035])
            if np.linalg.norm(grip[:2] - goal[:2]) < 0.03 and abs(grip[2] - target[2]) < 0.025:
                self.phase = "release"
                self.phase_steps = 0
            return self._move(grip, target, -1.0)

        if self.phase == "release":
            if self.phase_steps >= 12:
                self.phase = "done"
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
