"""Episode-aware demonstration serialization and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

OBS_DIM = 28
ACTION_DIM = 4


@dataclass(frozen=True)
class DemonstrationDataset:
    observations: np.ndarray
    actions: np.ndarray
    episode_offsets: np.ndarray
    episode_success: np.ndarray
    seeds: np.ndarray

    @property
    def n_episodes(self) -> int:
        return int(len(self.episode_offsets) - 1)

    @property
    def n_transitions(self) -> int:
        return int(len(self.observations))

    def validate(self) -> None:
        obs = np.asarray(self.observations)
        actions = np.asarray(self.actions)
        offsets = np.asarray(self.episode_offsets)
        success = np.asarray(self.episode_success)
        seeds = np.asarray(self.seeds)
        if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
            raise ValueError(f"observations must have shape (N,{OBS_DIM}), got {obs.shape}")
        if actions.shape != (len(obs), ACTION_DIM):
            raise ValueError(f"actions must have shape ({len(obs)},{ACTION_DIM}), got {actions.shape}")
        if not np.isfinite(obs).all() or not np.isfinite(actions).all():
            raise ValueError("dataset contains non-finite observations/actions")
        if np.any(np.abs(actions) > 1.0001):
            raise ValueError("actions exceed the normalized [-1, 1] bounds")
        if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(obs):
            raise ValueError("episode_offsets must start at 0 and end at the transition count")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("empty or invalid episodes are not allowed")
        if success.shape != (len(offsets) - 1,) or seeds.shape != success.shape:
            raise ValueError("episode labels and seeds must have one entry per episode")

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            observations=self.observations.astype(np.float32),
            actions=self.actions.astype(np.float32),
            episode_offsets=self.episode_offsets.astype(np.int64),
            episode_success=self.episode_success.astype(bool),
            seeds=self.seeds.astype(np.int64),
            schema_version=np.array([1], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str | Path) -> DemonstrationDataset:
        with np.load(path, allow_pickle=False) as data:
            required = {"observations", "actions", "episode_offsets", "episode_success", "seeds", "schema_version"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"missing dataset fields: {sorted(missing)}")
            if int(data["schema_version"][0]) != 1:
                raise ValueError("unsupported demonstration schema version")
            dataset = cls(
                *(data[k].copy() for k in ("observations", "actions", "episode_offsets", "episode_success", "seeds"))
            )
        dataset.validate()
        return dataset
