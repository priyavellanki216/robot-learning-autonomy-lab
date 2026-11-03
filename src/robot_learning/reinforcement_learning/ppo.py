"""Stable-Baselines3 PPO training helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from robot_learning.environments.fetch_pick_place import make_env


def train_ppo(
    checkpoint_prefix: str | Path, *, timesteps: int = 20_000, seed: int = 7, n_steps: int = 256, batch_size: int = 64
) -> Path:
    if timesteps < n_steps or n_steps < 2 or batch_size < 2:
        raise ValueError("timesteps must be >= n_steps >= 2 and batch_size >= 2")
    env = make_env()
    try:
        observation, _ = env.reset(seed=seed)
        if not env.observation_space.contains(observation):
            raise ValueError("environment reset observation is outside its declared space")
        next_observation, reward, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        if not env.observation_space.contains(next_observation) or not np.isfinite(reward):
            raise ValueError("environment smoke step returned invalid observation/reward")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise TypeError("Gymnasium step must return boolean termination flags")
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=3e-4,
            gamma=0.98,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=1,
            device="cpu",
        )
        model.learn(total_timesteps=timesteps, progress_bar=False)
        path = Path(checkpoint_prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        return path.with_suffix(".zip")
    finally:
        env.close()


def load_ppo_policy(checkpoint: str | Path):
    model = PPO.load(str(checkpoint), device="cpu")
    return lambda observation: model.predict(observation, deterministic=True)[0]
