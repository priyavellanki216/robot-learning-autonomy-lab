"""Behavior Cloning for continuous Fetch actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from robot_learning.demonstrations.dataset import DemonstrationDataset


class BCNetwork(nn.Module):
    def __init__(self, obs_dim: int = 28, action_dim: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


@dataclass(frozen=True)
class BCTrainingResult:
    train_loss: float
    validation_loss: float
    epochs: int
    train_samples: int
    validation_samples: int
    demonstrations: int


def _split_indices(
    dataset: DemonstrationDataset, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0 or dataset.n_episodes < 2:
        raise ValueError("need at least two episodes and a validation fraction between 0 and 1")
    rng = np.random.default_rng(seed)
    episode_order = rng.permutation(dataset.n_episodes)
    n_val = max(1, int(round(dataset.n_episodes * validation_fraction)))
    n_val = min(n_val, dataset.n_episodes - 1)
    val_eps = set(episode_order[:n_val].tolist())
    train_parts, val_parts = [], []
    for episode, (start, end) in enumerate(zip(dataset.episode_offsets[:-1], dataset.episode_offsets[1:])):
        (val_parts if episode in val_eps else train_parts).append(np.arange(start, end))
    return np.concatenate(train_parts), np.concatenate(val_parts)


def train_behavior_cloning(
    dataset: DemonstrationDataset,
    checkpoint: str | Path,
    *,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 7,
    validation_fraction: float = 0.2,
) -> BCTrainingResult:
    dataset.validate()
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    train_idx, val_idx = _split_indices(dataset, validation_fraction, seed)
    mean = dataset.observations[train_idx].mean(axis=0).astype(np.float32)
    std = np.maximum(dataset.observations[train_idx].std(axis=0), 1e-6).astype(np.float32)
    x = (dataset.observations - mean) / std
    train_ds = TensorDataset(
        torch.from_numpy(x[train_idx]), torch.from_numpy(dataset.actions[train_idx].astype(np.float32))
    )
    val_x = torch.from_numpy(x[val_idx])
    val_y = torch.from_numpy(dataset.actions[val_idx].astype(np.float32))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = BCNetwork(dataset.observations.shape[1], dataset.actions.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    final_train = float("nan")
    for _ in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        final_train = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y))
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    target = Path(checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_mean": torch.from_numpy(mean),
            "obs_std": torch.from_numpy(std),
            "obs_dim": len(mean),
            "action_dim": dataset.actions.shape[1],
            "metadata": asdict(
                BCTrainingResult(final_train, best, epochs, len(train_idx), len(val_idx), dataset.n_episodes)
            ),
        },
        target,
    )
    return BCTrainingResult(final_train, best, epochs, len(train_idx), len(val_idx), dataset.n_episodes)


class BehaviorCloningPolicy:
    def __init__(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.mean = np.asarray(payload["obs_mean"], dtype=np.float32)
        self.std = np.asarray(payload["obs_std"], dtype=np.float32)
        self.model = BCNetwork(int(payload["obs_dim"]), int(payload["action_dim"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def predict(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != self.mean.shape or not np.isfinite(observation).all():
            raise ValueError("invalid observation supplied to Behavior Cloning policy")
        with torch.no_grad():
            action = self.model(torch.from_numpy(((observation - self.mean) / self.std).astype(np.float32))).numpy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)
