"""Simulation policy action validation and safe-stop behavior."""

from __future__ import annotations

import numpy as np


class InvalidActionError(ValueError):
    """Raised when a policy action cannot be safely executed."""


def validate_action(action: np.ndarray, *, dimension: int = 4, limit: float = 1.0) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (dimension,):
        raise InvalidActionError(f"expected action shape ({dimension},), got {value.shape}")
    if not np.isfinite(value).all():
        raise InvalidActionError("action contains NaN or infinite values")
    return np.clip(value, -limit, limit)


def safe_stop_action(dimension: int = 4) -> np.ndarray:
    """Stop Cartesian movement and open the gripper for the Fetch interface."""
    if dimension != 4:
        return np.zeros(dimension, dtype=np.float32)
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
