from __future__ import annotations

import numpy as np
import pytest

from robot_learning.controllers.scripted_expert import ScriptedPickPlaceExpert
from robot_learning.demonstrations.collect import collect_demonstrations
from robot_learning.demonstrations.dataset import DemonstrationDataset
from robot_learning.environments.fetch_pick_place import make_env
from robot_learning.evaluation.evaluate import evaluate_policy
from robot_learning.imitation_learning.behavior_cloning import BCNetwork, BehaviorCloningPolicy, train_behavior_cloning
from robot_learning.reinforcement_learning.ppo import load_ppo_policy, train_ppo
from robot_learning.safety.action_validator import InvalidActionError, safe_stop_action, validate_action


def test_environment_reset_step_and_seed_reproducibility():
    env = make_env()
    try:
        obs_a, _ = env.reset(seed=31)
        obs_b, _ = env.reset(seed=31)
        assert env.observation_space.shape == (28,)
        assert env.action_space.shape == (4,)
        assert env.observation_space.contains(obs_a)
        np.testing.assert_allclose(obs_a, obs_b)
        obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "is_success" in info
    finally:
        env.close()


def test_action_validator_clips_and_rejects_bad_values():
    np.testing.assert_allclose(validate_action(np.array([2, -2, 0.3, 0.0])), [1, -1, 0.3, 0])
    np.testing.assert_array_equal(safe_stop_action(), [0, 0, 0, 1])
    with pytest.raises(InvalidActionError):
        validate_action(np.zeros(3))
    with pytest.raises(InvalidActionError):
        validate_action(np.array([0, 0, np.nan, 0]))


def test_demonstration_collection_and_roundtrip(tmp_path):
    path = tmp_path / "demos.npz"
    data = collect_demonstrations(4, 71, path, max_steps=50)
    loaded = DemonstrationDataset.load(path)
    loaded.validate()
    assert loaded.n_episodes == 4
    assert loaded.n_transitions > 0
    assert loaded.observations.shape[1] == 28
    assert loaded.actions.shape[1] == 4
    assert data.episode_offsets[-1] == data.n_transitions


def test_corrupted_dataset_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        observations=np.zeros((4, 27)),
        actions=np.zeros((4, 4)),
        episode_offsets=np.array([0, 4]),
        episode_success=np.array([True]),
        seeds=np.array([1]),
        schema_version=np.array([1]),
    )
    with pytest.raises(ValueError):
        DemonstrationDataset.load(path)


def test_behavior_cloning_forward_train_save_load(tmp_path):
    data = collect_demonstrations(4, 101, tmp_path / "demos.npz")
    model = BCNetwork()
    assert model(__import__("torch").zeros(2, 28)).shape == (2, 4)
    checkpoint = tmp_path / "bc.pt"
    result = train_behavior_cloning(data, checkpoint, epochs=3, seed=2)
    policy = BehaviorCloningPolicy(checkpoint)
    action = policy.predict(data.observations[0])
    assert checkpoint.exists()
    assert np.isfinite(result.validation_loss)
    assert action.shape == (4,) and np.all(np.abs(action) <= 1)


def test_scripted_baseline_executes_real_task():
    result = evaluate_policy(
        ScriptedPickPlaceExpert(), [2, 7], label="baseline", max_steps=50, observation_noise_std=0.01
    )
    assert result["episodes"] == 2
    assert result["observation_noise_std"] == 0.01
    assert 0.0 <= result["success_rate"] <= 1.0
    assert result["mean_episode_length"] > 0
    assert len(result["episodes_detail"]) == 2


def test_ppo_training_checkpoint_and_inference(tmp_path):
    checkpoint = train_ppo(tmp_path / "ppo_smoke", timesteps=128, n_steps=128, batch_size=32, seed=5)
    policy = load_ppo_policy(checkpoint)
    action = policy(np.zeros(28, dtype=np.float32))
    assert checkpoint.exists()
    assert action.shape == (4,)
    assert np.isfinite(action).all()
