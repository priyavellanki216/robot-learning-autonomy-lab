# Reward and controller design

PPO uses the upstream `FetchPickAndPlace-v4` dense reward: a negative distance between the achieved object position and target, with the environment's goal-success semantics. No extra reward terms are added, reducing the chance of reward hacking through hand-authored shaping. SB3 receives the genuine simulator reward on every transition.

The scripted baseline and demonstration expert use interpretable waypoint phases: move above object, descend, close gripper, lift, transport toward target while preserving measured object-to-gripper offset, lower, and release. The expert is deterministic given the observations but the simulation's resets are seeded and randomized. It is not an oracle policy and does not assign success; `info["is_success"]` comes from the environment.

This setup does not separately expose collision counts or contact-force thresholds. The experiment reports those as unavailable rather than estimating them from unrelated signals. Drop is a diagnostic inferred from an object that had first lifted and then returned near its initial height without task success; it is not a simulator-provided event label.
