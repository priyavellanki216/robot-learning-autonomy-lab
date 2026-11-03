# Environment contract

The environment wraps Gymnasium Robotics `FetchPickAndPlace-v4`, a simulated Fetch arm, parallel-jaw gripper, object, table, target, and MuJoCo contact dynamics. It does not use a predetermined success value or kinematic teleportation. Each transition is delegated to the real simulator.

- **Observation:** `float32[28]`, concatenating the source Fetch 25D observation and desired target XYZ (3D). The Fetch vector includes gripper XYZ, object XYZ, object-relative XYZ, gripper state, object rotation, object and gripper velocities.
- **Action:** `float32[4]` bounded by `[-1,1]`: normalized Cartesian gripper delta XYZ and gripper open/close command. The simulator maps the control to its mocap/gripper actuation.
- **Reward:** Gymnasium Robotics dense pick-and-place distance reward; success is the environment's actual goal-distance criterion.
- **Episode:** wrapper preserves Gymnasium `(observation, reward, terminated, truncated, info)` semantics; training scripts use an explicit 50-step experiment horizon.
- **Randomization:** Gymnasium Robotics randomizes object/goal starts on reset under its native reset implementation. Fixed seeds enable comparable episodes.

A compatibility shim converts MuJoCo's NumPy integer joint enum scalar to `int` for Gymnasium Robotics utility membership checks. This was required by the exact tested package combination; it does not change simulation dynamics.
