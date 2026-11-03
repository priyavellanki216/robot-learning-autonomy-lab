# Known limitations

- This is a simulation-only system using the Gymnasium Robotics Fetch arm model. No physical robot was available or controlled.
- The sandbox exposed six CPU cores and roughly 8 GiB RAM but no NVIDIA GPU/CUDA device. Training is CPU-only.
- ROS 2 and Gazebo were absent. The ROS 2 package is source-only, unbuilt and untested.
- Display variables exist, but interactive rendering was not verified; use the headless path by default.
- The scripted expert relies on a small hand-authored finite-state controller. It is a reproducible baseline, not a general manipulation planner.
- PPO with a small feasible compute budget is a smoke/research run, not evidence of convergence. A low learned success rate is a valid result.
- Collision events and robust contact-force/safety envelopes are not separately measured. No physical safety or sim-to-real claim is made.
- The MuJoCo enum workaround is scoped to Gymnasium Robotics joint helper comparisons and is covered by environment construction/reset tests.
