# Manus execution environment audit

Audit performed 2026-11-03 in the project sandbox. This describes that execution environment at audit time, not a promise about other Manus devices or future sessions.

| Capability | Finding | Evidence/status |
|---|---|---|
| OS / architecture | Ubuntu 24.04.5 LTS, x86_64 | Available and tested (`uname`, `/etc/os-release`) |
| Python | 3.12.3 | Available and tested |
| CPU / RAM | 6 visible CPU cores; about 7.9 GiB total memory, about 6.6 GiB available during audit | Available and tested from `/proc` |
| Storage | 49 GiB root mount, about 40 GiB free during audit | Available and tested via `df` |
| GPU / CUDA | No GPU exposed; `nvidia-smi` and `nvcc` absent; `torch.cuda.is_available()` false | CUDA training unavailable; actual experiments ran on CPU |
| PyTorch | 2.15.0 | Available and tested: BC train/checkpoint/inference and PPO rollout/checkpoint/inference |
| Gymnasium | 1.2.3 | Available and tested: reset/step and automated environment tests |
| Gymnasium Robotics | 1.4.2 | Available and tested with the Fetch task after applying the documented NumPy enum compatibility shim |
| MuJoCo | 3.17.0 | Available and tested through the genuine Fetch manipulation environment |
| Stable-Baselines3 | 2.7.0 | Available and tested: PPO training and checkpoint loading |
| OpenCV | Not installed | Not required by this project |
| ROS 2 / `rclpy` | Not installed; `ros2` command and `/opt/ros` absent | Not available; ROS package remains source-only and untested at runtime |
| Gazebo | Not installed | Not available; no Gazebo integration claimed |
| Graphical simulation | `DISPLAY=:0` existed, but an actual simulator renderer was not tested | Available but untested; headless mode is the documented default |
| Background processes | A short background process launched and completed | Available and tested; this is not a persistent daemon guarantee |
| Workspace files | Temporary file write/read succeeded; project outputs were generated and retained through this task | Available and tested within this task; persistence across hibernation/session lifetimes was not independently verified |
| Git / GitHub | Git 2.43.0 and GitHub CLI 2.103.2; authenticated to `priyavellanki216` | Available and tested; repository `main` could be read and the final commit push was confirmed |

## Dependency compatibility

The tested pinned set is listed in `requirements.txt` and `pyproject.toml`. `pip check` reported no broken requirements after refreshing the editable project metadata. The observed combination of Gymnasium Robotics and MuJoCo rejected NumPy joint enum membership for a Fetch slide joint during setup; `src/robot_learning/simulation/mujoco_compat.py` contains the small comparison-only runtime shim, and environment construction/reset/step are covered by tests. The shim does not replace physics or task behavior.

The sandbox does not provide a verified graphical display path, CUDA device, ROS 2, or Gazebo runtime. No installation or integration claims are made for those capabilities.
