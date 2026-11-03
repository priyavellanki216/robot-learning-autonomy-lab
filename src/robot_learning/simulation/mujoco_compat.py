"""Small compatibility shim for Gymnasium Robotics with MuJoCo's NumPy enum scalars.

Some MuJoCo releases expose ``model.jnt_type`` as NumPy integer scalars whose
hash differs from the pybind enum hash. Gymnasium Robotics compares these using
tuple membership, which can reject a valid slide joint. This shim preserves the
upstream utility implementation while making those four comparisons integer-safe.
"""

from __future__ import annotations

import inspect


def patch_gymnasium_robotics_joint_helpers() -> bool:
    """Patch joint helpers only when the affected implementation is installed.

    Returns True when a patch was applied, False when no affected source is
    present. Import this before constructing a Gymnasium Robotics environment.
    """
    import gymnasium_robotics.utils.mujoco_utils as utils

    source = inspect.getsource(utils)
    broken = "assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)"
    fixed = "assert int(joint_type) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))"
    if broken not in source:
        return False

    namespace = dict(utils.__dict__)
    exec(compile(source.replace(broken, fixed), inspect.getsourcefile(utils) or "<mujoco_utils>", "exec"), namespace)
    for name in ("set_joint_qpos", "set_joint_qvel", "get_joint_qpos", "get_joint_qvel"):
        setattr(utils, name, namespace[name])
    return True
