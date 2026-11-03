# ROS 2 integration status

The audit found no ROS 2 executable, `rclpy`, or `/opt/ros` installation in the Manus sandbox. Therefore the `robot_policy` ROS 2 package is **source-only and not runtime-verified**. No ROS 2 tests are claimed to pass.

The Python node accepts `std_msgs/msg/Float32MultiArray` on `/robot/observation` with 28 values in the documented feature order and publishes a 4-element normalized action on `/robot/action`. Parameters select the checkpoint and topic names. Shape, finite-value and action-bound checks are applied; inference errors are logged and produce the safe-stop fallback. The safe-stop is a simulation-interface fallback, not a certified physical robot stop.

On ROS 2 Jazzy, create a workspace, install the core project in the same Python environment, install `rclpy`/`std_msgs`, then run `colcon build --symlink-install` from `ros2_ws`. Runtime verification must confirm the Python interpreter and PyTorch ABI are compatible with the ROS installation before executing `ros2 run robot_policy policy_node`.
