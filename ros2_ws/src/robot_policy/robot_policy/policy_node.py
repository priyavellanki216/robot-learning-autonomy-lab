"""ROS 2 policy inference node (source-only in the current sandbox)."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from robot_learning.imitation_learning.behavior_cloning import BehaviorCloningPolicy
from robot_learning.safety.action_validator import safe_stop_action, validate_action


class PolicyExecutionNode(Node):
    """Consumes 28D observations and publishes normalized 4D Fetch actions.

    Message contract: data=[25 Fetch observation values, desired_goal XYZ].
    Output: [delta_x, delta_y, delta_z, gripper_open_close], normalized [-1,1].
    This node is not a certified or hardware safety controller.
    """

    def __init__(self) -> None:
        super().__init__("robot_policy")
        self.declare_parameter("checkpoint", "experiments/checkpoints/bc_policy.pt")
        self.declare_parameter("observation_topic", "/robot/observation")
        self.declare_parameter("action_topic", "/robot/action")
        self.policy = BehaviorCloningPolicy(self.get_parameter("checkpoint").value)
        output_topic = self.get_parameter("action_topic").value
        input_topic = self.get_parameter("observation_topic").value
        self.publisher = self.create_publisher(Float32MultiArray, output_topic, 10)
        self.subscription = self.create_subscription(Float32MultiArray, input_topic, self._on_observation, 10)

    def _publish(self, action: np.ndarray) -> None:
        message = Float32MultiArray()
        message.data = np.asarray(action, dtype=np.float32).tolist()
        self.publisher.publish(message)

    def _on_observation(self, message: Float32MultiArray) -> None:
        try:
            observation = np.asarray(message.data, dtype=np.float32)
            if observation.shape != (28,):
                raise ValueError(f"expected 28 values, got {observation.shape}")
            self._publish(validate_action(self.policy.predict(observation)))
        except Exception as exc:  # Keep callback safe; report and command stop fallback.
            self.get_logger().error(f"Policy inference rejected observation: {exc}")
            self._publish(safe_stop_action())


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PolicyExecutionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
