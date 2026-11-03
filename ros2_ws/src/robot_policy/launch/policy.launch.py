"""Launch the policy node with configurable checkpoint and topic names."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("checkpoint", default_value="experiments/checkpoints/bc_policy.pt"),
            DeclareLaunchArgument("observation_topic", default_value="/robot/observation"),
            DeclareLaunchArgument("action_topic", default_value="/robot/action"),
            Node(
                package="robot_policy",
                executable="policy_node",
                name="robot_policy",
                output="screen",
                parameters=[
                    {
                        "checkpoint": LaunchConfiguration("checkpoint"),
                        "observation_topic": LaunchConfiguration("observation_topic"),
                        "action_topic": LaunchConfiguration("action_topic"),
                    }
                ],
            ),
        ]
    )
