from glob import glob

from setuptools import find_packages, setup

package_name = "robot_policy"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Priya Vellanki",
    maintainer_email="priyavellanki216@users.noreply.github.com",
    description="BC policy inference node with validated actions.",
    license="MIT",
    entry_points={"console_scripts": ["policy_node = robot_policy.policy_node:main"]},
)
