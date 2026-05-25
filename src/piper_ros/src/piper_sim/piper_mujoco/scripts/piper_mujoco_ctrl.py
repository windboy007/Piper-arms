#!/usr/bin/env python3
# coding=utf-8

import sys
print(sys.executable)

import os

import mujoco_py
import rclpy
from ament_index_python.packages import get_package_share_directory
from mujoco_py import MjSim, MjViewer
from mujoco_py.generated import const
from rclpy.node import Node
from sensor_msgs.msg import JointState


class MujocoModel(Node):
    def __init__(self):
        super().__init__("mujoco_joint_controller")
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)

        self.joint_targets = {}

        pkg_share_dir = get_package_share_directory("piper_description")
        model_path = os.path.join(pkg_share_dir, "mujoco_model", "piper_description.xml")
        model_path = os.path.abspath(model_path)

        self.get_logger().info(f"The model path is: {model_path}")

        model = mujoco_py.load_model_from_path(model_path)
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)

        self.front_cam_name = "front_cam"
        self.front_cam_ready = self.has_camera(self.front_cam_name)
        if self.front_cam_ready:
            self.front_cam_id = self.sim.model.camera_name2id(self.front_cam_name)
            self.viewer.cam.type = const.CAMERA_FIXED
            self.viewer.cam.fixedcamid = self.front_cam_id
            self.get_logger().info("front_cam is ready and attached to viewer.")
        else:
            self.front_cam_id = -1
            self.get_logger().warn("front_cam was not found in the MuJoCo model.")

        self.timer = self.create_timer(0.01, self.control_loop)
        self.tolerance = 0.05

    def joint_state_callback(self, msg):
        for i, name in enumerate(msg.name):
            self.joint_targets[name] = msg.position[i]

        if "joint7" in self.joint_targets:
            self.joint_targets["joint8"] = -self.joint_targets["joint7"]

    def pos_ctrl(self, joint_name, target_angle):
        if joint_name not in self.sim.model.joint_names:
            self.get_logger().warn(f"Joint {joint_name} not found in Mujoco model.")
            return

        try:
            actuator_id = self.sim.model.actuator_name2id(joint_name)
            self.sim.data.ctrl[actuator_id] = target_angle
        except Exception as e:
            self.get_logger().error(f"Error controlling joint {joint_name}: {e}")

    def has_camera(self, camera_name):
        try:
            self.sim.model.camera_name2id(camera_name)
            return True
        except ValueError:
            return False

    def control_loop(self):
        for joint, target_angle in self.joint_targets.items():
            if joint in self.sim.model.joint_names:
                self.pos_ctrl(joint, target_angle)

        self.sim.step()
        self.viewer.render()


def main():
    rclpy.init()
    mujoco_node = MujocoModel()
    rclpy.spin(mujoco_node)
    mujoco_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()