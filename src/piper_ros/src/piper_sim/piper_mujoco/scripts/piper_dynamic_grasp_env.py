#!/usr/bin/env python3
# coding=utf-8

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Optional

import mujoco_py
import numpy as np
from ament_index_python.packages import get_package_share_directory
from mujoco_py import MjSim

try:
    from mujoco_py import MjViewer
except ImportError:  # pragma: no cover - viewer is optional in headless mode
    MjViewer = None


@dataclass
class DynamicGraspConfig:
    camera_name: str = "top_cam"
    image_width: int = 640
    image_height: int = 480
    sim_steps_per_action: int = 5
    episode_steps: int = 480
    seed: int = 0
    enable_viewer: bool = False
    include_camera_obs: bool = True
    object_center: np.ndarray = field(default_factory=lambda: np.array([0.40, 0.0], dtype=float))
    object_amplitude: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))
    object_frequency: float = 0.0
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81], dtype=float))
    floor_z: float = -0.12
    table_pos: np.ndarray = field(default_factory=lambda: np.array([0.22, 0.0, -0.02], dtype=float))
    table_size: np.ndarray = field(default_factory=lambda: np.array([0.48, 0.45, 0.02], dtype=float))
    floor_color: np.ndarray = field(default_factory=lambda: np.array([0.72, 0.72, 0.74, 1.00], dtype=float))
    table_color: np.ndarray = field(default_factory=lambda: np.array([0.78, 0.64, 0.36, 1.00], dtype=float))
    table_leg_color: np.ndarray = field(default_factory=lambda: np.array([0.34, 0.22, 0.10, 1.00], dtype=float))
    object_size: np.ndarray = field(default_factory=lambda: np.array([0.018, 0.018, 0.035], dtype=float))
    object_color: np.ndarray = field(default_factory=lambda: np.array([0.24, 0.48, 0.92, 1.00], dtype=float))
    object_mass: float = 0.14
    object_friction: np.ndarray = field(default_factory=lambda: np.array([4.50, 0.12, 0.02], dtype=float))
    object_spawn_clearance: float = 0.002
    finger_surface_friction: np.ndarray = field(default_factory=lambda: np.array([6.00, 0.20, 0.03], dtype=float))
    gripper_kp: float = 18000.0
    gripper_force_limit: float = 220.0
    grasp_site_color: np.ndarray = field(default_factory=lambda: np.array([0.18, 0.38, 0.95, 0.70], dtype=float))
    home_joint_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.10, -1.75, 0.0, 0.55, 0.0, 0.035], dtype=float)
    )
    success_height: float = 0.12


class PiperDynamicGraspEnv:
    """Minimal MuJoCo environment for dynamic grasp experiments."""

    ROBOT_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
    MIRRORED_GRIPPER_JOINT = "joint8"
    OBJECT_SLIDE_JOINT_NAMES = ("object_slide_x", "object_slide_y")
    OBJECT_SLIDE_ACTUATOR_NAMES = ("object_slide_x_ctrl", "object_slide_y_ctrl")
    OBJECT_FREE_JOINT_NAME = "target_object_freejoint"
    GENERATED_SCENE_NAME = "piper_dynamic_grasp_scene.generated.xml"

    def __init__(self, config: Optional[DynamicGraspConfig] = None):
        self.config = config or DynamicGraspConfig()
        if self.config.enable_viewer and self.config.include_camera_obs:
            raise ValueError("Do not enable viewer and offscreen camera rendering in the same process.")

        self.object_is_kinematic = self._object_is_kinematic()
        self.rng = np.random.default_rng(self.config.seed)
        self._generated_scene_path = self._build_scene_file()
        self.model = mujoco_py.load_model_from_path(self._generated_scene_path)
        self.sim = MjSim(self.model)
        self.viewer = None
        if self.config.enable_viewer:
            if MjViewer is None:
                raise RuntimeError("MjViewer is unavailable in the current mujoco_py installation.")
            self.viewer = MjViewer(self.sim)
            self._configure_viewer_camera()

        self.robot_ctrl_low, self.robot_ctrl_high = self._read_ctrl_limits(self.ROBOT_JOINT_NAMES)
        self.current_joint_target = np.array(self.config.home_joint_pos, dtype=float).copy()
        self.object_phase = 0.0
        self.elapsed_sim_steps = 0
        self.elapsed_env_steps = 0
        self.object_body_id = self.sim.model.body_name2id("target_object")
        self.left_grasp_site_id = self.sim.model.site_name2id("left_finger_grasp_site")
        self.right_grasp_site_id = self.sim.model.site_name2id("right_finger_grasp_site")
        self.grasp_marker_mocap_id = self._get_mocap_id("grasp_center_marker")

        self.reset()

    def _build_scene_file(self) -> str:
        pkg_share_dir = get_package_share_directory("piper_description")
        base_model_path = os.path.join(pkg_share_dir, "mujoco_model", "piper_description.xml")
        base_model_path = os.path.abspath(base_model_path)
        tree = ET.parse(base_model_path)
        root = tree.getroot()

        self._ensure_option(root)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("MuJoCo model does not contain a worldbody node.")

        self._ensure_camera(worldbody)
        self._ensure_light(worldbody)
        self._ensure_floor(worldbody)
        self._ensure_table(worldbody)
        self._ensure_target_body(worldbody)
        self._tune_gripper_contact_surfaces(worldbody)
        self._ensure_grasp_sites(worldbody)
        self._ensure_grasp_center_marker(worldbody)
        self._ensure_object_actuators(root)
        self._tune_gripper_actuators(root)

        generated_path = os.path.join(os.path.dirname(base_model_path), self.GENERATED_SCENE_NAME)
        ET.indent(tree, space="    ")
        tree.write(generated_path, encoding="utf-8", xml_declaration=False)
        return generated_path

    def _ensure_option(self, root: ET.Element) -> None:
        option = root.find("option")
        if option is None:
            option = ET.Element("option")
            root.insert(1, option)
        option.set("gravity", self._vec3(self.config.gravity))
        option.set("timestep", "0.002")

    def _ensure_light(self, worldbody: ET.Element) -> None:
        if any(light.get("name") == "dynamic_scene_light" for light in worldbody.findall("light")):
            return
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "dynamic_scene_light",
                "pos": "0.45 0.0 1.20",
                "dir": "0 0 -1",
                "diffuse": "1.0 1.0 1.0",
                "specular": "0.25 0.25 0.25",
                "ambient": "0.35 0.35 0.35",
                "directional": "true",
                "castshadow": "false",
            },
        )

    def _ensure_camera(self, worldbody: ET.Element) -> None:
        if any(cam.get("name") == self.config.camera_name for cam in worldbody.findall("camera")):
            return
        if self.config.camera_name == "top_cam":
            ET.SubElement(
                worldbody,
                "camera",
                {
                    "name": "top_cam",
                    "pos": "0.42 0.00 0.95",
                    "xyaxes": "1 0 0 0 1 0",
                    "fovy": "55",
                },
            )
            return
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": self.config.camera_name,
                "mode": "targetbody",
                "target": "link4",
                "pos": "1.00 0.00 0.25",
                "fovy": "45",
            },
        )

    def _ensure_floor(self, worldbody: ET.Element) -> None:
        if any(geom.get("name") == "dynamic_floor" for geom in worldbody.findall("geom")):
            return
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "dynamic_floor",
                "type": "plane",
                "pos": f"0 0 {self.config.floor_z:.6f}",
                "size": "1.5 1.5 0.1",
                "rgba": self._vec4(self.config.floor_color),
                "friction": "1.0 0.005 0.0001",
            },
        )

    def _ensure_table(self, worldbody: ET.Element) -> None:
        if any(body.get("name") == "dynamic_grasp_table" for body in worldbody.findall("body")):
            return
        table_body = ET.SubElement(
            worldbody,
            "body",
            {"name": "dynamic_grasp_table", "pos": self._vec3(self.config.table_pos)},
        )
        ET.SubElement(
            table_body,
            "geom",
            {
                "name": "dynamic_grasp_table_top",
                "type": "box",
                "size": self._vec3(self.config.table_size),
                "rgba": self._vec4(self.config.table_color),
                "friction": "1.2 0.01 0.0002",
            },
        )

        table_bottom_z = float(self.config.table_pos[2] - self.config.table_size[2])
        leg_half_height = max(0.02, (table_bottom_z - self.config.floor_z) * 0.5)
        leg_center_z = self.config.floor_z + leg_half_height
        leg_x_offset = max(0.12, float(self.config.table_size[0] - 0.05))
        leg_y_offset = max(0.10, float(self.config.table_size[1] - 0.05))
        leg_half_size = np.array([0.03, 0.03, leg_half_height], dtype=float)

        # Add four legs so the table reads visually as furniture instead of a floating solid block.
        for leg_index, (x_sign, y_sign) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)), start=1):
            leg_pos = np.array(
                [
                    x_sign * leg_x_offset,
                    y_sign * leg_y_offset,
                    leg_center_z - float(self.config.table_pos[2]),
                ],
                dtype=float,
            )
            ET.SubElement(
                table_body,
                "geom",
                {
                    "name": f"dynamic_grasp_table_leg_{leg_index}",
                    "type": "box",
                    "pos": self._vec3(leg_pos),
                    "size": self._vec3(leg_half_size),
                    "rgba": self._vec4(self.config.table_leg_color),
                    "friction": "1.0 0.01 0.0002",
                },
            )

    def _ensure_target_body(self, worldbody: ET.Element) -> None:
        if any(body.get("name") == "target_object" for body in worldbody.findall("body")):
            return
        object_height = (
            self.config.table_pos[2]
            + self.config.table_size[2]
            + self.config.object_size[2]
            + self.config.object_spawn_clearance
        )
        object_xy = np.array([self.config.object_center[0], self.config.object_center[1]], dtype=float)
        body_pos = (
            np.array([0.0, 0.0, object_height], dtype=float)
            if self.object_is_kinematic
            else np.array([object_xy[0], object_xy[1], object_height], dtype=float)
        )
        object_body = ET.SubElement(
            worldbody,
            "body",
            {"name": "target_object", "pos": self._vec3(body_pos)},
        )
        if self.object_is_kinematic:
            ET.SubElement(
                object_body,
                "joint",
                {
                    "name": "object_slide_x",
                    "type": "slide",
                    "axis": "1 0 0",
                    "limited": "true",
                    "range": "0.24 0.58",
                    "damping": "3",
                },
            )
            ET.SubElement(
                object_body,
                "joint",
                {
                    "name": "object_slide_y",
                    "type": "slide",
                    "axis": "0 1 0",
                    "limited": "true",
                    "range": "-0.18 0.18",
                    "damping": "3",
                },
            )
        else:
            ET.SubElement(object_body, "freejoint", {"name": self.OBJECT_FREE_JOINT_NAME})
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": "target_object_geom",
                "type": "box",
                "size": self._vec3(self.config.object_size),
                "mass": f"{float(self.config.object_mass):.6f}",
                "rgba": self._vec4(self.config.object_color),
                "friction": self._vec3(self.config.object_friction),
                "condim": "6",
                "solref": "0.005 1",
                "solimp": "0.99 0.999 0.001",
            },
        )
        ET.SubElement(
            object_body,
            "site",
            {
                "name": "target_object_site",
                "pos": "0 0 0",
                "size": "0.006",
                "rgba": "1 0 0 1",
            },
        )

    def _tune_gripper_contact_surfaces(self, worldbody: ET.Element) -> None:
        left_body = None
        right_body = None
        for body in worldbody.iter("body"):
            if body.get("name") == "link7":
                left_body = body
            elif body.get("name") == "link8":
                right_body = body

        if left_body is None or right_body is None:
            raise RuntimeError("Gripper finger bodies link7/link8 were not found in the MuJoCo model.")

        for body in (left_body, right_body):
            for geom in body.findall("geom"):
                geom.set("friction", self._vec3(self.config.finger_surface_friction))
                geom.set("condim", "6")
                geom.set("solref", "0.003 1")
                geom.set("solimp", "0.995 0.999 0.0005")

    def _ensure_grasp_sites(self, worldbody: ET.Element) -> None:
        left_body = None
        right_body = None
        for body in worldbody.iter("body"):
            if body.get("name") == "link7":
                left_body = body
            elif body.get("name") == "link8":
                right_body = body

        if left_body is None or right_body is None:
            raise RuntimeError("Gripper finger bodies link7/link8 were not found in the MuJoCo model.")

        if not any(site.get("name") == "left_finger_grasp_site" for site in left_body.findall("site")):
            ET.SubElement(
                left_body,
                "site",
                {
                    "name": "left_finger_grasp_site",
                    "pos": "0 -0.045 0.000",
                    "size": "0.004",
                    "rgba": "0.15 0.90 0.40 0.90",
                },
            )

        if not any(site.get("name") == "right_finger_grasp_site" for site in right_body.findall("site")):
            ET.SubElement(
                right_body,
                "site",
                {
                    "name": "right_finger_grasp_site",
                    "pos": "0 -0.045 0.000",
                    "size": "0.004",
                    "rgba": "0.15 0.90 0.40 0.90",
                },
            )

    def _ensure_grasp_center_marker(self, worldbody: ET.Element) -> None:
        if any(body.get("name") == "grasp_center_marker" for body in worldbody.findall("body")):
            return
        marker_body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "grasp_center_marker",
                "mocap": "true",
                "pos": "0 0 0",
            },
        )
        ET.SubElement(
            marker_body,
            "site",
            {
                "name": "grasp_center",
                "pos": "0 0 0",
                "size": "0.006",
                "rgba": self._vec4(self.config.grasp_site_color),
            },
        )

    def _ensure_object_actuators(self, root: ET.Element) -> None:
        if not self.object_is_kinematic:
            return
        actuator = root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")

        existing_names = {node.get("name") for node in actuator.findall("position")}
        if "object_slide_x_ctrl" not in existing_names:
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": "object_slide_x_ctrl",
                    "joint": "object_slide_x",
                    "kp": "200",
                    "ctrllimited": "true",
                    "ctrlrange": "0.24 0.58",
                    "forcelimited": "true",
                    "forcerange": "-30 30",
                },
            )
        if "object_slide_y_ctrl" not in existing_names:
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": "object_slide_y_ctrl",
                    "joint": "object_slide_y",
                    "kp": "200",
                    "ctrllimited": "true",
                    "ctrlrange": "-0.18 0.18",
                    "forcelimited": "true",
                    "forcerange": "-30 30",
                },
            )

    def _tune_gripper_actuators(self, root: ET.Element) -> None:
        actuator = root.find("actuator")
        if actuator is None:
            return

        for node in actuator.findall("position"):
            joint_name = node.get("joint")
            if joint_name not in (self.MIRRORED_GRIPPER_JOINT, "joint7"):
                continue
            node.set("kp", f"{float(self.config.gripper_kp):.1f}")
            force_limit = float(self.config.gripper_force_limit)
            node.set("forcerange", f"{-force_limit:.1f} {force_limit:.1f}")

    def _configure_viewer_camera(self) -> None:
        if self.viewer is None:
            return
        try:
            cam_id = self.sim.model.camera_name2id(self.config.camera_name)
        except ValueError:
            return
        self.viewer.cam.fixedcamid = cam_id
        self.viewer.cam.type = 2

    def _read_ctrl_limits(self, actuator_names) -> tuple[np.ndarray, np.ndarray]:
        lows = []
        highs = []
        for name in actuator_names:
            actuator_id = self.sim.model.actuator_name2id(name)
            ctrl_low, ctrl_high = self.sim.model.actuator_ctrlrange[actuator_id]
            lows.append(ctrl_low)
            highs.append(ctrl_high)
        return np.array(lows, dtype=float), np.array(highs, dtype=float)

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.sim.reset()
        self.elapsed_sim_steps = 0
        self.elapsed_env_steps = 0
        self.current_joint_target = np.clip(
            np.array(self.config.home_joint_pos, dtype=float),
            self.robot_ctrl_low,
            self.robot_ctrl_high,
        )

        for joint_name, target in zip(self.ROBOT_JOINT_NAMES, self.current_joint_target):
            self._set_joint_qpos(joint_name, target)
            self._set_actuator_ctrl(joint_name, target)

        mirrored_target = -float(self.current_joint_target[-1])
        self._set_joint_qpos(self.MIRRORED_GRIPPER_JOINT, mirrored_target)
        self._set_actuator_ctrl(self.MIRRORED_GRIPPER_JOINT, mirrored_target)

        if self.object_is_kinematic:
            self.object_phase = self.rng.uniform(0.0, 2.0 * math.pi)
            x0, y0 = self._object_target_xy(0.0)
            self._set_joint_qpos("object_slide_x", x0)
            self._set_joint_qpos("object_slide_y", y0)
            self._set_actuator_ctrl("object_slide_x_ctrl", x0)
            self._set_actuator_ctrl("object_slide_y_ctrl", y0)
        else:
            self.object_phase = 0.0
            object_height = (
                self.config.table_pos[2]
                + self.config.table_size[2]
                + self.config.object_size[2]
                + self.config.object_spawn_clearance
            )
            free_object_qpos = np.array(
                [self.config.object_center[0], self.config.object_center[1], object_height, 1.0, 0.0, 0.0, 0.0],
                dtype=float,
            )
            self._set_joint_qpos_values(self.OBJECT_FREE_JOINT_NAME, free_object_qpos)
            self._set_joint_qvel_values(self.OBJECT_FREE_JOINT_NAME, np.zeros(6, dtype=float))

        self.sim.forward()
        self._update_grasp_center_marker()
        if self.viewer is not None:
            self.viewer.render()
        return self.get_obs()

    def set_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (len(self.ROBOT_JOINT_NAMES),):
            raise ValueError(f"Expected action shape {(len(self.ROBOT_JOINT_NAMES),)}, got {action.shape}.")

        clipped_action = np.clip(action, self.robot_ctrl_low, self.robot_ctrl_high)
        self.current_joint_target = clipped_action
        return self.current_joint_target.copy()

    def step(self, action: np.ndarray):
        self.set_action(action)

        for _ in range(self.config.sim_steps_per_action):
            if self.object_is_kinematic:
                x_target, y_target = self._object_target_xy(self.sim_time)
                self._set_actuator_ctrl("object_slide_x_ctrl", x_target)
                self._set_actuator_ctrl("object_slide_y_ctrl", y_target)

            for joint_name, joint_target in zip(self.ROBOT_JOINT_NAMES, self.current_joint_target):
                self._set_actuator_ctrl(joint_name, joint_target)

            mirrored_target = -float(self.current_joint_target[-1])
            self._set_actuator_ctrl(self.MIRRORED_GRIPPER_JOINT, mirrored_target)
            self.sim.step()
            self.elapsed_sim_steps += 1

        self._update_grasp_center_marker()
        if self.viewer is not None:
            self.viewer.render()
        self.elapsed_env_steps += 1
        obs = self.get_obs()
        reward, info = self._compute_reward(obs)
        done = bool(info["success"] or self.elapsed_env_steps >= self.config.episode_steps)
        info["elapsed_env_steps"] = self.elapsed_env_steps
        info["sim_time"] = self.sim_time
        return obs, reward, done, info

    def get_obs(self) -> Dict[str, np.ndarray]:
        robot_qpos = np.array([self._get_joint_qpos(name) for name in self.ROBOT_JOINT_NAMES], dtype=float)
        object_position = self.sim.data.body_xpos[self.object_body_id].copy()
        if self.object_is_kinematic:
            object_qpos = np.array([self._get_joint_qpos(name) for name in self.OBJECT_SLIDE_JOINT_NAMES], dtype=float)
            object_qvel = np.array([self._get_joint_qvel(name) for name in self.OBJECT_SLIDE_JOINT_NAMES], dtype=float)
        else:
            object_qpos = object_position.copy()
            object_qvel = self.sim.data.body_xvelp[self.object_body_id].copy()
        grasp_position = self._compute_grasp_position()

        obs = {
            "qpos": robot_qpos,
            "gripper_opening": np.array([robot_qpos[-1]], dtype=float),
            "object_joint_pos": object_qpos,
            "object_joint_vel": object_qvel,
            "object_position": object_position,
            "grasp_position": grasp_position,
        }

        if self.config.include_camera_obs:
            obs["image"] = self.render_camera()

        return obs

    def render_camera(self, camera_name: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None):
        if self.viewer is not None:
            raise RuntimeError("Offscreen rendering is disabled while viewer mode is active.")
        image = self.sim.render(
            width=width or self.config.image_width,
            height=height or self.config.image_height,
            camera_name=camera_name or self.config.camera_name,
            depth=False,
        )
        return np.flipud(image).copy()

    def sample_random_action(self) -> np.ndarray:
        return self.rng.uniform(self.robot_ctrl_low, self.robot_ctrl_high)

    def close(self) -> None:
        self.viewer = None

    @property
    def sim_time(self) -> float:
        return self.elapsed_sim_steps * float(self.model.opt.timestep)

    def _object_target_xy(self, sim_time: float) -> tuple[float, float]:
        if self.config.object_frequency <= 0.0 or not np.any(np.abs(self.config.object_amplitude) > 1e-9):
            return float(self.config.object_center[0]), float(self.config.object_center[1])
        omega = 2.0 * math.pi * self.config.object_frequency
        x = self.config.object_center[0] + self.config.object_amplitude[0] * math.sin(omega * sim_time + self.object_phase)
        y = self.config.object_center[1] + self.config.object_amplitude[1] * math.sin(0.5 * omega * sim_time + 0.5 * self.object_phase)
        return float(x), float(y)

    def _compute_reward(self, obs: Dict[str, np.ndarray]) -> tuple[float, Dict[str, float]]:
        distance = float(np.linalg.norm(obs["grasp_position"] - obs["object_position"]))
        object_height = float(obs["object_position"][2])
        gripper_opening = float(obs["gripper_opening"][0])

        approach_reward = -distance
        close_bonus = 0.30 if distance < 0.06 and gripper_opening < 0.02 else 0.0
        lift_bonus = 10.0 if object_height > self.config.success_height else 0.0
        reward = approach_reward + close_bonus + lift_bonus - 0.01

        info = {
            "distance_to_object": distance,
            "object_height": object_height,
            "gripper_opening": gripper_opening,
            "success": object_height > self.config.success_height,
        }
        return reward, info

    def _set_actuator_ctrl(self, actuator_name: str, value: float) -> None:
        actuator_id = self.sim.model.actuator_name2id(actuator_name)
        self.sim.data.ctrl[actuator_id] = value

    def _set_joint_qpos(self, joint_name: str, value: float) -> None:
        qpos_addr = self._scalar_addr(self.sim.model.get_joint_qpos_addr(joint_name))
        self.sim.data.qpos[qpos_addr] = value

    def _set_joint_qpos_values(self, joint_name: str, values: np.ndarray) -> None:
        qpos_addr = self.sim.model.get_joint_qpos_addr(joint_name)
        if not isinstance(qpos_addr, tuple):
            raise ValueError(f"Joint {joint_name} does not expose a vector qpos address.")
        start, end = int(qpos_addr[0]), int(qpos_addr[1])
        self.sim.data.qpos[start:end] = np.asarray(values, dtype=float)

    def _set_joint_qvel_values(self, joint_name: str, values: np.ndarray) -> None:
        qvel_addr = self.sim.model.get_joint_qvel_addr(joint_name)
        if not isinstance(qvel_addr, tuple):
            raise ValueError(f"Joint {joint_name} does not expose a vector qvel address.")
        start, end = int(qvel_addr[0]), int(qvel_addr[1])
        self.sim.data.qvel[start:end] = np.asarray(values, dtype=float)

    def _compute_grasp_position(self) -> np.ndarray:
        left_pos = self.sim.data.site_xpos[self.left_grasp_site_id].copy()
        right_pos = self.sim.data.site_xpos[self.right_grasp_site_id].copy()
        return 0.5 * (left_pos + right_pos)

    def _update_grasp_center_marker(self) -> None:
        if self.grasp_marker_mocap_id is None:
            return
        self.sim.data.mocap_pos[self.grasp_marker_mocap_id] = self._compute_grasp_position()
        self.sim.forward()

    def _get_joint_qpos(self, joint_name: str) -> float:
        qpos_addr = self._scalar_addr(self.sim.model.get_joint_qpos_addr(joint_name))
        return float(self.sim.data.qpos[qpos_addr])

    def _get_joint_qvel(self, joint_name: str) -> float:
        qvel_addr = self._scalar_addr(self.sim.model.get_joint_qvel_addr(joint_name))
        return float(self.sim.data.qvel[qvel_addr])

    @staticmethod
    def _scalar_addr(addr) -> int:
        if isinstance(addr, tuple):
            return int(addr[0])
        return int(addr)

    def _get_mocap_id(self, body_name: str) -> Optional[int]:
        body_id = self.sim.model.body_name2id(body_name)
        mocap_id = int(self.sim.model.body_mocapid[body_id])
        return None if mocap_id < 0 else mocap_id

    def _object_is_kinematic(self) -> bool:
        return self.config.object_frequency > 0.0 and np.any(np.abs(self.config.object_amplitude) > 1e-9)

    @staticmethod
    def _vec3(values: np.ndarray) -> str:
        return " ".join(f"{float(v):.6f}" for v in values[:3])

    @staticmethod
    def _vec4(values: np.ndarray) -> str:
        return " ".join(f"{float(v):.6f}" for v in values[:4])
