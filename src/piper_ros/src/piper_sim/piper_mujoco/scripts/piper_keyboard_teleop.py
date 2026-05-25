#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
import os
import sys
import time
import types
from datetime import datetime

import numpy as np

from piper_dynamic_grasp_env import DynamicGraspConfig, PiperDynamicGraspEnv

import glfw

try:
    from mujoco_py.generated import const
except ImportError:  # pragma: no cover - overlay constants depend on mujoco_py build
    const = None


STATUS_PRINT_INTERVAL_SEC = 0.75
CONTROL_GROUPS = (
    ("Macro", (("I", "Down"), ("K", "Up"), ("J", "Left"), ("L", "Right"))),
    ("Macro", (("U", "Forward"), ("O", "Back"), ("N", "Yaw Left"), ("M", "Yaw Right"))),
    ("Gripper", (("[", "Open"), ("]", "Close"), ("BACK", "Reset"), ("ENTER", "Print State"))),
    ("Joint", (("1/2", "J1 -/+"), ("3/4", "J2 -/+"), ("5/6", "J3 -/+"), ("7/8", "J4 -/+"))),
    ("Joint", (("9/0", "J5 -/+"), ("Z/X", "J6 -/+"), ("ESC", "Quit"))),
)


class TeleopEpisodeRecorder:
    def __init__(self, root_dir: str, enabled: bool):
        self.enabled = enabled
        self.root_dir = os.path.abspath(root_dir)
        self.session_dir = ""
        self.episode_index = 0
        self._episode_start_time = 0.0
        self._episode_seed = 0
        self._frames = []

        if self.enabled:
            session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.root_dir, f"teleop_session_{session_stamp}")
            os.makedirs(self.session_dir, exist_ok=True)

    def start_episode(self, seed: int) -> None:
        if not self.enabled:
            return
        self.episode_index += 1
        self._episode_start_time = time.time()
        self._episode_seed = int(seed)
        self._frames = []

    def append_step(self, action: np.ndarray, obs: dict, reward: float, done: bool, info: dict) -> None:
        if not self.enabled:
            return
        self._frames.append(
            {
                "action": np.asarray(action, dtype=float).copy(),
                "qpos": np.asarray(obs["qpos"], dtype=float).copy(),
                "object_position": np.asarray(obs["object_position"], dtype=float).copy(),
                "grasp_position": np.asarray(obs["grasp_position"], dtype=float).copy(),
                "reward": float(reward),
                "done": bool(done),
                "success": bool(info["success"]),
                "distance_to_object": float(info["distance_to_object"]),
                "object_height": float(info["object_height"]),
                "gripper_opening": float(info["gripper_opening"]),
                "elapsed_env_steps": int(info["elapsed_env_steps"]),
                "sim_time": float(info["sim_time"]),
                "wall_time": time.time(),
            }
        )

    def finalize_episode(self, reason: str) -> None:
        if not self.enabled or not self._frames:
            self._frames = []
            return

        episode_name = f"episode_{self.episode_index:05d}"
        episode_dir = os.path.join(self.session_dir, episode_name)
        os.makedirs(episode_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(episode_dir, "trajectory.npz"),
            action=np.stack([frame["action"] for frame in self._frames], axis=0),
            qpos=np.stack([frame["qpos"] for frame in self._frames], axis=0),
            object_position=np.stack([frame["object_position"] for frame in self._frames], axis=0),
            grasp_position=np.stack([frame["grasp_position"] for frame in self._frames], axis=0),
            reward=np.array([frame["reward"] for frame in self._frames], dtype=float),
            done=np.array([frame["done"] for frame in self._frames], dtype=bool),
            success=np.array([frame["success"] for frame in self._frames], dtype=bool),
            distance_to_object=np.array([frame["distance_to_object"] for frame in self._frames], dtype=float),
            object_height=np.array([frame["object_height"] for frame in self._frames], dtype=float),
            gripper_opening=np.array([frame["gripper_opening"] for frame in self._frames], dtype=float),
            elapsed_env_steps=np.array([frame["elapsed_env_steps"] for frame in self._frames], dtype=int),
            sim_time=np.array([frame["sim_time"] for frame in self._frames], dtype=float),
            wall_time=np.array([frame["wall_time"] for frame in self._frames], dtype=float),
        )

        metadata = {
            "episode_index": self.episode_index,
            "seed": self._episode_seed,
            "reason": reason,
            "num_steps": len(self._frames),
            "started_at_unix": self._episode_start_time,
            "ended_at_unix": time.time(),
        }
        with open(os.path.join(episode_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved episode to {episode_dir}")
        self._frames = []


def ensure_ipdb_stub() -> None:
    """Prevent mujoco_py's built-in I-key debug hook from crashing when ipdb is absent."""
    if "ipdb" in sys.modules:
        return

    try:
        import ipdb  # type: ignore  # pragma: no cover - only used when available
    except ImportError:
        stub_module = types.ModuleType("ipdb")

        def _set_trace(*_args, **_kwargs):
            return None

        stub_module.set_trace = _set_trace
        sys.modules["ipdb"] = stub_module


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard teleop for Piper MuJoCo grasp debugging.")
    parser.add_argument("--episode-steps", type=int, default=200000, help="Maximum number of environment steps to run.")
    parser.add_argument("--sim-steps-per-action", type=int, default=1, help="MuJoCo steps executed for each action.")
    parser.add_argument("--camera-name", type=str, default="top_cam", help="Camera used by the viewer.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for object phase.")
    parser.add_argument(
        "--record-dir",
        type=str,
        default="",
        help="Optional directory for saving teleop trajectory episodes as .npz logs.",
    )
    parser.add_argument(
        "--dynamic-object",
        action="store_true",
        help="Enable the moving object trajectory instead of the default static object.",
    )
    parser.add_argument(
        "--joint-step",
        type=float,
        default=0.002,
        help="Per-step increment for direct joint control keys.",
    )
    parser.add_argument(
        "--macro-step",
        type=float,
        default=0.004,
        help="Per-step increment for coarse motion macro keys.",
    )
    return parser.parse_args()


def build_config(args) -> DynamicGraspConfig:
    object_amplitude = np.array([0.10, 0.06], dtype=float) if args.dynamic_object else np.array([0.0, 0.0], dtype=float)
    object_frequency = 0.20 if args.dynamic_object else 0.0
    return DynamicGraspConfig(
        camera_name=args.camera_name,
        sim_steps_per_action=args.sim_steps_per_action,
        episode_steps=args.episode_steps,
        seed=args.seed,
        enable_viewer=True,
        include_camera_obs=False,
        object_amplitude=object_amplitude,
        object_frequency=object_frequency,
    )


def is_pressed(window, key) -> bool:
    return glfw.get_key(window, key) == glfw.PRESS


def any_pressed(window, keys) -> bool:
    return any(is_pressed(window, key) for key in keys)


def apply_macro_controls(window, action: np.ndarray, macro_step: float) -> None:
    # Coarse Cartesian-like macros for quick debugging. If a direction feels reversed,
    # use the opposite key and we can later map the correct joint signs into the policy.
    if is_pressed(window, glfw.KEY_I):  # down
        action[1] += 1.00 * macro_step
        action[2] -= 1.45 * macro_step
        action[4] -= 0.85 * macro_step
    if is_pressed(window, glfw.KEY_K):  # up
        action[1] -= 1.00 * macro_step
        action[2] += 1.45 * macro_step
        action[4] += 0.85 * macro_step
    if is_pressed(window, glfw.KEY_J):  # left
        action[0] += 0.85 * macro_step
    if is_pressed(window, glfw.KEY_L):  # right
        action[0] -= 0.85 * macro_step
    if is_pressed(window, glfw.KEY_U):  # forward
        action[1] += 0.80 * macro_step
        action[2] -= 1.10 * macro_step
        action[4] += 0.45 * macro_step
    if is_pressed(window, glfw.KEY_O):  # backward
        action[1] -= 0.80 * macro_step
        action[2] += 1.10 * macro_step
        action[4] -= 0.45 * macro_step
    if is_pressed(window, glfw.KEY_N):  # yaw left
        action[3] += 0.90 * macro_step
    if is_pressed(window, glfw.KEY_M):  # yaw right
        action[3] -= 0.90 * macro_step


def apply_joint_controls(window, action: np.ndarray, joint_step: float) -> None:
    # Direct joint nudges help determine which single joint drives the unexpected motion.
    joint_key_pairs = (
        (glfw.KEY_1, glfw.KEY_2, 0),
        (glfw.KEY_3, glfw.KEY_4, 1),
        (glfw.KEY_5, glfw.KEY_6, 2),
        (glfw.KEY_7, glfw.KEY_8, 3),
        (glfw.KEY_9, glfw.KEY_0, 4),
        (glfw.KEY_Z, glfw.KEY_X, 5),
    )
    for minus_key, plus_key, joint_idx in joint_key_pairs:
        if is_pressed(window, minus_key):
            action[joint_idx] -= joint_step
        if is_pressed(window, plus_key):
            action[joint_idx] += joint_step

    if is_pressed(window, glfw.KEY_LEFT_BRACKET):  # open gripper
        action[6] += joint_step
    if is_pressed(window, glfw.KEY_RIGHT_BRACKET):  # close gripper
        action[6] -= joint_step


def print_help() -> None:
    print("Keyboard teleop started.")
    print("Macro controls:")
    print("  I: down")
    print("  K: up")
    print("  J: left")
    print("  L: right")
    print("  U: forward")
    print("  O: back")
    print("  N: wrist yaw left")
    print("  M: wrist yaw right")
    print("  [: gripper open")
    print("  ]: gripper close")
    print("Direct joint nudges:")
    print("  1/2 joint1 -, +")
    print("  3/4 joint2 -, +")
    print("  5/6 joint3 -, +")
    print("  7/8 joint4 -, +")
    print("  9/0 joint5 -, +")
    print("  Z/X joint6 -, +")
    print("Other:")
    print("  Backspace: reset to home pose")
    print("  Enter: print current state once")
    print("  ESC: quit")


def add_overlay_line(viewer, grid, label: str, value: str) -> None:
    if const is None or not hasattr(viewer, "add_overlay"):
        return
    viewer.add_overlay(grid, label, value)


def draw_control_overlay(env) -> None:
    viewer = env.viewer
    if viewer is None or const is None or not hasattr(viewer, "add_overlay"):
        return

    if hasattr(viewer, "_overlay"):
        viewer._overlay.clear()

    window = viewer.window
    pressed_states = {
        "I": is_pressed(window, glfw.KEY_I),
        "K": is_pressed(window, glfw.KEY_K),
        "J": is_pressed(window, glfw.KEY_J),
        "L": is_pressed(window, glfw.KEY_L),
        "U": is_pressed(window, glfw.KEY_U),
        "O": is_pressed(window, glfw.KEY_O),
        "N": is_pressed(window, glfw.KEY_N),
        "M": is_pressed(window, glfw.KEY_M),
        "[": is_pressed(window, glfw.KEY_LEFT_BRACKET),
        "]": is_pressed(window, glfw.KEY_RIGHT_BRACKET),
        "BACK": is_pressed(window, glfw.KEY_BACKSPACE),
        "ENTER": is_pressed(window, glfw.KEY_ENTER),
        "ESC": is_pressed(window, glfw.KEY_ESCAPE),
        "1/2": any_pressed(window, (glfw.KEY_1, glfw.KEY_2)),
        "3/4": any_pressed(window, (glfw.KEY_3, glfw.KEY_4)),
        "5/6": any_pressed(window, (glfw.KEY_5, glfw.KEY_6)),
        "7/8": any_pressed(window, (glfw.KEY_7, glfw.KEY_8)),
        "9/0": any_pressed(window, (glfw.KEY_9, glfw.KEY_0)),
        "Z/X": any_pressed(window, (glfw.KEY_Z, glfw.KEY_X)),
    }

    overlay_grids = (
        const.GRID_TOPLEFT,
        const.GRID_TOPRIGHT,
        const.GRID_BOTTOMLEFT,
        const.GRID_BOTTOMRIGHT,
    )

    for grid, (title, items) in zip(overlay_grids, CONTROL_GROUPS):
        add_overlay_line(viewer, grid, title, "Action")
        for keys, meaning in items:
            prefix = "[*]" if pressed_states.get(keys, False) else "[ ]"
            add_overlay_line(viewer, grid, f"{prefix} {keys}", meaning)


def format_array(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values, dtype=float), precision=3, suppress_small=True)


def main():
    ensure_ipdb_stub()
    args = parse_args()
    config = build_config(args)
    env = PiperDynamicGraspEnv(config)
    recorder = TeleopEpisodeRecorder(args.record_dir, enabled=bool(args.record_dir))
    obs = env.reset(seed=args.seed)
    recorder.start_episode(seed=args.seed)
    action = env.current_joint_target.copy()
    viewer_window = env.viewer.window
    last_print_ts = 0.0
    previous_edge_state = {"BACK": False, "ENTER": False}
    episode_finished = False

    print_help()
    print(f"Initial object position: {format_array(obs['object_position'])}")
    if args.record_dir:
        print(f"Recording teleop episodes to: {recorder.session_dir}")
    try:
        while not glfw.window_should_close(viewer_window):
            next_action = action.copy()
            if not episode_finished:
                apply_macro_controls(viewer_window, next_action, args.macro_step)
                apply_joint_controls(viewer_window, next_action, args.joint_step)
                next_action = np.clip(next_action, env.robot_ctrl_low, env.robot_ctrl_high)

            reset_pressed = is_pressed(viewer_window, glfw.KEY_BACKSPACE)
            if reset_pressed and not previous_edge_state["BACK"]:
                recorder.finalize_episode(reason="reset")
                obs = env.reset(seed=args.seed)
                recorder.start_episode(seed=args.seed)
                next_action = env.current_joint_target.copy()
                episode_finished = False
                print(f"Reset environment. object={format_array(obs['object_position'])}")
            previous_edge_state["BACK"] = reset_pressed

            draw_control_overlay(env)
            if not episode_finished:
                obs, reward, done, info = env.step(next_action)
                recorder.append_step(next_action, obs, reward, done, info)
                action = env.current_joint_target.copy()
            else:
                reward = 0.0
                done = True
                info = {
                    "distance_to_object": 0.0,
                    "object_height": float(obs["object_position"][2]),
                    "gripper_opening": float(obs["qpos"][-1]),
                    "success": False,
                    "elapsed_env_steps": 0,
                    "sim_time": 0.0,
                }

            help_pressed = is_pressed(viewer_window, glfw.KEY_ENTER)
            should_print_status = (time.time() - last_print_ts) >= STATUS_PRINT_INTERVAL_SEC
            if (help_pressed and not previous_edge_state["ENTER"]) or should_print_status:
                delta = obs["object_position"] - obs["grasp_position"]
                print(
                    f"reward={reward:.4f} distance={info['distance_to_object']:.4f} "
                    f"height={info['object_height']:.4f} gripper={info['gripper_opening']:.4f} "
                    f"delta={format_array(delta)} qpos={format_array(obs['qpos'])}"
                )
                last_print_ts = time.time()
            previous_edge_state["ENTER"] = help_pressed

            if done:
                if not episode_finished:
                    recorder.finalize_episode(reason="done")
                    print(f"Episode finished with success={info['success']}. Press Backspace to reset or ESC to quit.")
                episode_finished = True
    except KeyboardInterrupt:
        recorder.finalize_episode(reason="keyboard_interrupt")
        print("Stopped by user.")
    finally:
        recorder.finalize_episode(reason="exit")
        env.close()


if __name__ == "__main__":
    main()
