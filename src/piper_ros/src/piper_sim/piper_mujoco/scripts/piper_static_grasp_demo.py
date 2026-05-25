#!/usr/bin/env python3
# coding=utf-8

import argparse

import numpy as np

from piper_dynamic_grasp_env import DynamicGraspConfig, PiperDynamicGraspEnv
from scripted_grasp_policy import compute_static_grasp_target


def parse_args():
    parser = argparse.ArgumentParser(description="Static grasp demo runner for Piper MuJoCo.")
    parser.add_argument("--viewer", action="store_true", help="Use the MuJoCo viewer instead of offscreen rendering.")
    parser.add_argument("--episode-steps", type=int, default=760, help="Maximum number of environment steps to run.")
    parser.add_argument("--loop", action="store_true", help="Run continuously until interrupted with Ctrl+C.")
    parser.add_argument("--sim-steps-per-action", type=int, default=5, help="MuJoCo steps executed for each action.")
    parser.add_argument("--camera-name", type=str, default="top_cam", help="Camera used for observations.")
    parser.add_argument("--width", type=int, default=640, help="Rendered image width.")
    parser.add_argument("--height", type=int, default=480, help="Rendered image height.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for object phase.")
    parser.add_argument(
        "--dynamic-object",
        action="store_true",
        help="Enable the moving object trajectory instead of the default static object.",
    )
    parser.add_argument(
        "--no-camera-obs",
        action="store_true",
        help="Disable offscreen image observations. Required when --viewer is active.",
    )
    return parser.parse_args()


def build_config(args) -> DynamicGraspConfig:
    object_amplitude = np.array([0.10, 0.06], dtype=float) if args.dynamic_object else np.array([0.0, 0.0], dtype=float)
    object_frequency = 0.20 if args.dynamic_object else 0.0
    return DynamicGraspConfig(
        camera_name=args.camera_name,
        image_width=args.width,
        image_height=args.height,
        sim_steps_per_action=args.sim_steps_per_action,
        episode_steps=args.episode_steps,
        seed=args.seed,
        enable_viewer=args.viewer,
        include_camera_obs=not args.no_camera_obs,
        object_amplitude=object_amplitude,
        object_frequency=object_frequency,
    )


def main():
    args = parse_args()
    config = build_config(args)

    if config.enable_viewer and config.include_camera_obs:
        raise ValueError("Use --no-camera-obs together with --viewer to avoid viewer/render conflicts.")

    env = PiperDynamicGraspEnv(config)
    obs = env.reset(seed=args.seed)
    print(f"Initial object position: {obs['object_position']}")

    step_idx = 0
    last_phase_name = None
    try:
        while True:
            phase_name, phase_step, phase_duration, action, delta = compute_static_grasp_target(env, step_idx)
            if phase_name != last_phase_name:
                print(f"[phase] {phase_name} ({phase_step + 1}/{phase_duration})")
                last_phase_name = phase_name

            obs, reward, done, info = env.step(action)
            if step_idx % 20 == 0:
                print(
                    f"step={step_idx:03d} phase={phase_name} reward={reward:.4f} "
                    f"distance={info['distance_to_object']:.4f} "
                    f"height={info['object_height']:.4f} "
                    f"gripper={info['gripper_opening']:.4f} "
                    f"delta={np.array2string(delta, precision=3)} "
                    f"object={np.array2string(obs['object_position'], precision=3)} "
                    f"grasp={np.array2string(obs['grasp_position'], precision=3)}"
                )

            if done:
                print(f"Episode finished at step {step_idx} with success={info['success']}.")
                if not args.loop:
                    break
                obs = env.reset()
                print(f"Reset environment, new object position: {obs['object_position']}")
                last_phase_name = None

            step_idx += 1
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
