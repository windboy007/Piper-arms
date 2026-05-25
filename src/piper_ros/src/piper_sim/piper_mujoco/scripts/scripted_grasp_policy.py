#!/usr/bin/env python3
# coding=utf-8

import numpy as np


PHASE_SEQUENCE = (
    ("home_open", 40),
    ("align_xy", 180),
    ("descend", 170),
    ("close_gripper", 90),
    ("lift", 120),
    ("hold", 80),
)

PHASE_POSES = {
    "home_open": np.array([0.00, 1.10, -1.75, 0.00, 0.55, 0.00, 0.035], dtype=float),
    "align_xy": np.array([0.04, 1.02, -1.70, -0.12, 0.60, 0.03, 0.035], dtype=float),
    "descend": np.array([0.04, 1.34, -2.18, -0.18, 0.42, 0.02, 0.035], dtype=float),
    "close_gripper": np.array([0.04, 1.34, -2.18, -0.18, 0.42, 0.02, 0.003], dtype=float),
    "lift": np.array([0.04, 1.12, -1.86, -0.12, 0.54, 0.00, 0.003], dtype=float),
    "hold": np.array([0.02, 0.98, -1.58, -0.10, 0.48, 0.00, 0.006], dtype=float),
}


def _phase_window(step_idx: int) -> tuple[str, int, int]:
    cycle_step = step_idx % sum(duration for _, duration in PHASE_SEQUENCE)
    for phase_name, duration in PHASE_SEQUENCE:
        if cycle_step < duration:
            return phase_name, cycle_step, duration
        cycle_step -= duration
    phase_name, duration = PHASE_SEQUENCE[-1]
    return phase_name, duration - 1, duration


def _compute_object_delta(env) -> np.ndarray:
    object_position = env.sim.data.body_xpos[env.object_body_id].copy()
    grasp_position = env._compute_grasp_position()
    return object_position - grasp_position


def compute_static_grasp_target(env, step_idx: int) -> tuple[str, int, int, np.ndarray, np.ndarray]:
    phase_name, phase_step, phase_duration = _phase_window(step_idx)
    target = PHASE_POSES[phase_name].copy()
    delta = _compute_object_delta(env)

    # Clamp the feedback term so the arm stays smooth and easy to tune.
    dx = float(np.clip(delta[0], -0.08, 0.08))
    dy = float(np.clip(delta[1], -0.08, 0.08))
    dz = float(np.clip(delta[2], -0.08, 0.08))

    if phase_name == "align_xy":
        # Small yaw and wrist corrections keep the midpoint centered laterally.
        target[0] += 0.55 * dy
        target[3] += 1.10 * dy

        # Shoulder/elbow pair mainly handles forward-back alignment in x.
        target[1] += 1.25 * dx
        target[2] -= 1.75 * dx
        target[4] -= 0.50 * dx

    if phase_name in ("descend", "close_gripper"):
        # Stop chasing left-right aggressively during the final approach.
        target[0] += 0.02 * dy
        target[3] += 0.04 * dy

        # If the object is below the grasp center (dz < 0), push the arm downward.
        z_alignment_error = np.clip(dz, -0.10, 0.10)
        target[1] += 2.60 * z_alignment_error
        target[2] -= 2.05 * z_alignment_error
        target[4] -= 1.25 * z_alignment_error

    if phase_name == "lift":
        # Keep the grasp roughly centered during lift so the object is less likely to slip.
        target[0] += 0.25 * dy
        target[3] += 0.40 * dy
        target[1] += 0.50 * dx
        target[2] -= 0.55 * dx

    return phase_name, phase_step, phase_duration, np.clip(target, env.robot_ctrl_low, env.robot_ctrl_high), delta


def scripted_grasp_phase(step_idx: int) -> tuple[str, int, int, np.ndarray]:
    phase_name, phase_step, phase_duration = _phase_window(step_idx)
    return phase_name, phase_step, phase_duration, PHASE_POSES[phase_name].copy()


def scripted_joint_target(env, step_idx: int) -> np.ndarray:
    _, _, _, target, _ = compute_static_grasp_target(env, step_idx)
    return target
