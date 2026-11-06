# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared table, camera, grasp, and collection defaults in SI units."""

import math
from pathlib import Path

import yaml

# Shared YAML follows the reference cameras / robot / controller organization.
_SHARED_SETTINGS = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "settings.yaml").read_text())
TABLE_SIZE = tuple(_SHARED_SETTINGS["table"]["size_m"])
TABLE_HEIGHT = _SHARED_SETTINGS["table"]["height_m"]
TABLE_LEG_WIDTH = _SHARED_SETTINGS["table"]["leg_width_m"]
_CAMERA_SETTINGS = _SHARED_SETTINGS["cameras"]
OBJECT_START_ASLEEP = _SHARED_SETTINGS.get("objects", {}).get("start_asleep", True)
CANDIDATE_LIMITS = _SHARED_SETTINGS["objects"]["candidate_limits"]
CANDIDATE_LAYOUT = _SHARED_SETTINGS["objects"]["candidate_layout"]
CANDIDATE_CYLINDER_HEIGHT = _SHARED_SETTINGS["objects"]["cylinder_height_m"]
OBJECT_FORWARD_OFFSETS = tuple(_SHARED_SETTINGS["objects"]["layout"]["forward_m"])
OBJECT_LATERAL_OFFSETS = tuple(_SHARED_SETTINGS["objects"]["layout"]["lateral_m"])
ROBOT_POSITION = tuple(_SHARED_SETTINGS["robot"]["base_position"])
ROBOT_YAW_DEG = float(_SHARED_SETTINGS["robot"]["base_yaw_deg"])
if len(ROBOT_POSITION) != 3 or not all(math.isfinite(v) for v in (*ROBOT_POSITION, ROBOT_YAW_DEG)):
    raise ValueError("Robot base_position must be finite XYZ and base_yaw_deg must be finite")
ROBOT_ROTATION = (0.0, 0.0, math.sin(math.radians(ROBOT_YAW_DEG) / 2), math.cos(math.radians(ROBOT_YAW_DEG) / 2))
if (
    not OBJECT_FORWARD_OFFSETS
    or not OBJECT_LATERAL_OFFSETS
    or not all(math.isfinite(v) for v in (*OBJECT_FORWARD_OFFSETS, *OBJECT_LATERAL_OFFSETS))
):
    raise ValueError("Object layout must contain nonempty finite forward_m and lateral_m offsets")
_APPEARANCE = _SHARED_SETTINGS["appearance"]
CANDIDATE_COLOR = tuple(_APPEARANCE["candidate_color"])
if len(CANDIDATE_COLOR) != 3 or any(
    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
    for v in CANDIDATE_COLOR
):
    raise ValueError("appearance.candidate_color must contain three finite RGB values in [0, 1]")
COLORS = tuple(tuple(color) for color in _APPEARANCE["object_colors"])
TABLE_COLOR = tuple(_APPEARANCE["table_color"])
MATERIAL_ROUGHNESS = _APPEARANCE["roughness"]
DOME_INTENSITY = _SHARED_SETTINGS["lighting"]["dome_intensity"]
KEY_INTENSITY = _SHARED_SETTINGS["lighting"]["key_intensity"]
RENDER_SETTINGS = _SHARED_SETTINGS["rendering"]
if RENDER_SETTINGS["mode"] not in ("realtime", "path_traced"):
    raise ValueError("rendering.mode must be realtime or path_traced")
if RENDER_SETTINGS["path_tracing_samples"] < 1:
    raise ValueError("rendering.path_tracing_samples must be positive")
_SCENE_CAMERA = _CAMERA_SETTINGS["scene"]
_WRIST_CAMERA = _CAMERA_SETTINGS["wrist"]
CAMERAS = dict(_SCENE_CAMERA["views"])
CAMERA_POSITION = tuple(CAMERAS["top"]["position"])
CAMERA_TARGET = tuple(CAMERAS["top"]["target"])
CAMERAS["wrist"] = {
    "body": _WRIST_CAMERA["body"],
    "offset_position": _WRIST_CAMERA["offset_pos"],
    "offset_rotation_xyzw": _WRIST_CAMERA["offset_rot"],
}
if set(CAMERAS) != {"top", "left", "right", "wrist"}:
    raise ValueError("settings.yaml must define top, left, right, and wrist cameras")
if _WRIST_CAMERA["convention"] != "ros":
    raise ValueError("Franka wrist offset_rot must use ROS optical convention and XYZW order")
IMAGE_WIDTH, IMAGE_HEIGHT = _CAMERA_SETTINGS["resolution"]
if _CAMERA_SETTINGS["update_period"] <= 0:
    raise ValueError("Camera update_period must be positive")
CAMERA_HZ = 1 / _CAMERA_SETTINGS["update_period"]
CAMERA_DATA_TYPES = _CAMERA_SETTINGS["data_types"]
if "rgb" not in CAMERA_DATA_TYPES or set(CAMERA_DATA_TYPES) - {"rgb", "distance_to_image_plane"}:
    raise ValueError("Camera data_types supports rgb and distance_to_image_plane")
CAMERA_FOCAL_LENGTH = _SCENE_CAMERA["focal_length"]
CAMERA_HORIZONTAL_APERTURE = _SCENE_CAMERA["horizontal_aperture"]
CAMERA_CLIPPING_RANGE = _SCENE_CAMERA["clipping_range"]
WRIST_CAMERA_FOCAL_LENGTH = _WRIST_CAMERA["focal_length"]
WRIST_CAMERA_HORIZONTAL_APERTURE = _WRIST_CAMERA["horizontal_aperture"]
WRIST_CAMERA_CLIPPING_RANGE = _WRIST_CAMERA["clipping_range"]
ROBOT_TCP_OFFSET = _SHARED_SETTINGS["robot"]["tcp_offset"]
if len(ROBOT_TCP_OFFSET) != 3 or ROBOT_TCP_OFFSET[:2] != [0.0, 0.0]:
    raise ValueError("The v1 overhead grasp controller requires an axial TCP offset [0, 0, z]")
_CONTROLLER = _SHARED_SETTINGS["controller"]
IK_METHOD = _CONTROLLER["ik_method"]
if IK_METHOD not in {"dls", "pinv", "svd", "trans"}:
    raise ValueError("Unsupported controller.ik_method")
_SIMULATION = _SHARED_SETTINGS["simulation"]
PHYSICS_DT = 1 / _SIMULATION["physics_hz"]
CONTROL_STEPS = _SIMULATION["control_steps"]
_SETTLING = _SIMULATION["settling"]
SETTLE_SECONDS = _SETTLING["min_s"]
SETTLE_MAX_SECONDS = _SETTLING["max_s"]
SETTLE_HOLD_SECONDS = _SETTLING["hold_s"]
SETTLE_LINEAR_SPEED = _SETTLING["linear_speed_m_s"]
SETTLE_ANGULAR_SPEED = _SETTLING["angular_speed_rad_s"]
_SUCCESS = _SHARED_SETTINGS["success"]
LIFT_HEIGHT = _SUCCESS["lift_height_m"]
SUCCESS_HOLD_SECONDS = _SUCCESS["hold_s"]
MAX_DISTRACTOR_MOVEMENT = _SUCCESS["max_distractor_movement_m"]
MAX_HAND_DISTANCE = _SUCCESS["max_hand_distance_m"]
TIMEOUT_SECONDS = _SIMULATION["timeout_s"]
HAND_TO_GRASP = ROBOT_TCP_OFFSET[2]  # Panda hand origin to finger grasp centre.
APPROACH_CLEARANCE = _CONTROLLER["approach_clearance_m"]
LIFT_DISTANCE = _CONTROLLER["lift_distance_m"]
GRIPPER_CLOSED_WIDTH = _SUCCESS["gripper_closed_width_m"]
POSITION_TOLERANCE = _CONTROLLER["position_tolerance_m"]
ORIENTATION_TOLERANCE = _CONTROLLER["orientation_tolerance_rad"]
GRIPPER_VELOCITY_TOLERANCE = _CONTROLLER["gripper_velocity_tolerance_m_s"]


def validate_settings(settings):
    """Reject invalid timing or success overrides before starting the simulator."""
    from core.timing import capture_stride

    positive = (
        "PHYSICS_DT",
        "CAMERA_HZ",
        "TIMEOUT_SECONDS",
        "SUCCESS_HOLD_SECONDS",
        "LIFT_HEIGHT",
        "LIFT_DISTANCE",
        "MAX_DISTRACTOR_MOVEMENT",
        "MAX_HAND_DISTANCE",
        "GRIPPER_CLOSED_WIDTH",
        "POSITION_TOLERANCE",
        "ORIENTATION_TOLERANCE",
        "GRIPPER_VELOCITY_TOLERANCE",
        "SETTLE_SECONDS",
        "SETTLE_MAX_SECONDS",
        "SETTLE_HOLD_SECONDS",
        "SETTLE_LINEAR_SPEED",
        "SETTLE_ANGULAR_SPEED",
        "TABLE_HEIGHT",
        "TABLE_LEG_WIDTH",
        "APPROACH_CLEARANCE",
        "HAND_TO_GRASP",
        "CAMERA_FOCAL_LENGTH",
        "WRIST_CAMERA_FOCAL_LENGTH",
        "CAMERA_HORIZONTAL_APERTURE",
        "WRIST_CAMERA_HORIZONTAL_APERTURE",
    )
    for name in positive:
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    for name in ("CONTROL_STEPS", "IMAGE_WIDTH", "IMAGE_HEIGHT"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    capture_stride(settings.PHYSICS_DT, settings.CONTROL_STEPS, settings.CAMERA_HZ)
    if max(settings.SETTLE_SECONDS, settings.SETTLE_HOLD_SECONDS) > settings.SETTLE_MAX_SECONDS:
        raise ValueError("SETTLE_MAX_SECONDS must cover the minimum settle time and stable hold")
    if settings.TIMEOUT_SECONDS <= settings.SUCCESS_HOLD_SECONDS:
        raise ValueError("TIMEOUT_SECONDS must exceed SUCCESS_HOLD_SECONDS")
    if settings.LIFT_DISTANCE <= settings.LIFT_HEIGHT:
        raise ValueError("LIFT_DISTANCE must exceed the required LIFT_HEIGHT")
    if settings.GRIPPER_CLOSED_WIDTH >= 0.08:
        raise ValueError("GRIPPER_CLOSED_WIDTH must be below the Panda's fully open 0.08 m width")
    for name in ("CAMERA_CLIPPING_RANGE", "WRIST_CAMERA_CLIPPING_RANGE"):
        limits = getattr(settings, name)
        if len(limits) != 2 or not all(math.isfinite(v) for v in limits) or not 0 < limits[0] < limits[1]:
            raise ValueError(f"{name} must be ordered positive near/far distances")
    placement = getattr(settings, "PLACEMENT_CONTROL", {})
    if not isinstance(placement, dict):
        raise ValueError("PLACEMENT_CONTROL must be a mapping")
    for name, value in placement.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"PLACEMENT_CONTROL.{name} must be a positive finite number")
