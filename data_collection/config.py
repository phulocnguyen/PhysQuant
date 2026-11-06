# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict YAML run configuration, with explicitly supplied CLI overrides."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CollectionConfig:
    task: str = ""
    task_prompt: str | None = None
    num_episodes: int = 10
    success_target: int | None = None
    max_attempts: int = 50
    seed: int = 712
    capture_profile: str = "full"
    # Optional lower image capture rate; it must divide the physics rate. None keeps settings.yaml.
    camera_hz: float | None = None
    # Cloned environments stepped together in one simulator; 1 keeps the single-scene loop.
    num_envs: int = 1
    video: bool = False
    output_dir: str = "datasets"
    export_lerobot: bool = False
    lerobot_hz: int = 30
    task_settings: dict = field(default_factory=dict)
    randomization: dict = field(default_factory=dict, init=False)


def load_config(path: Path, overrides: dict, config_type=CollectionConfig) -> CollectionConfig:
    """Load YAML safely and reject misspelled fields before creating output files."""
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("YAML configuration must be a mapping")
    known = {name for name, definition in config_type.__dataclass_fields__.items() if definition.init}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    raw.update(overrides)
    cfg = config_type(**raw)
    if not isinstance(cfg.task, str) or not cfg.task or cfg.capture_profile not in ("full", "training"):
        raise ValueError("Unsupported task or capture_profile")
    for key in ("num_episodes", "max_attempts", "lerobot_hz"):
        value = getattr(cfg, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg.success_target is not None and (
        isinstance(cfg.success_target, bool) or not isinstance(cfg.success_target, int) or cfg.success_target < 1
    ):
        raise ValueError("success_target must be a positive integer or null")
    if cfg.success_target is not None and cfg.success_target > cfg.max_attempts:
        raise ValueError("success_target cannot exceed max_attempts")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(cfg.task_settings, dict):
        raise ValueError("task_settings must be a mapping")
    if not isinstance(cfg.export_lerobot, bool):
        raise ValueError("export_lerobot must be true or false")
    if not isinstance(cfg.video, bool):
        raise ValueError("video must be true or false")
    if cfg.camera_hz is not None and (
        isinstance(cfg.camera_hz, bool) or not isinstance(cfg.camera_hz, (int, float)) or cfg.camera_hz <= 0
    ):
        raise ValueError("camera_hz must be a positive number or null")
    if isinstance(cfg.num_envs, bool) or not isinstance(cfg.num_envs, int) or cfg.num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    from environments.randomization import resolve_randomization

    shared_path = Path(__file__).resolve().parents[1] / "configs" / "domain_randomization.yaml"
    cfg.randomization = resolve_randomization(yaml.safe_load(shared_path.read_text()))
    return cfg


def apply_task_settings(settings: dict, task_module) -> None:
    """Apply named task constants before scene configuration classes are imported."""
    camera_keys = {
        "CAMERAS",
        "IMAGE_WIDTH",
        "IMAGE_HEIGHT",
        "CAMERA_HZ",
        "CAMERA_FOCAL_LENGTH",
        "CAMERA_HORIZONTAL_APERTURE",
        "CAMERA_DATA_TYPES",
        "CAMERA_CLIPPING_RANGE",
        "WRIST_CAMERA_FOCAL_LENGTH",
        "WRIST_CAMERA_HORIZONTAL_APERTURE",
        "WRIST_CAMERA_CLIPPING_RANGE",
        "ROBOT_TCP_OFFSET",
        "HAND_TO_GRASP",
        "IK_METHOD",
        "ROBOT_POSITION",
        "ROBOT_ROTATION",
        "ROBOT_YAW_DEG",
        "OBJECT_FORWARD_OFFSETS",
        "OBJECT_LATERAL_OFFSETS",
    }
    if camera_keys & set(settings):
        raise ValueError("Camera, robot, object layout, TCP, and IK settings belong in configs/settings.yaml")
    known = {name for name in vars(task_module) if name.isupper() and not name.startswith("_")}
    unknown = set(settings) - known
    if unknown:
        raise ValueError(f"Unknown task_settings: {sorted(unknown)}")
    for name, value in settings.items():
        setattr(task_module, name, value)
