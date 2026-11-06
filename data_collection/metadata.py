# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact episode outcomes, shared recording information, and private scene truth.

Human-readable metadata describes the attempt. Geometry and sampled physics live
in HDF5 ``scene_json``; shared recording conventions live in ``recording_json``.
Readers also support the earlier, expanded metadata layout.
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml


def find_datasets(root: Path) -> list[Path]:
    """Find native recordings, following archive links once and avoiding cycles."""
    return _find_recording_files(root, ("dataset.hdf5", "episode.hdf5"))


def find_episode_metadata(root: Path) -> list[Path]:
    """Find atomic attempt sidecars, including interrupted archive recordings."""
    return _find_recording_files(root, ("metadata.json",))


def _find_recording_files(root: Path, names: tuple[str, ...]) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    seen, paths = set(), set()
    for directory, children, files in os.walk(root, followlinks=True):
        resolved = Path(directory).resolve()
        if resolved in seen:
            children.clear()
            continue
        seen.add(resolved)
        for name in names:
            if name in files:
                paths.add((resolved / name).resolve())
    return sorted(paths)


def episode_dataset(directory):
    """Locate an isolated recording or the historical shared worker recording."""
    directory = Path(directory)
    isolated = directory / "dataset.hdf5"
    return isolated if isolated.exists() else directory.parent / "dataset.hdf5"


def episode_directory(dataset, name):
    """Locate sidecars for either supported recording layout."""
    parent = Path(dataset).parent
    return parent if parent.name == name else parent / name


def write_json_atomic(path, value):
    """Publish a complete JSON record after flushing its temporary sibling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def save_run_config(directory, config, settings, output_dir):
    """Save one resolved configuration, excluding the separately recorded scenario."""
    resolved = asdict(config)
    resolved.pop("scenario", None)
    resolved.pop("scenarios", None)
    resolved["output_dir"] = str(output_dir)
    resolved["task_settings"] = {
        key: value for key, value in vars(settings).items() if key.isupper() and not key.startswith("_")
    }
    (directory / "run_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    return resolved


def recording_info(settings, joint_names, capture_profile, success_protocol):
    """Describe shared sensor/action contracts once per dataset."""
    return {
        "robot": "franka",
        "joint_names": list(joint_names),
        "control_dt_s": settings.PHYSICS_DT * settings.CONTROL_STEPS,
        "camera_hz": settings.CAMERA_HZ,
        "camera_views": list(settings.CAMERAS),
        "image_size_hw": [settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH],
        "capture_profile": capture_profile,
        "collection_mode": "scripted",
        "identification_source": "oracle_authored_geometry",
        "success_protocol": success_protocol,
        "action_schema": "joint_position_target; joint_names order; radians, finger metres",
        "pose_schema": "world XYZ metres; quaternion XYZW",
        "camera_pose_schema": "world XYZ metres; ROS optical quaternion XYZW",
        "depth_schema": "distance_to_image_plane metres; depth_valid mask",
        "transition_schema": "obs=state_t; actions=command_t; states=state_t+1",
        "image_alignment": "obs/camera_index selects latest camera frame at or before action",
    }


def episode_info(task, scenario, seed, target_index):
    """Keep the episode sidecar limited to identity, instruction, and outcome."""
    return {
        "benchmark": "Benchmark",
        "task": task,
        **{key: scenario[key] for key in ("scenario_id", "case", "condition", "level", "split")},
        "prompt": scenario["instruction"],
        "seed": seed,
        "target_id": scenario["target_id"],
        "target_index": target_index,
        "observability_status": scenario["observability_status"],
        "accepted_demonstration": False,
        **{
            key: scenario[key]
            for key in ("episode_id", "scene_id", "template_id", "template_version", "manipulation_protocol")
            if key in scenario
        },
    }


def read_metadata(episode):
    """Read either layout, adding shared recording information for consumers."""
    shared = json.loads(episode.file.attrs.get("recording_json", "{}"))
    return shared | json.loads(episode.attrs["metadata_json"])


def read_scene(episode):
    """Return exact scene truth without adding it to the public episode summary."""
    if "scene_json" in episode.attrs:
        scene = json.loads(episode.attrs["scene_json"])
        if "shared_scene_id" in scene:
            identity = scene.pop("shared_scene_id")
            shared = episode.file[f"scene_specs/{identity}"]
            objects = json.loads(shared.attrs["objects_json"])
            scene["scenario"]["objects"] = objects
            scene["objects"] = [
                dict(obj, **pose) for obj, pose in zip(objects, scene.pop("initial_object_poses"), strict=True)
            ]
        return scene
    metadata = json.loads(episode.attrs["metadata_json"])
    return {
        "scenario": metadata.get("query", {}).get("scenario"),
        "objects": metadata.get("objects", []),
        "randomization_samples": metadata.get("randomization_samples", {}),
    }


def read_settings(dataset):
    """Use the captured settings; never current defaults."""
    if "run_config_json" in dataset.attrs:
        return json.loads(dataset.attrs["run_config_json"])["task_settings"]
    root = Path(dataset.filename).parent
    return yaml.safe_load((root / "run_config.yaml").read_text())["task_settings"]
