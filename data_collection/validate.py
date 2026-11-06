# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate native episode lengths, physics states, and causal RGB-D alignment."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from data_collection.metadata import read_metadata, read_scene


def validate_dataset(path: Path, episode_name=None) -> dict:
    """Raise on malformed data, incomplete episodes, or future camera observations."""
    report = {"episodes": 0, "successes": 0, "transitions": 0}
    with h5py.File(path, "r") as dataset:
        if "data" not in dataset:
            raise ValueError("No episode data")
        names = [episode_name] if episode_name is not None else dataset["data"]
        for name in names:
            episode = dataset["data"][name]
            if not episode.attrs.get("finalized", False):
                raise ValueError(f"{name}: incomplete episode")
            n = int(episode.attrs["num_samples"])
            metadata = read_metadata(episode)
            _validate_metadata(episode, name, metadata)
            if n == 0:
                if episode.attrs["success"]:
                    raise ValueError(f"{name}: empty successful episode")
                report["episodes"] += 1
                continue
            if len(episode["actions"]) != n or len(episode["timestamps"]) != n:
                raise ValueError(f"{name}: inconsistent transition lengths")
            times = episode["timestamps"][:]
            if not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
                raise ValueError(f"{name}: invalid timestamps")
            if not np.allclose(np.diff(times), metadata["control_dt_s"]):
                raise ValueError(f"{name}: unexpected control rate")
            for group in ("obs", "states"):

                def check(key, value):
                    if isinstance(value, h5py.Dataset):
                        if len(value) != n or not np.isfinite(value[:]).all():
                            raise ValueError(f"{name}/{group}/{key}: invalid state data")

                episode[group].visititems(check)
            scene = read_scene(episode).get("scenario") or {}
            for key, state in episode["states"].items():
                if key not in episode["obs"]:
                    raise ValueError(f"{name}/{key}: missing pre-action observation")
                observation = episode[f"obs/{key}"]
                if observation.shape != state.shape:
                    raise ValueError(f"{name}/{key}: state/observation shape mismatch")
                # Labelled reference interventions deliberately change object
                # state between actions. Robot proprioception remains continuous.
                intervention = scene.get("diagnostic_intervention") and key.startswith("object_")
                if not intervention and not np.array_equal(observation[1:], state[:-1]):
                    raise ValueError(f"{name}/{key}: broken transition continuity")
                if "initial_state" in episode and key in episode["initial_state"]:
                    if not np.array_equal(observation[0], episode[f"initial_state/{key}"][()]):
                        raise ValueError(f"{name}/{key}: initial observation differs from reset state")
            if len(episode["success"]) != n or not np.isfinite(episode["actions"][:]).all():
                raise ValueError(f"{name}: invalid actions/success")
            if bool(episode["success"][-1]) != bool(episode.attrs["success"]):
                raise ValueError(f"{name}: inconsistent success label")
            if "control_trace" in episode:
                _validate_trace_alignment(episode, metadata)
            for view in metadata["camera_views"]:
                _validate_camera(episode, view, times, metadata)
            report["episodes"] += 1
            report["successes"] += int(episode.attrs["success"])
            report["transitions"] += n
    return report


def _validate_trace_alignment(episode, metadata):
    """Tie physics-rate success evidence to the transitions actually exported."""
    trace = episode["control_trace"]
    hz = metadata.get("pick_sampling_hz", metadata.get("long_horizon_sampling_hz", metadata.get("control_sampling_hz")))
    end_times = episode["timestamps"][:] + metadata["control_dt_s"]
    indices = np.rint(end_times * hz).astype(np.int64) - 1
    if np.any(indices < 0) or np.any(indices >= len(trace["timestamp_s"])):
        raise ValueError("Physics trace does not cover recorded transitions")
    for key in episode["states"]:
        if key in trace and not np.array_equal(episode[f"states/{key}"][:], trace[key][:][indices]):
            raise ValueError(f"{episode.name}/{key}: transition differs from physics trace")


def _validate_camera(episode, view, times, metadata):
    """Check causal indices, calibrated poses, RGB dimensions and depth masks."""
    name = episode.name
    stream = episode[f"sensors/{view}"]
    clock = stream["timestamps"][:]
    indices = episode[f"obs/camera_index/{view}"][:]
    if not len(clock) or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError(f"{name}/{view}: invalid camera clock")
    expected = np.searchsorted(clock, times, side="right") - 1
    if np.any(expected < 0) or not np.array_equal(indices, expected):
        raise ValueError(f"{name}/{view}: invalid causal frame alignment")
    rgb = stream["rgb"]
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"{name}/{view}: invalid RGB")
    for key, shape in (("intrinsics", (3, 3)), ("position", (3,)), ("quaternion_ros_xyzw", (4,))):
        values = stream[key][:]
        if values.shape != (len(clock), *shape) or not np.isfinite(values).all():
            raise ValueError(f"{name}/{view}/{key}: invalid camera calibration")
    intrinsics = stream["intrinsics"][:]
    if np.any(intrinsics[:, (0, 1), (0, 1)] <= 0) or not np.allclose(intrinsics[:, 2], [0, 0, 1], atol=1e-6):
        raise ValueError(f"{name}/{view}: invalid pinhole calibration")
    if not np.allclose(np.linalg.norm(stream["quaternion_ros_xyzw"][:], axis=1), 1, atol=1e-4):
        raise ValueError(f"{name}/{view}: invalid camera quaternion")
    depth_fields = {"distance_to_image_plane", "depth_valid"}
    if metadata.get("capture_profile") == "full" and not depth_fields.issubset(stream):
        raise ValueError(f"{name}/{view}: full capture requires depth and its validity mask")
    for key in stream:
        if len(stream[key]) != len(clock):
            raise ValueError(f"{name}/{view}/{key}: camera stream lengths differ")
    if "distance_to_image_plane" in stream:
        depth = stream["distance_to_image_plane"]
        if "depth_valid" not in stream or depth.shape != stream["depth_valid"].shape:
            raise ValueError(f"{name}/{view}: depth and mask dimensions differ")
        if depth.shape[1:] not in (rgb.shape[1:3], (*rgb.shape[1:3], 1)):
            raise ValueError(f"{name}/{view}: depth dimensions differ from RGB")
        for i in range(len(clock)):
            valid = np.isfinite(depth[i]) & (depth[i] > 0)
            if not np.array_equal(valid, stream["depth_valid"][i]):
                raise ValueError(f"{name}/{view}: invalid depth mask")


def _validate_metadata(episode, name, metadata):
    """Keep episode labels consistent with immutable task truth and review status."""
    if bool(metadata.get("success")) != bool(episode.attrs["success"]):
        raise ValueError(f"{name}: metadata success disagrees with recorded outcome")
    scenario = read_scene(episode)["scenario"]
    if scenario and scenario.get("benchmark") == "Benchmark":
        from tasks.registry import get_task

        get_task(metadata["task"]).validate_recorded_scenario(scenario)
        from evaluation.offline.review import accepted

        if metadata.get("accepted_demonstration", False) != accepted(metadata):
            raise ValueError(f"{name}: inconsistent acceptance label")
        if (
            "scene_json" not in episode.attrs
            and metadata.get("observability_status") != scenario["observability_status"]
        ):
            raise ValueError(f"{name}: inconsistent visibility labels")
        if "scene_json" in episode.attrs:
            for key in ("scenario_id", "case", "condition", "level", "split", "target_id"):
                if metadata[key] != scenario[key]:
                    raise ValueError(f"{name}: metadata {key} disagrees with scene")
            if metadata["target_index"] != scenario["target_index"] or metadata["prompt"] != scenario["instruction"]:
                raise ValueError(f"{name}: target or instruction disagrees with scene")
        if scenario.get("manipulation_protocol") == "pick_place_v1":
            for key in ("episode_id", "scene_id", "template_id", "template_version", "manipulation_protocol"):
                if metadata.get(key) != scenario[key]:
                    raise ValueError(f"{name}: metadata {key} disagrees with scene")
            if metadata["success"]:
                metrics = metadata.get("placement_metrics", {})
                if not metrics.get("success") or metadata.get("success_protocol") != "pick_place_v1":
                    raise ValueError(f"{name}: successful placement requires matching protocol and metrics")
                if (
                    not all(
                        metrics.get(key)
                        for key in (
                            "correct_selection",
                            "lifted",
                            "released",
                            "contained",
                            "supported",
                            "slow",
                            "distractors_preserved",
                        )
                    )
                    or metrics.get("stable_seconds", 0) < scenario["goal"]["stable_hold_s"]
                ):
                    raise ValueError(f"{name}: placement checks do not establish success")
        if scenario.get("manipulation_protocol") == "spatial_control_v1":
            from tasks.control.validation import validate_recording

            validate_recording(episode, scenario, metadata)
        if scenario.get("manipulation_protocol") == "pick_hold_v1":
            from tasks.common.picking import validate_recording

            validate_recording(episode, scenario, metadata)
        if scenario.get("manipulation_protocol") == "long_horizon_v1":
            from tasks.multi_object.validation import validate_recording

            validate_recording(episode, scenario, metadata)


def main() -> None:
    """Run the dataset tool from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    print(json.dumps(validate_dataset(parser.parse_args().dataset), indent=2))


if __name__ == "__main__":
    main()
