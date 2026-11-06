# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reconstruct Multi-object event history and outcomes from physics-rate traces."""

import numpy as np
from assets.geometry import world_aabb

from .evaluator import MultiObjectEvaluator


def validate_recording(episode, scenario, metadata):
    for key in ("episode_id", "scene_id", "template_id", "template_version", "manipulation_protocol"):
        if metadata.get(key) != scenario[key]:
            raise ValueError(f"Multi-object metadata {key} differs from scenario")
    if metadata.get("success_protocol") != "long_horizon_v1" or "control_trace" not in episode:
        raise ValueError("Multi-object recording needs its protocol and physics-rate evidence")
    trace = episode["control_trace"]
    times = trace["timestamp_s"][:]
    hz = metadata["long_horizon_sampling_hz"]
    if hz <= 0 or not len(times) or not np.allclose(np.diff(np.r_[0.0, times]), 1 / hz, atol=1e-8):
        raise ValueError("Incomplete Multi-object trace")
    if metadata["success"] and not np.isclose(times[-1], metadata["episode_duration_s"], atol=1e-8):
        raise ValueError("Successful Multi-object trace must cover the completed transitions")
    substeps = metadata["control_dt_s"] * hz
    if not np.isclose(substeps, round(substeps)) or substeps < 1:
        raise ValueError("Multi-object trace/control rates are incompatible")
    phases = episode["phase"].asstr()[:]
    if len(phases) != episode.attrs["num_samples"]:
        raise ValueError("Multi-object action phases have inconsistent lengths")
    stop_actions = np.repeat(phases == "stop", round(substeps))
    if len(times) < len(stop_actions) or not np.array_equal(trace["stop_requested"][: len(stop_actions)], stop_actions):
        raise ValueError("Multi-object stop actions differ from physics-rate evidence")
    initial = {k: v[()] for k, v in episode["initial_state"].items()}
    evaluator = MultiObjectEvaluator(scenario, initial)
    arrays = {k: v[:] for k, v in trace.items()}
    for values in arrays.values():
        if len(values) != len(times) or not np.isfinite(values).all():
            raise ValueError("Invalid trajectory evidence")
    for i in range(len(times)):
        if "diagnostic_intervention" in scenario:
            applied = bool(arrays["intervention_applied"][i])
            if evaluator.intervention_applied and not applied:
                raise ValueError("Diagnostic intervention cannot be undone")
            if applied and not evaluator.intervention_applied:
                if not i or len(evaluator.events) != 1 or evaluator.events[0]["object_id"] != "A":
                    raise ValueError("Diagnostic intervention must follow A's stable release")
                shift = arrays["object_pose_xyzw"][i, 0, :2] - arrays["object_pose_xyzw"][i - 1, 0, :2]
                if not np.allclose(shift, scenario["diagnostic_intervention"]["offset_xy_m"], atol=0.001):
                    raise ValueError("Recorded reference perturbation differs from requested offset")
            evaluator.intervention_applied = applied
        obs = {
            k: arrays[k][i]
            for k in ("object_pose_xyzw", "object_velocity", "object_aabb_min", "object_aabb_max", "hand_pose_xyzw")
        }
        for j, (obj, pose) in enumerate(zip(scenario["objects"], obs["object_pose_xyzw"])):
            lo, hi = world_aabb(obj["geometry"], pose[:3], pose[3:])
            if not np.allclose(lo, obs["object_aabb_min"][j], atol=1e-6) or not np.allclose(
                hi, obs["object_aabb_max"][j], atol=1e-6
            ):
                raise ValueError("Multi-object bounds differ from authored geometry")
        success = evaluator.update(
            obs, arrays["finger_positions"][i], "stop" if arrays["stop_requested"][i] else "replay", 1 / hz
        )
        if success != bool(arrays["success"][i]):
            raise ValueError("Multi-object success differs from replay")
    if metadata["long_horizon_metrics"] != evaluator.metrics or bool(metadata["success"]) != (
        evaluator.metrics["success"] and metadata["termination_reason"] == "success"
    ):
        raise ValueError("Multi-object event history/outcome differs from replay")
