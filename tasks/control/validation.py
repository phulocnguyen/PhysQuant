# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay physics-rate evidence when validating recorded control outcomes."""

import numpy as np
from assets.geometry import world_aabb

from tasks.control.evaluator import ControlEvaluator


def validate_recording(episode, scenario, metadata):
    for key in ("episode_id", "scene_id", "template_id", "template_version", "manipulation_protocol"):
        if metadata.get(key) != scenario[key]:
            raise ValueError(f"Control metadata {key} differs from scenario")
    if metadata.get("success_protocol") != "spatial_control_v1":
        raise ValueError("Control recording requires its explicit success protocol")
    if "control_trace" not in episode:
        raise ValueError("Spatial control recording lacks physics-rate trajectory evidence")
    trace = episode["control_trace"]
    times = trace["timestamp_s"][:]
    hz = metadata.get("control_sampling_hz", 0)
    if (
        not len(times)
        or hz <= 0
        or not np.isfinite(times).all()
        or not np.allclose(np.diff(np.r_[0.0, times]), 1 / hz, atol=1e-8)
    ):
        raise ValueError("Control trace is incomplete or irregular")
    if metadata["success"] and not np.isclose(times[-1], metadata["episode_duration_s"], atol=1e-8):
        raise ValueError("Successful control trace must cover exactly the completed action transitions")
    for dataset in trace.values():
        if len(dataset) != len(times) or not np.isfinite(dataset[:]).all():
            raise ValueError("Invalid control trace values/lengths")
    initial = {key: dataset[()] for key, dataset in episode["initial_state"].items()}
    evaluator = ControlEvaluator(scenario, initial)
    fields = ("object_pose_xyzw", "object_velocity", "object_aabb_min", "object_aabb_max", "hand_pose_xyzw")
    arrays = {key: trace[key][:] for key in fields}
    fingers = trace["finger_positions"][:]
    recorded_success = trace["success"][:]
    recorded_transport = trace["transport_active"][:]
    recorded_violations = trace["constraint_violation"][:]
    # Recompute bounds: a fabricated point-sized AABB must not evade clearance scoring.
    for i in range(len(times)):
        obs = {key: values[i] for key, values in arrays.items()}
        for j, (obj, pose) in enumerate(zip(scenario["objects"], obs["object_pose_xyzw"])):
            lo, hi = world_aabb(obj["geometry"], pose[:3], pose[3:])
            if not np.allclose(lo, obs["object_aabb_min"][j], atol=1e-6) or not np.allclose(
                hi, obs["object_aabb_max"][j], atol=1e-6
            ):
                raise ValueError("Control trace bounds differ from complete geometry")
        success = evaluator.update(obs, fingers[i], "replay", 1 / hz)
        if (
            success != bool(recorded_success[i])
            or evaluator.trace[-1]["transport_active"] != bool(recorded_transport[i])
            or not np.isclose(evaluator.trace[-1]["constraint_violation"], recorded_violations[i], atol=1e-8)
        ):
            raise ValueError("Control trace score differs from replay")
    # Interrupted recordings may retain physics evidence after the final complete
    # action, but must never claim success on those incomplete transitions.
    expected = bool(evaluator.metrics["success"]) and metadata.get("termination_reason") == "success"
    if bool(metadata["success"]) != expected or metadata.get("control_metrics") != evaluator.metrics:
        raise ValueError("Control outcome/metrics differ from physics-rate replay")
