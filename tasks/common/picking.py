# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-and-hold success, with physics-rate evidence for independent replay."""

from pathlib import Path

import numpy as np
import yaml
from assets.geometry import rotate_vector, world_aabb
from environments import settings

PROTOCOL = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/protocols/pick_hold.yaml").read_text())
TRACE_FIELDS = (
    "object_pose_xyzw",
    "object_velocity",
    "object_aabb_min",
    "object_aabb_max",
    "hand_pose_xyzw",
    "joint_position",
    "joint_velocity",
)


class PickEvaluator:
    """Require a current stable grasp; an earlier lift alone cannot establish success.

    Finger aperture and the grasp point inside the target bounds are geometric
    grasp evidence, not a contact-force measurement. Distractor excursions latch
    over the complete trajectory, including references that cannot be selected.
    """

    def __init__(self, scenario, initial_observation):
        self.protocol = scenario["goal"]
        self.target = scenario["target_index"]
        self.width = scenario["objects"][self.target]["grasp_width_m"]
        self.initial = initial_observation["object_pose_xyzw"][:, :3].copy()
        self.peak_lifts = np.zeros(len(self.initial))
        self.max_displacement = np.zeros(len(self.initial))
        self.dwell = 0.0
        self.hold_pose = None
        self.time = 0.0
        self.metrics = {}
        self.trace = []

    def update(self, observation, fingers, phase, dt):
        p = self.protocol
        centres = observation["object_pose_xyzw"][:, :3]
        lifts = centres[:, 2] - self.initial[:, 2]
        self.peak_lifts = np.maximum(self.peak_lifts, lifts)
        self.max_displacement = np.maximum(self.max_displacement, np.linalg.norm(centres - self.initial, axis=1))
        others = np.arange(len(centres)) != self.target
        wrong = np.flatnonzero(others & (self.peak_lifts >= p["selection_lift_m"]))
        preserved = bool(np.all(self.max_displacement[others] <= p["max_distractor_movement_m"]))
        lo = observation["object_aabb_min"][self.target]
        hi = observation["object_aabb_max"][self.target]
        hand = observation["hand_pose_xyzw"]
        grasp = hand[:3] + rotate_vector(settings.ROBOT_TCP_OFFSET, hand[3:])
        tolerance = p["grasp_position_tolerance_m"]
        fingers_closed = bool(0.002 < np.sum(fingers) <= self.width + p["finger_width_tolerance_m"])
        held = bool(fingers_closed and np.all(grasp >= lo - tolerance) and np.all(grasp <= hi + tolerance))
        lifted = bool(lifts[self.target] >= p["lift_height_m"])
        clear = bool(lo[2] >= settings.TABLE_HEIGHT + p["table_clearance_m"])
        pose = observation["object_pose_xyzw"][self.target]
        if held and lifted and clear:
            # Contact impulses can produce high instantaneous velocities while
            # a held pose barely changes. Require a continuous bounded pose
            # window instead; steady motion and a drop cannot accumulate dwell.
            if self.hold_pose is None or (
                np.linalg.norm(pose[:3] - self.hold_pose[:3]) > p["hold_position_tolerance_m"]
                or 2 * np.arccos(np.clip(abs(np.dot(pose[3:], self.hold_pose[3:])), 0, 1))
                > p["hold_orientation_tolerance_rad"]
            ):
                self.hold_pose = pose.copy()
                self.dwell = 0.0
            self.dwell += dt
        else:
            self.hold_pose = None
            self.dwell = 0.0
        success = bool(not len(wrong) and preserved and self.dwell >= p["stable_hold_s"])
        self.metrics = dict(
            correct_selection=bool(lifted and not len(wrong)),
            wrong_object_indices=wrong.tolist(),
            lifted=lifted,
            held=held,
            clear_of_table=clear,
            fingers_closed=fingers_closed,
            stable_seconds=self.dwell,
            distractors_preserved=preserved,
            max_displacement_m=self.max_displacement.tolist(),
            peak_lifts_m=self.peak_lifts.tolist(),
            grasp_evidence="lift, aperture and grasp-point proxy; no contact sensor",
            success=success,
        )
        self.time += dt
        self.trace.append(
            dict(
                timestamp_s=self.time,
                **{key: observation[key].copy() for key in TRACE_FIELDS},
                finger_positions=np.asarray(fingers).copy(),
                success=success,
            )
        )
        return success


def validate_recording(episode, scenario, metadata):
    """Recompute pick outcomes from saved physics states, including failed attempts."""
    for key in ("episode_id", "scene_id", "template_id", "template_version", "manipulation_protocol"):
        if metadata.get(key) != scenario[key]:
            raise ValueError(f"Pick metadata {key} differs from scenario")
    if metadata.get("success_protocol") != PROTOCOL["id"]:
        raise ValueError("Pick recording requires its explicit success protocol")
    if not episode.attrs["num_samples"] and not metadata["success"] and "control_trace" not in episode:
        return
    if "control_trace" not in episode:
        raise ValueError("Pick recording lacks physics-rate trajectory evidence")
    trace = {key: value[:] for key, value in episode["control_trace"].items()}
    times = trace["timestamp_s"]
    hz = metadata.get("pick_sampling_hz", 0)
    if (
        not len(times)
        or not np.isclose(hz, 1 / settings.PHYSICS_DT)
        or any(len(values) != len(times) or not np.isfinite(values).all() for values in trace.values())
        or not np.allclose(np.diff(np.r_[0.0, times]), 1 / hz, atol=1e-8)
    ):
        raise ValueError("Pick trace is incomplete or irregular")
    if metadata["success"] and not np.isclose(times[-1], metadata["episode_duration_s"], atol=1e-8):
        raise ValueError("Successful pick trace must cover exactly the completed action transitions")
    initial = {key: value[()] for key, value in episode["initial_state"].items()}
    finger_ids = [i for i, name in enumerate(metadata["joint_names"]) if name.startswith("panda_finger_joint")]
    if len(finger_ids) != 2 or not np.array_equal(trace["finger_positions"], trace["joint_position"][:, finger_ids]):
        raise ValueError("Pick finger evidence differs from recorded robot state")
    evaluator = PickEvaluator(scenario, initial)
    for index in range(len(times)):
        obs = {key: trace[key][index] for key in TRACE_FIELDS}
        for obj_index, (obj, pose) in enumerate(zip(scenario["objects"], obs["object_pose_xyzw"])):
            lo, hi = world_aabb(obj["geometry"], pose[:3], pose[3:])
            if not np.allclose(lo, obs["object_aabb_min"][obj_index], atol=1e-6) or not np.allclose(
                hi, obs["object_aabb_max"][obj_index], atol=1e-6
            ):
                raise ValueError("Pick trace bounds differ from complete geometry")
        success = evaluator.update(obs, trace["finger_positions"][index], "replay", 1 / hz)
        if success != bool(trace["success"][index]):
            raise ValueError("Pick trace score differs from replay")
    expected = bool(evaluator.metrics["success"]) and metadata.get("termination_reason") == "success"
    if bool(metadata["success"]) != expected or metadata.get("pick_metrics") != evaluator.metrics:
        raise ValueError("Pick outcome/metrics differ from physics-rate replay")
