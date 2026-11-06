# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned placement geometry and stateful success evidence."""

from pathlib import Path

import numpy as np
import yaml

PROTOCOL = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/protocols/pick_place.yaml").read_text())


def goal_bounds(protocol=PROTOCOL):
    centre = np.asarray(protocol["goal_center_xy_m"])
    half = np.asarray(protocol["goal_size_xy_m"]) / 2
    return centre - half, centre + half


class PlacementEvaluator:
    """Accumulate whole-trajectory evidence; a returned success needs every gate.

    A conservative sphere encloses the open hand and fingers. Separation from
    that sphere proves release geometrically without claiming contact sensing.
    The table is the goal support surface. A bottom bound near that surface,
    containment, and a continuous low-velocity dwell reject suspended targets.
    """

    def __init__(self, initial_centres, target, table_height, protocol=PROTOCOL):
        self.initial = np.asarray(initial_centres).copy()
        self.target = target
        self.table_height = table_height
        self.protocol = protocol
        self.peak_lifts = np.zeros(len(initial_centres))
        self.max_displacement = np.zeros(len(initial_centres))
        self.dwell = 0.0
        self.metrics = {}

    def update(self, observation, fingers, phase, dt):
        p = self.protocol
        centres = observation["object_pose_xyzw"][:, :3]
        self.peak_lifts = np.maximum(self.peak_lifts, centres[:, 2] - self.initial[:, 2])
        self.max_displacement = np.maximum(self.max_displacement, np.linalg.norm(centres - self.initial, axis=1))
        others = np.arange(len(centres)) != self.target
        wrong = np.flatnonzero(others & (self.peak_lifts >= p["selection_lift_m"]))
        lo = observation["object_aabb_min"][self.target]
        hi = observation["object_aabb_max"][self.target]
        goal_lo, goal_hi = goal_bounds(p)
        margin = p["containment_margin_m"]
        contained = bool(np.all(lo[:2] >= goal_lo + margin) and np.all(hi[:2] <= goal_hi - margin))
        hand = observation["hand_pose_xyzw"][:3]
        separation = float(np.linalg.norm(hand - np.clip(hand, lo, hi)))
        released = bool(
            phase in ("retreat", "settled")
            and np.sum(fingers) >= 0.075
            and separation > p["gripper_enclosing_radius_m"]
        )
        velocity = observation["object_velocity"][self.target]
        supported = abs(float(lo[2]) - self.table_height) <= p["support_tolerance_m"]
        slow = bool(
            np.linalg.norm(velocity[:3]) <= p["linear_speed_m_s"]
            and np.linalg.norm(velocity[3:]) <= p["angular_speed_rad_s"]
        )
        lifted = self.peak_lifts[self.target] >= p["lift_height_m"]
        preserved = bool(np.all(self.max_displacement[others] <= p["max_distractor_movement_m"]))
        self.dwell = self.dwell + dt if released and contained and supported and slow else 0.0
        success = bool(lifted and not len(wrong) and preserved and self.dwell >= p["stable_hold_s"])
        self.metrics = {
            "correct_selection": bool(lifted and not len(wrong)),
            "wrong_object_indices": wrong.tolist(),
            "lifted": bool(lifted),
            "released": released,
            "contained": contained,
            "supported": bool(supported),
            "slow": slow,
            "stable_seconds": self.dwell,
            "distractors_preserved": preserved,
            "max_displacement_m": self.max_displacement.tolist(),
            "peak_lifts_m": self.peak_lifts.tolist(),
            "gripper_separation_m": separation,
            "grasp_evidence": "lift_and_motion_proxy; no contact sensor",
            "success": success,
        }
        return success
