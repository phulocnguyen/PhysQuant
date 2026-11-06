# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feedback-driven placement following the existing grasp and lift phases."""

import numpy as np

from controllers.grasp_lift import GraspLiftController


class PickPlaceController(GraspLiftController):
    """Transport above clutter, lower to support, release, and retreat."""

    def __init__(self, target_centre, object_width, *, goal_xy, clearance_z, protocol, settings):
        super().__init__(target_centre, object_width, settings)
        placed = self.positions["descend"].copy()
        placed[:2] = goal_xy
        placed[2] += 0.002
        transfer = placed.copy()
        transfer[2] = max(self.positions["lift"][2], clearance_z)
        self.positions["lift"][2] = transfer[2]
        retreat = placed.copy()
        retreat[2] += protocol["retreat_m"]
        self.positions.update(transfer=transfer, lower=placed, release=placed, retreat=retreat, settled=retreat)
        self.control = settings.PLACEMENT_CONTROL
        self.protocol = protocol
        self.goal_xy = np.asarray(goal_xy)
        self.control_dt = settings.PHYSICS_DT * settings.CONTROL_STEPS
        self.previous_hand = None
        self.lower_hold = 0.0
        self.open_hold = 0.0
        self.command_position = None
        self.command_velocity = np.zeros(3)

    def command(self, hand_pose, finger_positions, finger_velocities, *, object_bounds=None, object_velocity=None):
        speed = (
            np.linalg.norm(hand_pose[:3] - self.previous_hand) / self.control_dt
            if self.previous_hand is not None
            else float("inf")
        )
        self.previous_hand = hand_pose[:3].copy()
        if self.phase in ("approach", "descend", "close", "lift"):
            pose, finger, phase = super().command(hand_pose, finger_positions, finger_velocities)
            if phase != "hold":
                return self._limit_motion(pose, finger, phase, hand_pose)
            self.phase = "transfer"
        error = np.linalg.norm(hand_pose[:3] - self.positions[self.phase])
        angle = 2 * np.arccos(np.clip(abs(np.dot(hand_pose[3:], self.rotation)), 0, 1))
        reached = error <= self.settings.POSITION_TOLERANCE and angle <= self.settings.ORIENTATION_TOLERANCE
        if self.phase == "transfer" and reached:
            self.phase = "lower"
        elif self.phase == "lower":
            if object_bounds is not None:
                lo, hi = object_bounds
                half = np.asarray(self.protocol["goal_size_xy_m"]) / 2 - self.protocol["containment_margin_m"]
                supported = abs(lo[2] - self.settings.TABLE_HEIGHT) <= self.protocol["support_tolerance_m"]
                contained = np.all(lo[:2] >= self.goal_xy - half) and np.all(hi[:2] <= self.goal_xy + half)
                if supported and contained:
                    # Hold the measured support pose instead of pressing a slipped grasp into the table.
                    self.positions["lower"] = hand_pose[:3].copy()
                    self.positions["release"] = hand_pose[:3].copy()
                # A closed grasp can cause vertical contact jitter and rocking.
                # Sustained support bounds height; slow lateral motion prevents sliding.
                # The evaluator checks full linear/angular stability after release.
                settled = (
                    supported and contained and np.linalg.norm(object_velocity[:2]) <= self.protocol["linear_speed_m_s"]
                )
            else:
                settled = error <= self.control["placement_position_tolerance_m"]
            settled = (
                settled
                and angle <= self.control["placement_orientation_tolerance_rad"]
                and speed <= self.control["hand_settle_speed_m_s"]
            )
            self.lower_hold = self.lower_hold + self.control_dt if settled else 0.0
            if self.lower_hold >= self.control["settle_before_release_s"]:
                self.phase = "release"
        elif self.phase == "release" and np.sum(finger_positions) >= 0.075:
            self.open_hold += self.control_dt
            if self.open_hold >= self.control["settle_after_open_s"]:
                self.phase = "retreat"
        elif self.phase == "retreat" and reached:
            self.phase = "settled"
        finger = 0.04 if self.phase in ("release", "retreat", "settled") else 0.0
        position = self.positions[self.phase]
        if self.phase == "lower":
            delta = position - hand_pose[:3]
            position = hand_pose[:3] + delta * min(
                1.0, self.control["lower_step_m"] / max(np.linalg.norm(delta), 1e-12)
            )
        return self._limit_motion(np.concatenate((position, self.rotation)), finger, self.phase, hand_pose)

    def _limit_motion(self, goal, finger, phase, hand_pose):
        """Move a held object with bounded command speed and acceleration."""
        if phase in ("approach", "descend", "close"):
            self.command_position = hand_pose[:3].copy()
            self.command_velocity[:] = 0
            return goal, finger, phase
        if self.command_position is None:
            self.command_position = hand_pose[:3].copy()
        delta = goal[:3] - self.command_position
        distance = np.linalg.norm(delta)
        acceleration = self.control["motion_acceleration_m_s2"]
        limit = self.control["lower_speed_m_s" if phase == "lower" else "motion_speed_m_s"]
        speed = min(limit, np.sqrt(2 * acceleration * distance))
        desired = delta * (speed / max(distance, 1e-12))
        change = desired - self.command_velocity
        self.command_velocity += change * min(1.0, acceleration * self.control_dt / max(np.linalg.norm(change), 1e-12))
        step = self.command_velocity * self.control_dt
        if np.linalg.norm(step) >= distance and np.dot(step, delta) > 0:
            self.command_position = goal[:3].copy()
            self.command_velocity[:] = 0
        else:
            self.command_position += step
        goal[:3] = self.command_position
        return goal, finger, phase
