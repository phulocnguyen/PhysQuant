# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feedback-driven approach, descend, close, lift, and hold controller."""

import numpy as np
from environments import settings as defaults


class GraspLiftController:
    """Advance when measured motion completes, without scheduled phase durations."""

    def __init__(self, target_centre, object_width: float, settings=defaults):
        self.settings = settings
        self.object_width = object_width
        self.phase = "approach"
        grasp = np.asarray(target_centre, dtype=float).copy()
        grasp[2] += settings.HAND_TO_GRASP
        above = grasp.copy()
        above[2] += settings.APPROACH_CLEARANCE
        lifted = grasp.copy()
        lifted[2] += settings.LIFT_DISTANCE
        self.positions = {"approach": above, "descend": grasp, "close": grasp, "lift": lifted, "hold": lifted}
        self.rotation = np.array([1.0, 0.0, 0.0, 0.0])

    def command(self, hand_pose, finger_positions, finger_velocities):
        """Return world hand pose, finger target [m], and feedback-selected phase."""
        position_error = np.linalg.norm(hand_pose[:3] - self.positions[self.phase])
        orientation_error = 2 * np.arccos(np.clip(abs(np.dot(hand_pose[3:], self.rotation)), 0, 1))
        reached = (
            position_error <= self.settings.POSITION_TOLERANCE
            and orientation_error <= self.settings.ORIENTATION_TOLERANCE
        )
        if self.phase == "approach" and reached and np.sum(finger_positions) >= 0.07:
            self.phase = "descend"
        elif self.phase == "descend" and reached:
            self.phase = "close"
        elif self.phase == "close" and reached:
            closed = np.sum(finger_positions) <= self.object_width + 0.002
            stopped = np.max(np.abs(finger_velocities)) <= self.settings.GRIPPER_VELOCITY_TOLERANCE
            if closed and stopped:
                self.phase = "lift"
        elif self.phase == "lift" and reached:
            self.phase = "hold"
        finger = 0.04 if self.phase in ("approach", "descend") else 0.0
        return np.concatenate((self.positions[self.phase], self.rotation)), finger, self.phase
