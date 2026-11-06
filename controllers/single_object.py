# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-feedback numerical transport on the existing Franka grasp/release controller."""

import numpy as np
from tasks.control.goals import angle_error, yaw

from controllers.grasp_lift import GraspLiftController
from controllers.pick_place import PickPlaceController


class SingleObjectController(PickPlaceController):
    """Follow a validated primitive route; no teleportation or grasp attachment."""

    def __init__(self, initial_pose, width, resolved, protocol, settings, grasp_yaw, object_size):
        goal_protocol = dict(protocol, goal_center_xy_m=resolved["target_position_m"][:2])
        super().__init__(
            initial_pose[:3],
            width,
            goal_xy=resolved["target_position_m"][:2],
            clearance_z=resolved["transport_z_m"] + settings.HAND_TO_GRASP,
            protocol=goal_protocol,
            settings=settings,
        )
        self.resolved = resolved
        self.route = [np.asarray(p, dtype=float) for p in resolved["oracle_route_m"]]
        self.route_index = 0
        self.route_dwell = 0.0
        self.initial_yaw = yaw(initial_pose[3:])
        self.grasp_yaw = grasp_yaw
        self.command_yaw = grasp_yaw
        self.object_size = np.asarray(object_size)
        self.rotation = np.array([np.cos(grasp_yaw / 2), np.sin(grasp_yaw / 2), 0.0, 0.0])
        # Held-height tasks must not first overshoot to the generic 18 cm lift.
        self.positions["lift"][2] = resolved["transport_z_m"] + settings.HAND_TO_GRASP
        self.positions["hold"] = self.positions["lift"].copy()

    def command(
        self,
        hand_pose,
        finger_positions,
        finger_velocities,
        *,
        object_bounds=None,
        object_velocity=None,
        object_pose=None,
    ):
        if self.phase == "close":
            delta = yaw(object_pose[3:]) - self.grasp_yaw
            self.object_width = min(
                0.07, abs(np.sin(delta)) * self.object_size[0] + abs(np.cos(delta)) * self.object_size[1]
            )
        if self.phase in ("approach", "descend", "close", "lift"):
            pose, finger, phase = GraspLiftController.command(self, hand_pose, finger_positions, finger_velocities)
            if phase != "hold":
                return self._limit_motion(pose, finger, phase, hand_pose)
            self.phase = "transport"
        if self.phase in ("transport", "hold"):
            target = self.route[self.route_index]
            # Adjust hand goal using the measured held-object error. This is an
            # oracle demonstration controller, not an image-based policy.
            hand_goal = hand_pose[:3] + target - object_pose[:3]
            measured_hand_yaw = 2 * np.arctan2(hand_pose[4], hand_pose[3])
            measured_hand_yaw = self.command_yaw + (measured_hand_yaw - self.command_yaw + np.pi) % (2 * np.pi) - np.pi
            object_error = (self.resolved["target_yaw_rad"] - yaw(object_pose[3:]) + np.pi) % (2 * np.pi) - np.pi
            requested = measured_hand_yaw + object_error
            self.command_yaw += np.clip(
                requested - self.command_yaw,
                -self.protocol["rotation_speed_rad_s"] * self.control_dt,
                self.protocol["rotation_speed_rad_s"] * self.control_dt,
            )
            self.rotation = np.array([np.cos(self.command_yaw / 2), np.sin(self.command_yaw / 2), 0.0, 0.0])
            reached = (
                np.linalg.norm(target - object_pose[:3]) <= 0.005
                and angle_error(yaw(object_pose[3:]), self.resolved["target_yaw_rad"]) <= 0.04
            )
            self.route_dwell = self.route_dwell + self.control_dt if reached else 0.0
            if self.phase == "transport" and self.route_dwell >= 0.10:
                self.route_dwell = 0.0
                if self.route_index + 1 < len(self.route):
                    self.route_index += 1
                elif self.resolved["final_mode"] == "hold":
                    self.phase = "hold"
                else:
                    self.phase = "lower"
                    placed = hand_pose[:3] + np.asarray(self.resolved["target_position_m"]) - object_pose[:3]
                    placed[2] += 0.001
                    self.positions["lower"] = placed.copy()
                    self.positions["release"] = placed.copy()
                    retreat = placed.copy()
                    retreat[2] += self.protocol["retreat_m"]
                    self.positions["retreat"] = retreat.copy()
                    self.positions["settled"] = retreat.copy()
            return self._limit_motion(np.concatenate((hand_goal, self.rotation)), 0.0, self.phase, hand_pose)
        return super().command(
            hand_pose, finger_positions, finger_velocities, object_bounds=object_bounds, object_velocity=object_velocity
        )
