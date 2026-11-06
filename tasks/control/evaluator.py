# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physics-rate numerical and trajectory scoring, independent of oracle progress flags."""

from copy import deepcopy

import numpy as np

from tasks.control.goals import angle_error, horizontal_distance, solve, yaw


class ControlEvaluator:
    """Measure actual object motion; controller phase labels do not establish success.

    Version 2 arms transport after a held lift, then starts at the first >5 mm
    horizontal departure from that acquired pose. It ends after
    settling over the endpoint for 0.1 s while held, before lowering. Leaving
    that endpoint again reopens the interval. Every sample within these intervals
    contributes to permanent violation maxima; final recovery cannot erase them.
    Grasp evidence remains a lift/proximity/closed-finger proxy, not contact proof.
    """

    def __init__(self, scenario, initial):
        self.scenario = deepcopy(scenario)
        self.initial = initial["object_pose_xyzw"].copy()
        for obj, pose in zip(self.scenario["objects"], self.initial):
            obj["position_m"] = pose[:3].tolist()
            obj["rotation_xyzw"] = pose[3:].tolist()
        self.goal = solve(scenario["case"], scenario["numerical_goal"], self.scenario["objects"], scenario["goal"])
        self.protocol = scenario["goal"]
        self.peak_lifts = np.zeros(len(self.initial))
        self.max_displacement = np.zeros(len(self.initial))
        self.dwell = 0.0
        self.time = 0.0
        self.waypoint_index = 0
        self.waypoint_errors = [None] * len(self.goal["waypoints_m"])
        self.waypoint_order_violation = False
        self.transport_started = False
        self.transport_finished = False
        self.transport_anchor = None
        self.pretransport_violation = False
        self.end_dwell = 0.0
        self.transport_samples = 0
        self.violation_samples = 0
        self.violation_duration = 0.0
        self.max_violation = 0.0
        self.total_violation = 0.0
        self.min_clearance = None
        self.max_yaw_deviation = 0.0
        self.total_yaw_deviation = 0.0
        self.retention_lost = False
        self.previous_yaw = yaw(self.initial[0, 3:])
        self.accumulated_yaw = 0.0
        self.path_length = 0.0
        self.previous_position = self.initial[0, :3].copy()
        self.previous_rotation = self.initial[0, 3:].astype(float)
        self.metrics = {}
        self.trace = []

    def update(self, observation, fingers, phase, dt):
        self.time += dt
        p = self.protocol
        g = self.goal
        case = self.scenario["case"]
        poses = observation["object_pose_xyzw"]
        centre = poses[0, :3]
        lo = observation["object_aabb_min"][0]
        hi = observation["object_aabb_max"][0]
        vel = observation["object_velocity"][0]
        self.peak_lifts = np.maximum(self.peak_lifts, poses[:, 2] - self.initial[:, 2])
        self.max_displacement = np.maximum(
            self.max_displacement, np.linalg.norm(poses[:, :3] - self.initial[:, :3], axis=1)
        )
        travelled = float(np.linalg.norm(centre - self.previous_position))
        pose_linear_speed = travelled / dt
        rotation = poses[0, 3:].astype(float)
        rotation /= np.linalg.norm(rotation)
        previous_rotation = self.previous_rotation / np.linalg.norm(self.previous_rotation)
        pose_angular_speed = float(2 * np.arccos(np.clip(abs(np.dot(rotation, previous_rotation)), 0, 1)) / dt)
        self.previous_rotation = rotation.copy()
        self.path_length += travelled
        self.previous_position = centre.copy()
        current_yaw = yaw(poses[0, 3:])
        self.accumulated_yaw += (current_yaw - self.previous_yaw + np.pi) % (2 * np.pi) - np.pi
        self.previous_yaw = current_yaw
        hand = observation["hand_pose_xyzw"][:3]
        hand_distance = float(np.linalg.norm(hand - centre))
        grasp = bool(np.sum(fingers) < 0.075 and hand_distance < 0.18 and centre[2] - self.initial[0, 2] > 0.025)
        lifted = bool(self.peak_lifts[0] >= p["lift_height_m"])
        wrong = bool(np.any(self.peak_lifts[1:] >= p["selection_lift_m"]))
        preserved = bool(np.all(self.max_displacement[1:] <= p["max_distractor_movement_m"]))
        position_vector = centre - np.asarray(g["target_position_m"])
        position_error = float(np.linalg.norm(position_vector))
        rotation_error = angle_error(current_yaw, g["target_yaw_rad"])
        stable = bool(
            np.linalg.norm(vel[:3]) <= p["linear_speed_m_s"] and np.linalg.norm(vel[3:]) <= p["angular_speed_rad_s"]
        )
        if g["final_mode"] == "hold" and p.get("evidence_version", 1) >= 2:
            # Contact solver velocities can remain nonzero while successive poses
            # are stationary. Score held stability from actual physics-step motion;
            # retain the raw velocities in the trace for independent inspection.
            stable = bool(pose_linear_speed <= p["linear_speed_m_s"] and pose_angular_speed <= p["angular_speed_rad_s"])
        numerical = position_error <= p["position_tolerance_m"]
        if case.startswith("ST-2"):
            numerical &= rotation_error <= p["rotation_tolerance_rad"]
        clearance_errors = []
        if case.startswith("TP-2"):
            c = g["constraints"]
            if case == "TP-2C":
                actual = [lo[0] - observation["object_aabb_max"][1, 0], observation["object_aabb_min"][2, 0] - hi[0]]
                expected = c["gaps_m"]
            else:
                axis = c["axis"]
                sign = c["sign"]
                actual = [
                    lo[axis] - observation["object_aabb_max"][1, axis]
                    if sign > 0
                    else observation["object_aabb_min"][1, axis] - hi[axis]
                ]
                expected = [c["gap_m"]]
            clearance_errors = [float(a - b) for a, b in zip(actual, expected)]
            numerical &= (
                all(a >= 0 for a in actual)
                and max(map(abs, clearance_errors)) <= p["clearance_tolerance_m"]
                and rotation_error <= 0.05
            )
        waypoints = np.asarray(g["waypoints_m"])
        if len(waypoints) and grasp:
            errors = np.linalg.norm(waypoints - centre, axis=1)
            if self.waypoint_index < len(waypoints):
                if np.any(errors[self.waypoint_index + 1 :] <= p["waypoint_tolerance_m"]):
                    self.waypoint_order_violation = True
                if errors[self.waypoint_index] <= p["waypoint_tolerance_m"]:
                    self.waypoint_errors[self.waypoint_index] = float(errors[self.waypoint_index])
                    self.waypoint_index += 1
        ordered = self.waypoint_index == len(waypoints) and not self.waypoint_order_violation
        transport_active = False
        violation = 0.0
        clearance = None
        if case.startswith("CM-2"):
            xy_error = float(np.linalg.norm(position_vector[:2]))
            anchor = self.initial[0, :2]
            if p.get("evidence_version", 1) >= 2:
                if self.transport_anchor is None:
                    self.pretransport_violation |= bool(
                        np.linalg.norm(centre[:2] - anchor) > p["pretransport_max_horizontal_drift_m"]
                    )
                    acquired = grasp and centre[2] - self.initial[0, 2] >= p["lift_height_m"]
                    if case == "CM-2A":
                        acquired &= (
                            lo[2] - self.scenario["reference_frame"]["origin_m"][2]
                            >= g["constraints"]["minimum_clearance_m"]
                        )
                    if acquired:
                        self.transport_anchor = centre[:2].copy()
                anchor = centre[:2] if self.transport_anchor is None else self.transport_anchor
            departure = np.linalg.norm(centre[:2] - anchor) > p["transport_start_displacement_m"]
            self.transport_started |= bool(departure)
            if self.transport_finished and xy_error > p["transport_end_tolerance_m"] * 2:
                self.transport_finished = False
                self.end_dwell = 0.0
            transport_active = self.transport_started and not self.transport_finished
            if transport_active:
                self.transport_samples += 1
                self.retention_lost |= not grasp
                if case == "CM-2A":
                    clearance = float(lo[2] - self.scenario["reference_frame"]["origin_m"][2])
                    violation = max(0.0, g["constraints"]["minimum_clearance_m"] - clearance)
                elif case == "CM-2B":
                    deviation = angle_error(current_yaw, g["initial_yaw_rad"])
                    self.max_yaw_deviation = max(self.max_yaw_deviation, deviation)
                    self.total_yaw_deviation += deviation
                    violation = max(0.0, deviation - g["constraints"]["max_yaw_deviation_rad"])
                else:
                    clearance = horizontal_distance(
                        lo, hi, observation["object_aabb_min"][1], observation["object_aabb_max"][1]
                    )
                    violation = max(0.0, g["constraints"]["minimum_obstacle_clearance_m"] - clearance)
                if clearance is not None:
                    self.min_clearance = clearance if self.min_clearance is None else min(self.min_clearance, clearance)
                self.max_violation = max(self.max_violation, violation)
                self.total_violation += violation
                violating = violation > p["constraint_numeric_epsilon"]
                self.violation_samples += int(violating)
                self.violation_duration += dt * violating
                stopped = (
                    xy_error <= p["transport_end_tolerance_m"]
                    and np.linalg.norm(vel[:2]) <= p["linear_speed_m_s"]
                    and grasp
                )
                self.end_dwell = self.end_dwell + dt if stopped else 0.0
                self.transport_finished = self.end_dwell >= p["transport_end_dwell_s"]
        path_valid = ordered and (
            not case.startswith("CM-2")
            or self.transport_finished
            and self.transport_samples > 0
            and not self.violation_samples
            and not self.retention_lost
        )
        if len(waypoints) and 0 < self.waypoint_index < len(waypoints) and not grasp:
            self.retention_lost = True
        path_valid &= not self.retention_lost and not self.pretransport_violation
        separation = float(np.linalg.norm(hand - np.clip(hand, lo, hi)))
        released = bool(np.sum(fingers) >= 0.075 and separation > p["gripper_enclosing_radius_m"])
        supported = abs(lo[2] - self.scenario["reference_frame"]["origin_m"][2]) <= p["support_tolerance_m"]
        final = grasp if g["final_mode"] == "hold" else released and supported
        # A nearly horizontal local Z axis is not an upright table-supported pose.
        q = poses[0, 3:]
        upright = 1 - 2 * (q[0] * q[0] + q[1] * q[1]) >= np.cos(np.deg2rad(10))
        eligible = bool(
            numerical and stable and final and upright and path_valid and lifted and not wrong and preserved
        )
        self.dwell = self.dwell + dt if eligible else 0.0
        success = bool(self.dwell >= p["stable_hold_s"])
        self.metrics = dict(
            success=success,
            numerical_success=bool(numerical),
            position_error_m=position_error,
            signed_position_error_m=position_vector.tolist(),
            actual_displacement_m=(centre - self.initial[0, :3]).tolist(),
            rotation_error_rad=float(rotation_error),
            achieved_yaw_change_rad=float(self.accumulated_yaw),
            clearance_signed_errors_m=clearance_errors,
            correct_selection=bool(lifted and not wrong),
            lifted=lifted,
            released=released,
            supported=bool(supported),
            held=grasp,
            slow=stable,
            upright=bool(upright),
            stable_seconds=self.dwell,
            distractors_preserved=preserved,
            waypoints_reached=self.waypoint_index,
            waypoint_errors_m=self.waypoint_errors.copy(),
            waypoint_order_violation=self.waypoint_order_violation,
            path_length_m=self.path_length,
            trajectory_success=bool(path_valid),
            transport_samples=self.transport_samples,
            transport_finished=self.transport_finished,
            minimum_clearance_m=self.min_clearance,
            maximum_constraint_violation=self.max_violation,
            mean_constraint_violation=self.total_violation / max(1, self.transport_samples),
            violation_duration_s=self.violation_duration,
            violating_sample_fraction=self.violation_samples / max(1, self.transport_samples),
            maximum_yaw_deviation_rad=self.max_yaw_deviation,
            mean_yaw_deviation_rad=self.total_yaw_deviation / max(1, self.transport_samples),
            grasp_retention_lost=bool(self.retention_lost),
            grasp_evidence="lift/proximity/closed-finger proxy; no contact sensor",
        )
        if p.get("evidence_version", 1) >= 2:
            self.metrics.update(
                pose_linear_speed_m_s=pose_linear_speed,
                pose_angular_speed_rad_s=pose_angular_speed,
                pretransport_violation=self.pretransport_violation,
            )
        self.trace.append(
            dict(
                timestamp_s=self.time,
                object_pose_xyzw=poses.copy(),
                object_velocity=observation["object_velocity"].copy(),
                object_aabb_min=observation["object_aabb_min"].copy(),
                object_aabb_max=observation["object_aabb_max"].copy(),
                hand_pose_xyzw=observation["hand_pose_xyzw"].copy(),
                finger_positions=np.asarray(fingers).copy(),
                transport_active=bool(transport_active),
                constraint_violation=float(violation),
                success=success,
            )
        )
        return success
