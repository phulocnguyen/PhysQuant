# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sequential oracle actions grounded in achieved object/support poses."""

from copy import copy

import numpy as np
from assets.geometry import rotate_vector
from tasks.control.goals import yaw
from tasks.multi_object.planning import destination

from controllers.single_object import SingleObjectController


class MultiObjectController:
    def __init__(self, scenario, evaluator, settings):
        self.scenario = scenario
        self.evaluator = evaluator
        self.settings = settings
        plan = scenario["plan"]
        self.actions = []
        slots = {}
        for group, indices in enumerate(plan["solutions"][0]):
            for ordinal, i in enumerate(indices):
                b = plan["assignments"].get(scenario["objects"][i]["object_id"], group)
                slot = slots.get(b, 0) if plan["mode"] == "basket" else ordinal
                self.actions.append((i, b, slot))
                slots[b] = slot + 1
        self.index = 0
        self.child = None
        self.placed = {}
        self.last_pose = None
        self.phase = "approach"

    def command(self, hand_pose, fingers, velocities, *, observation, **unused):
        if self.scenario["case"].startswith("FDP-1") and self.child is None:
            self._feedback_plan(observation)
        if self.index >= len(self.actions):
            self.phase = "stop"
            return self.last_pose.copy(), 0.04, "stop"
        i, group, ordinal = self.actions[self.index]
        s = self.scenario
        obj = s["objects"][i]
        if self.child is None:
            point, angle = destination(s, i, group, ordinal, observation, self.placed.get(group, []))
            if s["plan"]["mode"] == "row" and ordinal:
                # A small insertion clearance prevents an edge from landing on
                # the preceding block. It remains inside the explicit 5 mm
                # adjacency and total-length tolerances; scoring is unchanged.
                count = len(s["plan"]["solutions"][0][group])
                point[0] += 0.003 / max(1, count - 1)
            pose = observation["object_pose_xyzw"][i]
            base = np.deg2rad(self.settings.ROBOT_YAW_DEG)
            grasp = base + (yaw(pose[3:]) + obj.get("grasp_yaw_offset_rad", 0) - base + np.pi / 2) % np.pi - np.pi / 2
            # The symmetric overhead grasp at yaw zero reaches Panda joint 7's
            # upper limit over the near-left basket. Use its equivalent pi yaw.
            if grasp < base:
                grasp += np.pi
            clearance = max(float(np.max(observation["object_aabb_max"][:, 2])) + 0.10, point[2] + 0.13, pose[2] + 0.13)
            resolved = dict(
                target_position_m=point,
                transport_z_m=clearance,
                target_yaw_rad=angle,
                final_mode="release",
                oracle_route_m=[[*point[:2], clearance]],
            )
            settings = copy(self.settings)
            settings.TABLE_HEIGHT = point[2] - obj["size_m"][2] / 2
            grasp_pose = pose.copy()
            # Use the asset's upper pinch point. Center grasps can hit the palm
            # on tall blocks or adjacent placed objects during descending sorts.
            offset = rotate_vector(obj["grasp_offset_local_m"], pose[3:])
            grasp_pose[:3] += offset
            self.child = SingleObjectController(
                grasp_pose, obj["grasp_width_m"], resolved, s["goal"], settings, grasp, obj["size_m"]
            )
            for stage in ("lift", "hold"):
                self.child.positions[stage][2] += offset[2]
        if s["plan"]["mode"] == "row" and self.child.phase == "lower":
            # Keep the held object's measured XY over the narrow insertion gap.
            # A fixed hand target otherwise retains small grasp-slip offsets.
            correction = (
                np.asarray(self.child.resolved["target_position_m"][:2]) - observation["object_pose_xyzw"][i, :2]
            )
            self.child.positions["lower"][:2] = hand_pose[:2] + correction
            self.child.positions["release"][:2] = hand_pose[:2] + correction
        pose, finger, phase = self.child.command(
            hand_pose,
            fingers,
            velocities,
            object_pose=observation["object_pose_xyzw"][i],
            object_bounds=(observation["object_aabb_min"][i], observation["object_aabb_max"][i]),
            object_velocity=observation["object_velocity"][i],
        )
        self.last_pose = pose.copy()
        self.phase = f"{obj['object_id']}:{phase}"
        if phase == "settled" and obj["object_id"] in [e["object_id"] for e in self.evaluator.events]:
            self.placed.setdefault(group, []).append(i)
            self.index += 1
            self.child = None
        return pose, finger, self.phase

    def _feedback_plan(self, observation):
        """Recompute the remaining numerical allocation from confirmed placements."""
        s = self.scenario
        identifiers = [o["object_id"] for o in s["objects"][: s["candidate_count"]]]
        used = [identifiers.index(event["object_id"]) for event in self.evaluator.events]
        if self.evaluator.metrics.get("numerical_success") and len(used) >= 2:
            self.actions = self.actions[: self.index]
            return
        solutions = [solution[0] for solution in s["plan"]["solutions"] if set(used).issubset(solution[0])]
        if s["case"] == "FDP-1C" and used:
            height = (
                float(max(observation["object_aabb_max"][i, 2] for i in used)) - s["reference_frame"]["origin_m"][2]
            )
            solutions = [
                indices
                for indices in solutions
                if abs(
                    height
                    + sum(s["objects"][i]["size_m"][2] for i in indices if i not in used)
                    - s["numerical_goal"]["target"] / 100
                )
                <= s["goal"]["dimension_tolerance_m"]
            ]
        remaining = next(
            ([i for i in indices if i not in used] for indices in solutions if any(i not in used for i in indices)),
            None,
        )
        if remaining is None:
            # Stop is an explicit decision; the independent evaluator labels this
            # an unsuccessful premature/invalid stop when the goal is not met.
            self.actions = self.actions[: self.index]
        else:
            self.actions = self.actions[: self.index] + [(remaining[0], 0, len(used))]
