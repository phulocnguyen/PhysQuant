# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical placement history and numerical end-state scoring at physics rate."""

from copy import deepcopy
from itertools import product

import numpy as np

from .planning import destination


class MultiObjectEvaluator:
    """Score measured state, not oracle phase completion; stop is an explicit action."""

    def __init__(self, scenario, initial):
        self.scenario = deepcopy(scenario)
        self.initial = initial["object_pose_xyzw"].copy()
        self.n = scenario["candidate_count"]
        self.time = 0.0
        self.dwell = 0.0
        self.peak_lifts = np.zeros(len(self.initial))
        self.max_displacement = np.zeros(len(self.initial))
        self.armed = np.zeros(self.n, dtype=bool)
        self.placement_dwell = np.zeros(self.n)
        self.resting_z = self.initial[: self.n, 2].copy()
        self.events = []
        self.grasp_events = []
        self.trace = []
        self.metrics = {}
        self.order_violation = False
        self.ever_released = set()
        self.intervention_applied = False

    def contained(self, i, b, obs):
        basket = self.scenario["baskets"][b]
        p = self.scenario["goal"]
        lo = obs["object_aabb_min"][i]
        hi = obs["object_aabb_max"][i]
        angle = basket["yaw_rad"]
        c, t = np.cos(angle), np.sin(angle)
        corners = np.array(list(product((lo[0], hi[0]), (lo[1], hi[1]))))
        local = (corners - basket["center_xy_m"]) @ np.array([[c, -t], [t, c]])
        half = np.array(basket["inner_size_xy_m"]) / 2 - p["wall_margin_m"]
        floor = self.scenario["reference_frame"]["origin_m"][2] + basket["bottom_m"]
        return bool(np.all(np.abs(local) <= half) and lo[2] >= floor - p["support_tolerance_m"])

    def supported(self, i, obs, basket=None):
        p = self.scenario["goal"]
        lo = obs["object_aabb_min"][i]
        hi = obs["object_aabb_max"][i]
        floor = self.scenario["reference_frame"]["origin_m"][2]
        if basket is not None:
            floor += self.scenario["baskets"][basket]["bottom_m"]
        if abs(lo[2] - floor) <= p["support_tolerance_m"]:
            return True
        for j in range(self.n):
            if i == j or basket is not None and not self.contained(j, basket, obs):
                continue
            a, b = obs["object_aabb_min"][j], obs["object_aabb_max"][j]
            if (
                abs(lo[2] - b[2]) <= p["support_tolerance_m"]
                and np.all(lo[:2] >= a[:2] - 0.003)
                and np.all(hi[:2] <= b[:2] + 0.003)
            ):
                return True
        return False

    def structure(self, indices, mode, target, obs):
        p = self.scenario["goal"]
        axis = 2 if mode == "tower" else 0
        ids = sorted(indices, key=lambda i: obs["object_aabb_min"][i, axis])
        lo = obs["object_aabb_min"]
        hi = obs["object_aabb_max"]
        poses = obs["object_pose_xyzw"]
        floor = self.scenario["reference_frame"]["origin_m"][2]
        ok = abs(lo[ids[0], 2] - floor) <= p["support_tolerance_m"]
        for previous, current in zip(ids, ids[1:]):
            ok &= abs(lo[current, axis] - hi[previous, axis]) <= p["join_tolerance_m"]
            if mode == "tower":
                ok &= bool(
                    np.all(lo[current, :2] >= lo[previous, :2] - 0.003)
                    and np.all(hi[current, :2] <= hi[previous, :2] + 0.003)
                )
            else:
                ok &= abs(poses[current, 1] - poses[previous, 1]) <= p["alignment_tolerance_m"]
                ok &= abs(lo[current, 2] - floor) <= p["support_tolerance_m"]
        dimension = float(hi[ids[-1], axis] - lo[ids[0], axis])
        return bool(ok and abs(dimension - target / 100) <= p["dimension_tolerance_m"]), dimension

    def update(self, obs, fingers, phase, dt):
        self.time += dt
        s = self.scenario
        p = s["goal"]
        plan = s["plan"]
        poses = obs["object_pose_xyzw"]
        lo = obs["object_aabb_min"]
        hi = obs["object_aabb_max"]
        hand = obs["hand_pose_xyzw"][:3]
        vel = obs["object_velocity"]
        self.peak_lifts = np.maximum(self.peak_lifts, poses[:, 2] - self.initial[:, 2])
        self.max_displacement = np.maximum(
            self.max_displacement, np.linalg.norm(poses[:, :3] - self.initial[:, :3], axis=1)
        )
        closed = np.sum(fingers) < 0.075
        stable = (np.linalg.norm(vel[: self.n, :3], axis=1) < p["linear_speed_m_s"]) & (
            np.linalg.norm(vel[: self.n, 3:], axis=1) < p["angular_speed_rad_s"]
        )
        upright = 1 - 2 * (poses[: self.n, 3] ** 2 + poses[: self.n, 4] ** 2) > np.cos(np.deg2rad(10))
        released = np.array(
            [
                not closed and np.linalg.norm(hand - np.clip(hand, lo[i], hi[i])) > p["gripper_enclosing_radius_m"]
                for i in range(self.n)
            ]
        )
        bins = [next((b for b in range(len(s["baskets"])) if self.contained(i, b, obs)), None) for i in range(self.n)]
        support = np.array([self.supported(i, obs, bins[i]) for i in range(self.n)])
        for i in range(self.n):
            held = (
                closed
                and np.linalg.norm(hand - poses[i, :3]) < 0.18
                and poses[i, 2] - self.resting_z[i] >= p["lift_height_m"]
            )
            if held and not self.armed[i]:
                self.armed[i] = True
                self.grasp_events.append(
                    dict(object_id=s["objects"][i]["object_id"], timestamp_s=self.time, pose_xyzw=poses[i].tolist())
                )
                order = plan["order"]
                if order and (
                    len(self.events) >= len(order) or s["objects"][i]["object_id"] != order[len(self.events)]
                ):
                    self.order_violation = True
            valid = self.armed[i] and released[i] and stable[i] and upright[i] and support[i]
            if plan["mode"] == "basket":
                valid &= bins[i] is not None
            self.placement_dwell[i] = self.placement_dwell[i] + dt if valid else 0.0
            if self.placement_dwell[i] >= p["placement_dwell_s"]:
                self.events.append(
                    dict(
                        object_id=s["objects"][i]["object_id"],
                        basket_id=None if bins[i] is None else f"basket_{bins[i]}",
                        placement_timestamp=self.time,
                        released=True,
                        contained=bins[i] is not None,
                        stable=True,
                        pose_xyzw=poses[i].tolist(),
                        measurement_values=s["objects"][i]["physical_properties_si"],
                    )
                )
                self.armed[i] = False
                self.placement_dwell[i] = 0.0
                self.ever_released.add(i)
                self.resting_z[i] = poses[i, 2]
        placed = [
            i
            for i in range(self.n)
            if i in self.ever_released and released[i] and stable[i] and upright[i] and support[i]
        ]
        case = s["case"]
        numerical = False
        used = []
        errors = []
        dimensions = []
        if plan["mode"] == "basket":
            inside = [i for i in placed if bins[i] is not None]
            if case.startswith("SAP"):
                expected = {
                    i: b
                    for i in range(self.n)
                    if (b := plan["assignments"].get(s["objects"][i]["object_id"])) is not None
                }
                numerical = all(i in inside and bins[i] == b for i, b in expected.items()) and set(inside) == set(
                    expected
                )
                used = list(expected)
            else:
                value = sum(
                    s["objects"][i]["physical_properties_si"][s["numerical_goal"]["measurement"]] for i in inside
                )
                factor = 1e6 if s["numerical_goal"]["measurement"] == "volume" else 100
                error = value * factor - s["numerical_goal"]["target"]
                errors = [float(error)]
                numerical = abs(error) <= 1e-5
                used = inside
        elif plan["mode"] == "chain":
            used = plan["solutions"][0][0]
            for ordinal, i in enumerate(used):
                target, _ = destination(s, i, 0, ordinal, obs, [])
                if ordinal == 0 and self.intervention_applied:
                    target[:2] = (np.asarray(target[:2]) + s["diagnostic_intervention"]["offset_xy_m"]).tolist()
                errors.append(float(np.linalg.norm(poses[i, :3] - target)))
            numerical = all(e <= p["position_tolerance_m"] for e in errors) and set(used).issubset(placed)
        else:
            targets = s["numerical_goal"].get("targets", [s["numerical_goal"].get("target")])
            for solution in plan["solutions"]:
                indices = sum(solution, [])
                if not set(indices).issubset(placed):
                    continue
                checks = [self.structure(group, plan["mode"], target, obs) for group, target in zip(solution, targets)]
                if len(solution) > 1:
                    ordered = np.mean(poses[solution[0], 0]) < np.mean(poses[solution[1], 0])
                else:
                    ordered = True
                if all(c[0] for c in checks) and ordered:
                    numerical = True
                    used = indices
                    dimensions = [c[1] for c in checks]
                    break
        unused = [i for i in range(self.n) if i not in used]
        preserved = all(
            self.max_displacement[i] <= p["max_distractor_movement_m"] and self.peak_lifts[i] < p["lift_height_m"]
            for i in unused
        )
        history = [e["object_id"] for e in self.events]
        order_ok = not self.order_violation and (not plan["order"] or history == plan["order"])
        no_extra = len(history) == len(used) and set(history) == {s["objects"][i]["object_id"] for i in used}
        stopped = phase == "stop"
        eligible = numerical and preserved and order_ok and no_extra and stopped
        self.dwell = self.dwell + dt if eligible else 0.0
        success = bool(self.dwell >= p["stable_hold_s"])
        self.metrics = dict(
            success=success,
            numerical_success=bool(numerical),
            sequence_success=bool(order_ok),
            stop_requested=stopped,
            correct_stopping=bool(stopped and numerical and no_extra),
            stable_seconds=self.dwell,
            placement_count=len(self.events),
            placement_events=deepcopy(self.events),
            grasp_events=deepcopy(self.grasp_events),
            final_basket_assignments={s["objects"][i]["object_id"]: bins[i] for i in placed},
            numerical_errors=errors,
            structure_dimensions_m=dimensions,
            distractors_preserved=bool(preserved),
            used_object_ids=[s["objects"][i]["object_id"] for i in used],
            order_violation=self.order_violation,
            collision_measurement="geometric containment/support only; contact impulses not recorded",
        )
        self.trace.append(
            dict(
                timestamp_s=self.time,
                **{
                    key: np.asarray(obs[key]).copy()
                    for key in (
                        "object_pose_xyzw",
                        "object_velocity",
                        "object_aabb_min",
                        "object_aabb_max",
                        "hand_pose_xyzw",
                    )
                },
                finger_positions=np.asarray(fingers).copy(),
                stop_requested=stopped,
                success=success,
            )
        )
        if "diagnostic_intervention" in s:
            self.trace[-1]["intervention_applied"] = self.intervention_applied
            self.metrics["diagnostic_intervention"] = dict(
                applied=self.intervention_applied, **s["diagnostic_intervention"]
            )
        return success
