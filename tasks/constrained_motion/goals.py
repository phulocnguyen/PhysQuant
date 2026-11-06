# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constrained Motion goals: waypoint following (CM-1) and constrained transport (CM-2)."""

import math

import numpy as np
from assets.geometry import world_aabb
from environments import settings

from tasks.control.goals import steps, vector
from tasks.spatial_transformation.goals import translation_goal


def resolve(goal):
    """Set waypoints, the oracle route, and transport constraints."""
    case, g, obj, refs, protocol = goal.case, goal.parameters, goal.obj, goal.refs, goal.protocol
    initial, height, transport_z = goal.initial, goal.height, goal.transport_z
    target, waypoints, route, constraints = goal.target, goal.waypoints, goal.route, goal.constraints
    clearance = protocol["transport_bottom_clearance_m"]
    if case.startswith("CM-2"):
        target, goal.final_mode = translation_goal(case, g, initial, protocol)
    if case.startswith("CM-1"):
        if case == "CM-1C":
            offsets = g["offsets_cm"]
            if len(offsets) != len(refs) or not 1 <= len(offsets) <= 2:
                raise ValueError("Reference waypoints require one offset for each of one or two references")
            for ref, offset in zip(refs.values(), offsets):
                xy = np.asarray(ref["position_m"][:2]) + vector(offset, 2, "offsets_cm") / 100
                waypoints.append([*xy, transport_z])
        else:
            steps = g["steps_cm"]
            if not (len(steps) == 2 if case == "CM-1A" else 3 <= len(steps) <= 4):
                raise ValueError("Two-step requires two segments; multi-waypoint requires three or four")
            xy = initial[:2].copy()
            for step in steps:
                delta = vector(step, 2, "steps_cm") / 100
                if np.count_nonzero(delta) != 1 or np.linalg.norm(delta) <= 2 * protocol["waypoint_tolerance_m"]:
                    raise ValueError("Each step must have one axis and exceed twice the waypoint tolerance")
                xy += delta
                waypoints.append([*xy, transport_z])
        target[:2] = waypoints[-1][:2]
        if any(
            np.linalg.norm(np.asarray(a) - b) <= 2 * protocol["waypoint_tolerance_m"]
            for i, a in enumerate(waypoints)
            for b in waypoints[:i]
        ):
            raise ValueError("Waypoint tolerance regions must be disjoint")
        route = waypoints.copy()
    if case == "CM-2A":
        minimum = float(g["minimum_clearance_cm"]) / 100
        if not math.isfinite(minimum) or not 0.04 <= minimum <= 0.20:
            raise ValueError("Minimum transport clearance must be in [4,20] cm")
        transport_z = settings.TABLE_HEIGHT + height / 2 + max(clearance, minimum + 0.035)
        constraints["minimum_clearance_m"] = minimum
    elif case == "CM-2B":
        limit = math.radians(float(g["max_yaw_deviation_deg"]))
        if (
            not math.radians(3) <= limit <= math.radians(30)
            or obj["geometry"]["kind"] != "box"
            or abs(obj["size_m"][0] - obj["size_m"][1]) < 0.025
        ):
            raise ValueError("Yaw constraint requires elongated box and tolerance 3–30 degrees")
        constraints["max_yaw_deviation_rad"] = limit
    elif case == "CM-2C":
        minimum = float(g["minimum_clearance_cm"]) / 100
        if not math.isfinite(minimum) or not 0.03 <= minimum <= 0.12 or g["detour_side"] not in (-1, 1):
            raise ValueError("Obstacle clearance must be 3–12 cm and detour side +/-1")
        obstacle = refs["A"]
        lo, hi = world_aabb(obstacle["geometry"], obstacle["position_m"], obstacle["rotation_xyzw"])
        if not initial[0] < lo[0] < hi[0] < target[0]:
            raise ValueError("Obstacle baseline requires start left and goal right of obstacle")
        side = g["detour_side"]
        detour_y = (hi[1] if side > 0 else lo[1]) + side * (minimum + obj["size_m"][1] / 2 + 0.035)
        route = [
            [initial[0], detour_y, transport_z],
            [target[0], detour_y, transport_z],
            [target[0], target[1], transport_z],
        ]
        constraints["minimum_obstacle_clearance_m"] = minimum
    goal.target, goal.transport_z, goal.route = target, transport_z, route


def action_text(case, g, *, table_relative=False):
    """Describe the waypoints or the transport constraint; ``table_relative`` does not change these cases."""
    if case.startswith("CM-1"):
        if case == "CM-1C":
            phrase = ", then ".join(
                f"{steps(offset)} from the {color} reference’s center"
                for offset, color in zip(g["offsets_cm"], ("red", "blue"))
            )
            return (
                f"Lift the orange block to the stated transport height and carry its center first {phrase}, "
                "in order. Release it on the table below the last point."
            )
        return (
            "Lift the orange block to the stated transport height, then carry it "
            f"{', then '.join(steps(v) for v in g['steps_cm'])}, visiting each intermediate point in order. "
            "Each displacement starts at the preceding point. Release it on the table below the last point."
        )
    requirement = (
        f"its bottom at least {g['minimum_clearance_cm']:g} cm above the table"
        if case == "CM-2A"
        else f"its yaw within {g['max_yaw_deviation_deg']:g} degrees of its initial long-axis orientation"
        if case == "CM-2B"
        else f"its complete body at least {g['minimum_clearance_cm']:g} cm horizontally clear of the red obstacle"
    )
    return (
        f"Pick up the orange block and carry it {steps(g['displacement_cm'])} from its starting center "
        f"while keeping {requirement} throughout horizontal transport. Then lower and release it on the table."
    )
