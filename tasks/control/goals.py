# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared control goal solving in the table frame (SI units).

Each control subgroup package resolves its own cases in ``goals.py``.
"""

import math
from types import SimpleNamespace

import numpy as np
from assets.geometry import describe
from environments import settings

from tasks.registry import CORE_TASKS, SUBGROUPS, subgroup_module, subgroup_of_case

# Variant prefix to core task, for the control subgroups only.
TASKS = {
    prefix: task
    for prefix, task in CORE_TASKS.items()
    if SUBGROUPS[subgroup_of_case(prefix + "A")]["engine"] == "control"
}
CASES = tuple(prefix + suffix for prefix in TASKS for suffix in "ABC")


def quaternion(yaw):
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def angle_error(actual, target, period=math.pi):
    """Undirected rectangular principal axes are equivalent modulo pi."""
    return abs((actual - target + period / 2) % period - period / 2)


def vector(values, length, label):
    v = np.asarray(values, dtype=float)
    if v.shape != (length,) or not np.isfinite(v).all():
        raise ValueError(f"{label} requires {length} finite numerical components")
    return v


def horizontal_distance(lo, hi, other_lo, other_hi):
    """Conservative complete-geometry distance: horizontal enclosing AABB separation.

    AABB separation never overstates true object separation. Unlike center distance,
    this bound includes both objects' dimensions and current orientation. It also
    prevents an over-the-obstacle shortcut in the around-obstacle baseline.
    """
    return float(np.linalg.norm(np.maximum(np.maximum(other_lo[:2] - hi[:2], lo[:2] - other_hi[:2]), 0)))


def steps(v):
    """Describe a table-frame displacement such as ``10 cm right and 8 cm forward``."""
    names = [("right", "left"), ("forward", "backward"), ("up", "down")]
    return " and ".join(f"{abs(x):g} cm {names[i][0 if x > 0 else 1]}" for i, x in enumerate(v) if x)


def solve(case, parameters, objects, protocol):
    """Resolve the endpoint, ordered mandatory waypoints, and oracle transport route.

    Uses measured settled poses at runtime, authored poses during generation.
    No commanded pose is evidence that the object actually visited a waypoint.
    The case's subgroup package fills in its endpoint, waypoints, and constraints.
    """
    obj = objects[0]
    initial = np.asarray(obj["position_m"], dtype=float)
    height = describe(obj["geometry"])["size_m"][2]
    start_yaw = yaw(obj["rotation_xyzw"])
    goal = SimpleNamespace(
        case=case,
        parameters=parameters,
        obj=obj,
        objects=objects,
        refs={o["object_id"]: o for o in objects[1:]},
        protocol=protocol,
        initial=initial,
        height=height,
        target=initial.copy(),
        start_yaw=start_yaw,
        target_yaw=start_yaw,
        transport_z=settings.TABLE_HEIGHT + height / 2 + protocol["transport_bottom_clearance_m"],
        waypoints=[],
        route=[],
        final_mode="release",
        constraints={},
    )
    subgroup_module(subgroup_of_case(case), "goals").resolve(goal)
    route = goal.route
    if not route:
        route = [[*goal.target[:2], goal.transport_z]]
    if goal.final_mode == "hold":
        route = [goal.target.tolist()]
    return dict(
        target_position_m=goal.target.tolist(),
        target_yaw_rad=goal.target_yaw,
        initial_yaw_rad=start_yaw,
        final_mode=goal.final_mode,
        waypoints_m=[[float(v) for v in point] for point in goal.waypoints],
        oracle_route_m=[[float(v) for v in point] for point in route],
        transport_z_m=float(route[0][2]),
        constraints=goal.constraints,
    )


def action_text(case, g, *, table_relative=False):
    """Render numerical meanings once; paraphrase wrappers cannot change parameters."""
    return subgroup_module(subgroup_of_case(case), "goals").action_text(case, g, table_relative=table_relative)
