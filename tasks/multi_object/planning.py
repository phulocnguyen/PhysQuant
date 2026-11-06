# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Enumerate admissible numerical solutions independently of oracle execution.

Each multi-object subgroup package solves its own cases in ``planning.py``.
"""

from itertools import combinations

import numpy as np

from tasks.registry import CORE_TASKS, SUBGROUPS, subgroup_module, subgroup_of_case

# Variant prefix to core task, for the multi-object subgroups only.
TASKS = {
    prefix: task
    for prefix, task in CORE_TASKS.items()
    if SUBGROUPS[subgroup_of_case(prefix + "A")]["engine"] == "multi_object"
}
CASES = tuple(f"{task}{variant}" for task in TASKS for variant in "ABC")


def subsets(values, target, minimum=1):
    return [
        list(indices)
        for n in range(minimum, len(values) + 1)
        for indices in combinations(range(len(values)), n)
        if np.isclose(sum(values[i] for i in indices), target, atol=1e-8, rtol=0)
    ]


def solve(case, objects, goal):
    """Return every valid component allocation; only ordering tasks fix the order.

    The case's subgroup package declares its goal field and allowed measurements,
    then solves the allocation.
    """
    subgroup = subgroup_module(subgroup_of_case(case), "planning")
    if set(goal) != {"measurement", subgroup.goal_field(case)}:
        raise ValueError("Missing or unknown numerical-goal fields")
    candidates = [o for o in objects if o.get("selectable", True)]
    ids = [o["object_id"] for o in candidates]
    attr = goal.get("measurement", "height")
    if attr not in ("length", "width", "height", "volume"):
        raise ValueError("Unsupported Multi-object measurement")
    if attr not in subgroup.MEASUREMENTS.get(case, {"length", "width", "height"}):
        raise ValueError("Measurement does not match the declared task variant")
    scale = 1e6 if attr == "volume" else 100
    values = [o["physical_properties_si"][attr] * scale for o in candidates]
    return subgroup.solve(case, goal, candidates, ids, values)


def strict_compare(value, boundary, operator):
    """Exclude floating-point equality from strict physical threshold operators."""
    return bool(
        not np.isclose(value, boundary, rtol=0, atol=1e-8)
        and (value > boundary if operator == "gt" else value < boundary)
    )


def destination(scene, object_index, group, ordinal, observation, placed):
    """Oracle goals use current achieved support/reference poses, never ideal predecessors."""
    objects = scene["objects"]
    obj = objects[object_index]
    goal = scene["numerical_goal"]
    plan = scene["plan"]
    mode = plan["mode"]
    table = scene["reference_frame"]["origin_m"][2]
    size = obj["size_m"]
    if mode == "basket":
        basket = scene["baskets"][group]
        angle = basket["yaw_rad"]
        local = np.asarray(basket["slots_xy_m"][ordinal])
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return [*(np.asarray(basket["center_xy_m"]) + rot @ local), table + basket["bottom_m"] + size[2] / 2], angle
    if mode == "chain":
        predecessor = scene["reference_index"] if ordinal == 0 else ordinal - 1
        base = observation["object_pose_xyzw"][predecessor, :3].copy()
        delta = np.asarray(goal["offsets_cm"][ordinal], dtype=float) / 100
        if scene["case"] == "FDP-2B":
            axis = int(np.flatnonzero(delta)[0])
            sign = np.sign(delta[axis])
            achieved_extent = (
                observation["object_aabb_max"][predecessor, axis] - observation["object_aabb_min"][predecessor, axis]
            )
            delta[axis] += sign * (achieved_extent + size[axis]) / 2
        return [*(base[:2] + delta), table + size[2] / 2], 0.0
    anchor = scene["construction_xy_m"][group]
    if not placed:
        return [*anchor, table + size[2] / 2], 0.0
    previous = placed[-1]
    pos = observation["object_pose_xyzw"][previous, :3]
    if mode == "tower":
        return [*pos[:2], observation["object_aabb_max"][previous, 2] + size[2] / 2], 0.0
    return [observation["object_aabb_max"][previous, 0] + size[0] / 2, pos[1], table + size[2] / 2], 0.0
