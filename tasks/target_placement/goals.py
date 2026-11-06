# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target Placement goals: coordinate placement (TP-1) and clearance placement (TP-2)."""

import math

import numpy as np
from assets.geometry import world_aabb

from tasks.control.goals import angle_error, steps, vector, yaw


def resolve(goal):
    """Set the placement endpoint; clearance placements also fix a zero target yaw."""
    case, g, refs, target = goal.case, goal.parameters, goal.refs, goal.target
    if case == "TP-1A":
        target[:2] = vector(g["position_cm"], 2, "position_cm") / 100
    elif case == "TP-1B":
        target[:2] = np.asarray(refs["A"]["position_m"][:2]) + vector(g["offset_cm"], 2, "offset_cm") / 100
    elif case == "TP-1C":
        offset = vector(g["offset_cm"], 2, "offset_cm") / 100
        target[:2] = [refs["A"]["position_m"][0] + offset[0], refs["B"]["position_m"][1] + offset[1]]
    if case.startswith("TP-2"):
        goal.target_yaw = 0.0
        goal.target, goal.constraints = clearance_goal(case, g, goal.objects, goal.protocol, target)


def action_text(case, g, *, table_relative=False):
    """Describe the placement relative to the tabletop centre or the references."""
    if case == "TP-1A":
        x, y = g["position_cm"]
        if table_relative:
            offset = steps([x, y]) or "at the center of the tabletop"
            anchor = " from the center of the tabletop" if x or y else ""
            return f"Place the orange block's center {offset}{anchor} and release it on the table."
        return f"Place the orange block’s center at table X = {x:g} cm, Y = {y:g} cm and release it on the table."
    if case == "TP-1B":
        return (
            f"Place the orange block’s center {steps(g['offset_cm'])} from the red reference’s center "
            "and release it on the table."
        )
    if case == "TP-1C":
        x, y = g["offset_cm"]
        if table_relative:
            return (
                f"Place the orange block's center {abs(x):g} cm {'right' if x >= 0 else 'left'} "
                f"of the red reference's center and {abs(y):g} cm {'forward' if y >= 0 else 'backward'} "
                "of the blue reference's center. Release it on the table."
            )
        return (
            f"Place the orange block with its center X {abs(x):g} cm {'right' if x >= 0 else 'left'} "
            f"of the red reference’s center and center Y {abs(y):g} cm {'forward' if y >= 0 else 'backward'} "
            "of the blue reference’s center. Release it on the table."
        )
    if case.startswith("TP-2"):
        if case == "TP-2C":
            a, b = g["gaps_cm"]
            if table_relative:
                return (
                    f"Place the orange block between the red block on its left and the blue block on its right, "
                    f"leaving surface gaps of {a:g} cm to the red block and {b:g} cm to the blue block. "
                    "Align their centers in a left-to-right row parallel to the table's long edges, "
                    "and release the block with its faces parallel to the table edges."
                )
            return (
                f"Place the orange block between the red and blue blocks with a {a:g} cm gap to the left red face "
                f"and a {b:g} cm gap to the right blue face. Align their centers along table X "
                "and release it with faces parallel to the table axes."
            )
        direction = (
            ("right" if g["sign"] > 0 else "left") if g["axis"] == "x" else ("forward" if g["sign"] > 0 else "backward")
        )
        if table_relative:
            direction = (
                ("to the right" if g["sign"] > 0 else "to the left")
                if g["axis"] == "x"
                else ("on the far side" if g["sign"] > 0 else "on the robot side")
            )
            alignment = "long" if g["axis"] == "x" else "short"
            return (
                f"Place the orange block {direction} of the red reference, leaving a {g['gap_cm']:g} cm "
                f"surface-to-surface gap. Align their centers parallel to the table's {alignment} edges "
                "and release it with faces parallel to the table edges."
            )
        return (
            f"Place the orange block {direction} of the red reference, leaving a {g['gap_cm']:g} cm "
            f"surface-to-surface gap along table {g['axis'].upper()}. Align the centers on that axis "
            "and release it with faces parallel to the table axes."
        )
    raise ValueError(f"Not a Target Placement case: {case!r}")


def clearance_goal(case, g, objects, protocol, target):
    """Solve one or two explicit surface gaps, including geometry extents."""
    obj = objects[0]
    refs = {o["object_id"]: o for o in objects[1:]}
    constraints = {}
    if any(angle_error(yaw(o["rotation_xyzw"]), 0) > 1e-4 for o in objects):
        raise ValueError("Clearance baselines require axis-aligned initial and final objects")
    a = refs["A"]
    a_lo, a_hi = world_aabb(a["geometry"], a["position_m"], a["rotation_xyzw"])
    size = np.asarray(obj["size_m"])
    if case == "TP-2C":
        gaps = vector(g["gaps_cm"], 2, "gaps_cm") / 100
        if np.any(gaps <= protocol["clearance_tolerance_m"]):
            raise ValueError("Gaps must exceed clearance tolerance")
        b = refs["B"]
        b_lo, _ = world_aabb(b["geometry"], b["position_m"], b["rotation_xyzw"])
        target[:2] = [a_hi[0] + gaps[0] + size[0] / 2, a["position_m"][1]]
        if abs(b_lo[0] - (target[0] + size[0] / 2) - gaps[1]) > 1e-6 or abs(b["position_m"][1] - target[1]) > 1e-6:
            raise ValueError("Two clearance constraints are inconsistent")
        constraints["gaps_m"] = gaps.tolist()
    else:
        axis = {"x": 0, "y": 1}.get(g["axis"])
        sign = g["sign"]
        gap = float(g["gap_cm"]) / 100
        if axis is None or sign not in (-1, 1) or not math.isfinite(gap) or gap <= protocol["clearance_tolerance_m"]:
            raise ValueError("Clearance requires axis x/y, sign +/-1 and a gap above tolerance")
        if case == "TP-2B" and a["geometry"]["kind"] != "cylinder":
            raise ValueError("Cross-shape baseline uses a cuboid and upright cylinder reference")
        target[:2] = a["position_m"][:2]
        target[axis] = (a_hi[axis] if sign > 0 else a_lo[axis]) + sign * (gap + size[axis] / 2)
        constraints.update(axis=axis, sign=sign, gap_m=gap)
    return target, constraints
