# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spatial Transformation goals: displacement (ST-1) and rotation (ST-2)."""

import math

import numpy as np

from tasks.control.goals import steps, vector, yaw


def resolve(goal):
    """Set the endpoint and final mode of a translation, or the target yaw of a rotation."""
    case, g, obj, refs = goal.case, goal.parameters, goal.obj, goal.refs
    if case.startswith("ST-1"):
        goal.target, goal.final_mode = translation_goal(case, g, goal.initial, goal.protocol)
    if case.startswith("ST-2"):
        angle = math.radians(float(g["angle_deg"]))
        if not math.isfinite(angle) or not 20 <= abs(g["angle_deg"]) < 180:
            raise ValueError("Rotation must be nonzero, distinguishable and below 180 degrees")
        goal.target_yaw = (yaw(refs["A"]["rotation_xyzw"]) if case == "ST-2C" else goal.start_yaw) + angle
        if obj["geometry"]["kind"] != "box" or abs(obj["size_m"][0] - obj["size_m"][1]) < 0.025:
            raise ValueError("Rotation requires a visibly elongated rectangular prism")


def action_text(case, g, *, table_relative=False):
    """Describe the displacement or rotation; ``table_relative`` does not change these cases."""
    if case.startswith("ST-1"):
        return f"Pick up the orange block, move it {steps(g['displacement_cm'])} from its starting center, " + (
            "and hold it there." if case == "ST-1C" else "and release it stably on the table."
        )
    if case.startswith("ST-2"):
        ref = "the red reference’s long axis" if case == "ST-2C" else "its starting orientation"
        direction = "counterclockwise" if g["angle_deg"] > 0 else "clockwise"
        return (
            f"Pick up the orange rectangular block, rotate it {abs(g['angle_deg']):g} degrees "
            f"{direction} from {ref}, and release it at its starting center. "
            "Angles are viewed from above; long axes are equivalent modulo 180 degrees."
        )
    raise ValueError(f"Not a Spatial Transformation case: {case!r}")


def translation_goal(case, g, initial, protocol):
    """Displacements preserve table support unless the protocol requires a hold."""
    target = initial.copy()
    final_mode = "release"
    delta = vector(g["displacement_cm"], 3, "displacement_cm") / 100
    if case == "ST-1A" and (np.count_nonzero(delta[:2]) != 1 or delta[2] != 0):
        raise ValueError("ST-1A requires one horizontal axis")
    if case == "ST-1B" and (np.count_nonzero(delta[:2]) != 2 or delta[2] != 0):
        raise ValueError("ST-1B requires both horizontal axes")
    if case == "ST-1C":
        if np.any(delta[:2]) or delta[2] < 0.06:
            raise ValueError("ST-1C requires a positive vertical displacement of at least 6 cm")
        final_mode = "hold"
    elif delta[2] != 0:
        raise ValueError("Released goals must rest on the table; use ST-1C for vertical hold")
    if np.linalg.norm(delta) < 2 * protocol["position_tolerance_m"]:
        raise ValueError("Displacement must exceed twice the physical position tolerance")
    target += delta
    return target, final_mode
