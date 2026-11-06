# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feedback-Driven Planning: stop on an accumulated target (FDP-1) or chain on achieved poses (FDP-2)."""

import numpy as np

from tasks.constructive_planning.planning import construction, construction_text

MEASUREMENTS = {
    "FDP-1A": {"length"},
    "FDP-1B": {"volume"},
    "FDP-1C": {"height"},
}


def goal_field(case):
    """Return the numerical-goal field that accompanies ``measurement``."""
    return "offsets_cm" if case.startswith("FDP-2") else "target"


def solve(case, goal, candidates, ids, values):
    """Return the placement chain, or every combination that reaches the stopping target."""
    if case.startswith("FDP-1"):
        return construction(case, goal, values, goal.get("measurement", "height"))
    if len(goal["offsets_cm"]) > len(ids) or len(goal["offsets_cm"]) < 2:
        raise ValueError("Feedback chain requires two or more existing objects")
    if (
        case in ("FDP-2A", "FDP-2B")
        and len(goal["offsets_cm"]) != 2
        or case == "FDP-2C"
        and len(goal["offsets_cm"]) < 3
    ):
        raise ValueError("FDP-2A/B have two placements; FDP-2C requires at least three")
    offsets = np.asarray(goal["offsets_cm"], dtype=float)
    if (
        offsets.shape != (len(offsets), 2)
        or not np.isfinite(offsets).all()
        or np.any(np.count_nonzero(offsets, axis=1) != 1)
    ):
        raise ValueError("Each chain step must name exactly one finite, nonzero table-axis offset")
    heights = [o["size_m"][2] for o in candidates[: len(offsets)]]
    if len(set(heights)) != len(heights):
        raise ValueError("Chain block labels must be identifiable by their distinct heights")
    n = len(goal["offsets_cm"])
    return dict(mode="chain", solutions=[[list(range(n))]], assignments={}, order=ids[:n])


def describe(s):
    """Describe the stopping condition or the chained placements."""
    g, case = s["numerical_goal"], s["case"]
    attr = g.get("measurement", "height")
    unit = "cubic centimeters" if attr == "volume" else "cm"
    candidates = s["objects"][: s["candidate_count"]]
    if case.startswith("FDP-2"):
        clauses = []
        for i, delta in enumerate(g["offsets_cm"]):
            axis = int(np.flatnonzero(delta)[0])
            value = delta[axis]
            direction = ("right" if value > 0 else "left") if axis == 0 else ("forward" if value > 0 else "backward")
            previous = "the reference" if i == 0 else f"block {chr(64 + i)} at its actually achieved position"
            label = f"block {chr(65 + i)} ({candidates[i]['size_m'][2] * 100:g} cm tall)"
            relation = (
                f"with a {abs(value):g} cm surface gap {direction} of"
                if case == "FDP-2B"
                else f"with its center {abs(value):g} cm {direction} of the center of"
            )
            clauses.append(f"place {label} {relation} {previous}")
        return (
            "Then ".join(c[0].upper() + c[1:] + ". " for c in clauses)
            + "Release each block stably before the next placement."
        )
    if case in ("FDP-1A", "FDP-1B"):
        return (
            f"Place objects into the basket until their combined {attr} is exactly {g['target']:g} {unit}, "
            "then stop. Do not add extra objects."
        )
    return construction_text(s)
