# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constructive Planning: one structure (CP-1) or two related structures (CP-2)."""

from itertools import product

import numpy as np

from tasks.multi_object.planning import subsets

MEASUREMENTS = {
    "CP-1A": {"height"},
    "CP-1B": {"length"},
    "CP-1C": {"height", "length"},
    "CP-2A": {"height"},
    "CP-2B": {"height"},
    "CP-2C": {"length"},
}


def goal_field(case):
    """Return the numerical-goal field that accompanies ``measurement``."""
    return "targets" if case.startswith("CP-2") else "target"


def solve(case, goal, candidates, ids, values):
    """Return every disjoint component combination that reaches the targets."""
    return construction(case, goal, values, goal.get("measurement", "height"))


def construction(case, goal, values, attr):
    """Solve tower, row, or basket-collection targets; Feedback-Driven Planning reuses this for FDP-1."""
    targets = goal.get("targets", [goal.get("target")])
    if len(targets) != (2 if case.startswith("CP-2") else 1):
        raise ValueError("Wrong number of construction/collection targets")
    if any(v is None or not np.isfinite(v) or v <= 0 for v in targets):
        raise ValueError("Construction/collection targets must be positive")
    if case == "CP-2A" and targets[0] != targets[1] or case == "CP-2B" and targets[0] == targets[1]:
        raise ValueError("CP-2A requires equal anchors; CP-2B requires a nonzero difference")
    choices = [subsets(values, t, minimum=2) for t in targets]
    solutions = [list(groups) for groups in product(*choices) if len(set(sum(groups, []))) == len(sum(groups, []))]
    if not solutions or case == "CP-1C" and len(solutions) < 2:
        raise ValueError("No valid disjoint combination, or multiple solutions missing")
    mode = "basket" if case in ("FDP-1A", "FDP-1B") else "tower" if attr == "height" else "row"
    return dict(mode=mode, solutions=solutions, assignments={}, order=[])


def describe(s):
    """Describe the requested structures."""
    return construction_text(s)


def construction_text(s):
    """Describe tower or row targets; FDP-1C reuses this with its own wording."""
    g, case, plan = s["numerical_goal"], s["case"], s["plan"]
    attr = g.get("measurement", "height")
    targets = g.get("targets", [g.get("target")])
    structure = "tower" if plan["mode"] == "tower" else "straight row along table +X with touching end faces"
    if s["template_version"] >= 3 and plan["mode"] != "tower":
        structure = "straight left-to-right row parallel to the table's long edges with touching end faces"
    if case == "CP-2B" and s["template_version"] >= 2:
        delta = targets[1] - targets[0]
        relation = "taller" if delta > 0 else "shorter"
        action = (
            f"Build a left tower {targets[0]:g} cm tall and a right tower "
            f"{abs(delta):g} cm {relation} than the left tower, then stop."
        )
    elif case == "FDP-1C" and s["template_version"] >= 2:
        action = (
            f"Add blocks to one tower, checking its achieved height after each placement, "
            f"until it is {targets[0]:g} cm tall, then stop."
        )
    elif len(targets) == 1:
        action = f"Build one {structure} with {attr} {targets[0]:g} cm using the available objects, then stop."
    else:
        action = (
            f"Build a left {structure} with {attr} {targets[0]:g} cm "
            f"and a right one with {attr} {targets[1]:g} cm, then stop."
        )
    action += " Leave unused objects untouched. Any valid component combination is allowed."
    return action
