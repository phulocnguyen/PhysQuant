# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sequencing and Allocation Planning: ordering (SAP-1) and numerical sorting (SAP-2)."""

from itertools import product

import numpy as np

from tasks.multi_object.planning import strict_compare

# Allowed measurements per case; unlisted cases accept length, width, or height.
MEASUREMENTS = {}


def goal_field(case):
    """Return the numerical-goal field that accompanies ``measurement``."""
    if case == "SAP-1B":
        return "direction"
    if case.startswith("SAP-1"):
        return "sequence"
    return "conditions" if case == "SAP-2C" else "boundaries"


def solve(case, goal, candidates, ids, values):
    """Return the basket order or bin assignment that the instruction fixes."""
    if case.startswith("SAP-1"):
        requested = goal.get("sequence")
        if case == "SAP-1B":
            if goal["direction"] not in ("ascending", "descending"):
                raise ValueError("Ordering direction must be ascending or descending")
            requested = sorted(values, reverse=goal["direction"] == "descending")
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Ordering requires a nonempty unique numerical sequence")
        if case == "SAP-1A" and len(requested) != len(ids) or case == "SAP-1C" and len(requested) >= len(ids):
            raise ValueError("SAP-1A orders all candidates; SAP-1C must leave a distractor")
        order = []
        for value in requested:
            matches = [i for i, v in enumerate(values) if np.isclose(v, value, atol=1e-8, rtol=0)]
            if len(matches) != 1:
                raise ValueError("Each ordering measurement must identify one object")
            order.append(matches[0])
        return dict(
            mode="basket", solutions=[[order]], assignments={ids[i]: 0 for i in order}, order=[ids[i] for i in order]
        )
    if case.startswith("SAP-2"):
        if case == "SAP-2C":
            terms = goal["conditions"]
            if any(
                set(t) != {"operator", "attribute", "value_cm"}
                or t["operator"] not in ("gt", "lt")
                or t["attribute"] not in ("length", "width", "height")
                or not np.isfinite(t["value_cm"])
                or t["value_cm"] <= 0
                for t in terms
            ):
                raise ValueError("Sorting conjunction requires explicit linear comparison operators")
            truth = [
                tuple(
                    strict_compare(o["physical_properties_si"][t["attribute"]] * 100, t["value_cm"], t["operator"])
                    for t in terms
                )
                for o in candidates
            ]
            if len(terms) != 2 or set(truth) != set(product((False, True), repeat=2)):
                raise ValueError("Conjunction sorting requires all four truth combinations")
            bins = [0 if all(t) else 1 for t in truth]
        else:
            cuts = goal["boundaries"]
            if (
                not np.isfinite(cuts).all()
                or min(cuts, default=0) <= 0
                or cuts != sorted(set(cuts))
                or len(cuts) != (1 if case == "SAP-2A" else 2)
            ):
                raise ValueError("Sorting boundaries must be strictly increasing")
            bins = [sum(not strict_compare(value, cut, "lt") for cut in cuts) for value in values]
        count = 3 if case == "SAP-2B" else 2
        if set(bins) != set(range(count)):
            raise ValueError("Every sorting bin must have an example")
        return dict(mode="basket", solutions=[[list(range(len(ids)))]], assignments=dict(zip(ids, bins)), order=[])
    raise ValueError(f"Not a Sequencing and Allocation Planning case: {case!r}")


def describe(s):
    """Describe the ordering or sorting rule."""
    g, case, plan = s["numerical_goal"], s["case"], s["plan"]
    attr = g.get("measurement", "height")

    def numbers(xs):
        return ", ".join(f"{v:g}" for v in xs)

    candidates = s["objects"][: s["candidate_count"]]
    if case == "SAP-1B" and s["template_version"] >= 2:
        return f"Place all objects into the basket in {g['direction']} order of {attr}."
    if case.startswith("SAP-1"):
        vals = [
            candidates[next(i for i, o in enumerate(candidates) if o["object_id"] == key)]["physical_properties_si"][
                attr
            ]
            * 100
            for key in plan["order"]
        ]
        return (
            f"Place the objects with {attr}s {numbers(vals)} cm into the basket in exactly that order. "
            "Leave all other objects untouched."
        )
    if case == "SAP-2C":
        clauses = [
            f"{t['attribute']} {'greater than' if t['operator'] == 'gt' else 'less than'} {t['value_cm']:g} cm"
            for t in g["conditions"]
        ]
        return (
            f"Place objects with {' and '.join(clauses)} in the left basket; "
            "place every other object in the right basket."
        )
    else:
        cuts = g["boundaries"]
        if len(cuts) == 1:
            return (
                f"Sort by {attr}: below {cuts[0]:g} cm into the left basket, "
                f"and at least {cuts[0]:g} cm into the right basket."
            )
        else:
            return (
                f"Sort by {attr}: below {cuts[0]:g} cm into the left basket, "
                f"from {cuts[0]:g} to below {cuts[1]:g} cm into the middle basket, "
                f"and at least {cuts[1]:g} cm into the right basket."
            )
