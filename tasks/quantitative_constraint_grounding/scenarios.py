# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-answer filters, conjunctions, arithmetic, and nearest-match tasks."""

import math

import numpy as np
from assets.geometry import describe, make
from environments import settings

from tasks.common.scenarios import MINIMUM_PICK_RADIUS_M, InfeasibleScenario
from tasks.relational_understanding.scenarios import (
    base_scene,
    constant,
    finish,
    make_reference,
    placement_valid,
    set_geometry,
)

CASES = (
    "threshold",
    "interval",
    "tolerance",
    "multi_attribute",
    "object_relation",
    "multi_reference",
    "arithmetic",
    "ratio",
    "derived_property",
    "nearest",
)


def _interval(cfg):
    bounds = cfg.interval_bounds if cfg.interval_bounds is not None else [cfg.value - 0.5, cfg.value + 0.5]
    if (
        len(bounds) != 2
        or bounds[0] >= bounds[1]
        or len(cfg.interval_closed) != 2
        or any(type(closed) is not bool for closed in cfg.interval_closed)
    ):
        raise ValueError("Intervals require increasing bounds and two boolean endpoint flags")
    return dict(
        op="interval",
        bounds=bounds,
        bounds_si=[constant(value, cfg.unit)["value_si"] for value in bounds],
        closed=cfg.interval_closed,
        unit=cfg.unit,
    )


def _validate_config(cfg):
    if cfg.case not in CASES or cfg.unit != ("cm3" if cfg.case == "derived_property" else "cm"):
        raise ValueError("Unsupported reasoning case or unit")
    if cfg.operator not in ("gt", "ge", "lt", "le"):
        raise ValueError("Threshold operator must be gt, ge, lt, or le")
    if not math.isfinite(cfg.gap) or cfg.gap <= 0:
        raise ValueError("Distractor gap must be finite and positive")


def generate(cfg):
    _validate_config(cfg)
    scene = base_scene(cfg)
    objects = scene["objects"]
    refs = []
    measure = {"kind": "attribute", "attribute": "volume" if cfg.case == "derived_property" else "height"}
    target = constant(cfg.value, cfg.unit)
    gap = constant(cfg.gap, cfg.unit)["value_si"]
    predicate = {"op": "eq", "measure": measure, "target": target}
    if cfg.case in ("arithmetic", "ratio", "derived_property"):
        reference_height = target["value_si"] / 2
        if cfg.case == "derived_property":
            reference_height /= 0.08 * 0.04
        if cfg.case == "arithmetic":
            refs = [
                make_reference(scene, "red", [-0.40, -0.08], target["value_si"] * 0.4),
                make_reference(scene, "blue", [-0.40, 0.20], target["value_si"] * 0.6),
            ]
            expression = {
                "kind": "sum",
                "terms": [
                    {"kind": "reference_attribute", "reference": r["object_id"], "attribute": "height"} for r in refs
                ],
            }
        else:
            refs = [make_reference(scene, "red", [-0.40, -0.08], reference_height)]
            expression = {
                "kind": "scale",
                "factor": 2,
                "terms": [
                    {"kind": "reference_attribute", "reference": "reference_red", "attribute": measure["attribute"]}
                ],
            }
        predicate["target"] = expression
    elif cfg.case == "threshold":
        predicate["op"] = cfg.operator
    elif cfg.case == "interval":
        predicate.update(_interval(cfg))
    elif cfg.case == "tolerance":
        if cfg.tolerance <= 0:
            raise ValueError("Tolerance must be positive and independent of the distractor gap")
        predicate.update(
            op="tolerance",
            tolerance=cfg.tolerance,
            tolerance_si=constant(cfg.tolerance, cfg.unit)["value_si"],
            unit=cfg.unit,
        )
    elif cfg.case == "nearest":
        predicate["op"] = "closest"
    elif cfg.case == "multi_attribute":
        predicate = {
            "op": "and",
            "terms": [
                predicate,
                {"op": "eq", "measure": {"kind": "attribute", "attribute": "width"}, "target": constant(4, "cm")},
            ],
        }
    elif cfg.case == "object_relation":
        refs = [make_reference(scene, "red", [-0.24, 0.02])]
        predicate = {
            "op": "and",
            "terms": [
                predicate,
                {
                    "op": "eq",
                    "measure": {"kind": "distance", "reference": "reference_red", "planar": True},
                    "target": constant(20, "cm"),
                },
            ],
        }
    elif cfg.case == "multi_reference":
        refs = [make_reference(scene, "red", [-0.25, 0]), make_reference(scene, "blue", [0.25, 0])]
        predicate = {
            "op": "and",
            "terms": [
                {
                    "op": "eq",
                    "measure": {"kind": "offset", "reference": "reference_red", "axis": 0, "frame": "table"},
                    "target": constant(20, "cm"),
                },
                {
                    "op": "eq",
                    "measure": {"kind": "offset", "reference": "reference_blue", "axis": 1, "frame": "table"},
                    "target": constant(15, "cm"),
                },
            ],
        }
    for index, obj in enumerate(objects):
        value = (
            target["value_si"]
            if index == 0
            else target["value_si"] + (-1 if index % 2 else 1) * math.ceil(index / 2) * gap
        )
        if cfg.case == "threshold":
            direction = 1 if cfg.operator in ("gt", "ge") else -1
            value = target["value_si"] + direction * gap if index == 0 else target["value_si"] - direction * gap * index
        if cfg.case == "nearest" and index == 0:
            value += 0.2 * gap
        if cfg.case in ("multi_attribute", "object_relation") and index == 1:
            value = target["value_si"]
        if cfg.case == "multi_reference":
            value = 0.06
        if value <= 0:
            raise InfeasibleScenario("Requested reasoning count/gap produces a nonpositive dimension")
        geometry = make("box", measure["attribute"], value)
        if cfg.case != "derived_property":
            geometry["size_m"][:2] = [0.06, 0.06 if cfg.case == "multi_attribute" and index == 1 else 0.04]
        set_geometry(obj, geometry)
        obj["position_m"][2] = settings.TABLE_HEIGHT + obj["size_m"][2] / 2
    for obj in objects + refs:
        if not describe(obj["geometry"])["grasp_feasible"]:
            raise InfeasibleScenario("Reasoning geometry exceeds the supported size/grasp envelope")
    scene["objects"] = objects + refs
    rng = np.random.default_rng(cfg.seed)
    for _ in range(100):
        placed = list(refs)
        for index, obj in enumerate(objects):
            for _ in range(1000):
                xy = rng.uniform([-0.32, -0.10], [0.32, 0.32])
                if cfg.case == "object_relation":
                    angle = rng.uniform(-math.pi, math.pi)
                    distance = 0.20 if index in (0, 2) else 0.26
                    xy = np.asarray(refs[0]["position_m"][:2]) + distance * np.asarray(
                        [math.cos(angle), math.sin(angle)]
                    )
                elif cfg.case == "multi_reference":
                    if index == 0:
                        xy = [-0.05, 0.15]
                    elif index % 2:
                        xy[0] = -0.05
                    else:
                        xy[1] = 0.15
                obj["position_m"][:2] = list(map(float, xy))
                if placement_valid(obj, placed, minimum_pick_radius_m=MINIMUM_PICK_RADIUS_M):
                    placed.append(obj)
                    break
            else:
                break
        else:
            return finish(scene, cfg, predicate, "quantitative_constraint_grounding")
    raise InfeasibleScenario("No feasible reasoning layout for these dimensions and references")
