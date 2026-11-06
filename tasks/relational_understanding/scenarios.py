# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-first layouts for exact coordinates, distances, clearances, and angles."""

import math
from copy import deepcopy

import numpy as np
from assets.geometry import describe
from environments import settings

from tasks.common.scenarios import (
    MINIMUM_PICK_RADIUS_M,
    InfeasibleScenario,
    identify,
    placement_valid,
    seal,
    table_reference_frame,
    validate,
)
from tasks.object_understanding.scenarios import generate as generate_ou

CASES = (
    "absolute_position",
    "directional_offset",
    "center_distance",
    "surface_clearance",
    "boundary_distance",
    "orientation",
    "relative_orientation",
    "reference_frame",
)


def set_geometry(obj, geometry):
    info = describe(geometry)
    obj.update(
        geometry=geometry,
        physical_properties_si=info["measurements"],
        internal_geometry=info["interior"],
        components=info["components"],
        size_m=info["size_m"],
        shape=geometry["kind"],
        radius_m=info["measurements"]["radius"],
        grasp_width_m=info["grasp_width_m"],
        grasp_offset_local_m=info["grasp_offset_local_m"],
        grasp_yaw_offset_rad=info["grasp_yaw_offset_rad"],
    )


def base_scene(cfg):
    base_cfg = deepcopy(cfg)
    base_cfg.case, base_cfg.value, base_cfg.gap, base_cfg.unit, base_cfg.shapes = "length", 10, 0.5, "cm", ["box"]
    scene = generate_ou(base_cfg)
    for obj in scene["objects"]:
        set_geometry(obj, {"kind": "box", "size_m": [0.06, 0.04, 0.06]})
        obj["position_m"][2] = settings.TABLE_HEIGHT + 0.03
        obj["rotation_xyzw"] = [0, 0, 0, 1]
        obj["selectable"] = True
    scene["query"] = {}
    scene["reference_frame"] = table_reference_frame()
    return scene


def make_reference(scene, label, position, height=0.06, angle=0.0):
    obj = deepcopy(scene["objects"][0])
    set_geometry(obj, {"kind": "box", "size_m": [0.08, 0.04, height]})
    obj.update(
        object_id=f"reference_{label}",
        name=f"reference_{label}",
        selectable=False,
        position_m=[*position, settings.TABLE_HEIGHT + height / 2],
        rotation_xyzw=[0, 0, math.sin(angle / 2), math.cos(angle / 2)],
        color_rgb=[0.65, 0.005, 0.005] if label == "red" else [0.005, 0.02, 0.65],
        marked_local_axis=True,
    )
    return obj


def constant(value, unit):
    scale = {"cm": 0.01, "degrees": math.pi / 180, "cm2": 1e-4, "cm3": 1e-6}[unit]
    return {"kind": "constant", "value": value, "unit": unit, "value_si": value * scale}


def finish(scene, cfg, predicate, family):
    def add_spatial_tolerance(term):
        if term["op"] == "and":
            for child in term["terms"]:
                add_spatial_tolerance(child)
        elif term["op"] == "eq" and term["measure"]["kind"] in (
            "distance",
            "clearance",
            "offset",
            "coordinate",
            "boundary_distance",
        ):
            term.update(tolerance_si=0.001, tolerance=0.1, unit="cm")

    add_spatial_tolerance(predicate)
    rng = np.random.default_rng(cfg.seed + 101)
    candidates = [o for o in scene["objects"] if o.get("selectable", True)]
    refs = [o for o in scene["objects"] if not o.get("selectable", True)]
    rng.shuffle(candidates)
    for index, obj in enumerate(candidates):
        obj["name"] = obj["object_id"] = f"object_{index}"
        # References keep their identifying colors; candidates offer no color cue.
        obj["color_rgb"] = list(settings.CANDIDATE_COLOR)
    scene["objects"] = candidates + refs
    scene.update(
        task=family,
        family=family,
        case=cfg.case,
        case_id=f"{family}/{cfg.case}",
        predicate=predicate,
        reference_count=len(refs),
        minimum_gap_si=constant(cfg.gap, cfg.unit)["value_si"],
    )
    if family == "quantitative_constraint_grounding":
        # Conjunctions and filters do not generally define a single scalar rank.
        scene.pop("target_rank", None)
    indices = identify(scene)
    if len(indices) != 1:
        raise ValueError(f"{cfg.case}: expected one valid target, found {len(indices)}")
    scene["target_index"] = indices[0]
    scene["target_id"] = scene["objects"][indices[0]]["object_id"]
    from tasks.common.appearance import apply_appearance

    apply_appearance(scene)
    seal(scene)
    validate(scene)
    return scene


def generate(cfg):
    if cfg.case not in CASES:
        raise ValueError("Unsupported relational case")
    angular = cfg.case in ("orientation", "relative_orientation")
    if cfg.unit != ("degrees" if angular else "cm"):
        raise ValueError("Relational positions use cm; orientations use degrees")
    if not math.isfinite(cfg.gap) or cfg.gap <= 0:
        raise ValueError("Distractor gap must be finite and positive")
    scene = base_scene(cfg)
    candidates = scene["objects"]
    local = cfg.case in ("relative_orientation", "reference_frame")
    ref = make_reference(scene, "red", [-0.24, 0.02], angle=math.pi / 4 if local else 0)
    no_reference = cfg.case in ("absolute_position", "boundary_distance", "orientation")
    refs = [] if no_reference else [ref]
    expression = {
        "absolute_position": {"kind": "coordinate", "axis": 0},
        "directional_offset": {"kind": "offset", "axis": 0, "frame": "table", "reference": "reference_red"},
        "center_distance": {"kind": "distance", "planar": True, "reference": "reference_red"},
        "surface_clearance": {"kind": "clearance", "reference": "reference_red"},
        "boundary_distance": {
            "kind": "boundary_distance",
            "axis": 0,
            "side": "lower",
            "boundary_m": -settings.TABLE_SIZE[0] / 2,
        },
        "orientation": {"kind": "orientation"},
        "relative_orientation": {"kind": "orientation", "reference": "reference_red"},
        "reference_frame": {"kind": "offset", "axis": 0, "frame": "reference", "reference": "reference_red"},
    }[cfg.case]
    target = constant(cfg.value, cfg.unit)
    rank = cfg.target_rank if cfg.target_rank is not None else cfg.candidate_count // 2
    rng = np.random.default_rng(cfg.seed)
    scene["objects"] = candidates + refs
    for restart in range(100):
        placed = list(refs)
        for index, obj in enumerate(candidates):
            value = constant(cfg.value + (index - rank) * cfg.gap, cfg.unit)["value_si"]
            if cfg.case == "center_distance" and value < 0.06:
                raise InfeasibleScenario(
                    "Center distance is below the reference/candidate collision and grasp clearance"
                )
            if angular:
                if not 0 <= value < math.pi:
                    raise InfeasibleScenario("Orientation candidates must be distinct within [0, 180) degrees")
                angle = value + (math.pi / 4 if local else 0)
                set_geometry(obj, {"kind": "box", "size_m": [0.10, 0.04, 0.06]})
                obj["rotation_xyzw"] = [0, 0, math.sin(angle / 2), math.cos(angle / 2)]
            for _ in range(1000):
                xy = rng.uniform([-0.45, -0.12], [0.32, 0.32])
                if cfg.case == "absolute_position":
                    xy[0] = value
                elif cfg.case == "boundary_distance":
                    xy[0] = -settings.TABLE_SIZE[0] / 2 + 0.03 + value
                elif cfg.case in ("directional_offset", "reference_frame"):
                    delta = [value, rng.uniform(-0.20, 0.30)]
                    angle = math.pi / 4 if local else 0
                    xy = np.asarray(ref["position_m"][:2]) + np.array(
                        [
                            math.cos(angle) * delta[0] - math.sin(angle) * delta[1],
                            math.sin(angle) * delta[0] + math.cos(angle) * delta[1],
                        ]
                    )
                elif cfg.case in ("center_distance", "surface_clearance"):
                    if value <= 0:
                        raise InfeasibleScenario("Distance candidates must be positive")
                    angle = rng.uniform(-math.pi, math.pi)
                    direction = np.asarray([math.cos(angle), math.sin(angle)])
                    distance = value
                    if cfg.case == "surface_clearance":
                        # Both boxes are axis-aligned and overlap vertically.
                        # First solve separation from one face. Beyond the next
                        # face, solve the quadratic distance to the corner.
                        half = (np.asarray(obj["size_m"][:2]) + ref["size_m"][:2]) / 2
                        ray = np.abs(direction)
                        crossing = np.divide(half, ray, out=np.full(2, np.inf), where=ray > 0)
                        axis = int(np.argmin(crossing))
                        distance = (half[axis] + value) / ray[axis]
                        if distance * ray[1 - axis] > half[1 - axis]:
                            a = float(ray @ ray)
                            b = float(ray @ half)
                            c = float(half @ half - value**2)
                            distance = (b + math.sqrt(max(0.0, b * b - a * c))) / a
                    xy = np.asarray(ref["position_m"][:2]) + distance * direction
                obj["position_m"][:2] = list(map(float, xy))
                if placement_valid(obj, placed, minimum_pick_radius_m=MINIMUM_PICK_RADIUS_M):
                    placed.append(obj)
                    break
            else:
                break
        else:
            tolerance = 0.5 if angular else 0.1
            return finish(
                scene,
                cfg,
                {
                    "op": "eq",
                    "measure": expression,
                    "target": target,
                    "tolerance": tolerance,
                    "unit": cfg.unit,
                    "tolerance_si": constant(tolerance, cfg.unit)["value_si"],
                },
                "relational_understanding",
            )
    raise InfeasibleScenario("No feasible relational layout for the requested values, count, and gap")
