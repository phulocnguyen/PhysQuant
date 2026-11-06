# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared immutable scene identities, layout validation, and instruction rendering."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from assets.geometry import describe, surface_area, world_aabb
from core.units import equal, to_si
from environments import settings

from tasks.common.picking import PROTOCOL as PICK_PROTOCOL
from tasks.common.placement import PROTOCOL, goal_bounds
from tasks.registry import SELECTION_SUBGROUPS, engine, subgroup

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = yaml.safe_load((ROOT / "configs/instructions/selection.yaml").read_text())
# Very close targets can satisfy the outer reach bound but hit wrist limits
# with the overhead grasp. This is a conservative sampling bound, not full IK.
MINIMUM_PICK_RADIUS_M = 0.32


class InfeasibleScenario(ValueError):
    """A declared measurement/count combination cannot fit the qualified embodiment."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def table_reference_frame():
    """Define the unmarked tabletop frame without changing world-space task goals."""
    width, depth = settings.TABLE_SIZE[:2]
    robot_x, robot_y = settings.ROBOT_POSITION[:2]
    if width <= depth or robot_y >= 0 or depth / 2 + robot_y >= width / 2 - abs(robot_x):
        raise ValueError("The table frame requires the robot at the negative-Y long edge")
    return dict(
        id="table",
        version=2,
        origin_m=[0.0, 0.0, settings.TABLE_HEIGHT],
        axes="+X right, +Y away from the robot-side edge, +Z up",
        table_size_xy_m=[width, depth],
        cue="table edges, with viewpoint from the robot-side edge toward the opposite edge",
        visual_axes=False,
    )


def compatibility_key(scene):
    """Assets and runtime settings that must remain unchanged inside a worker.

    Poses, query, template, and seeded camera/light/robot samples can reset safely.
    Geometry, count and reference markers require a fresh worker. New appearance
    scenes can update candidate materials at reset; legacy materials remain fixed.
    """
    return digest(
        {
            "objects": [
                {
                    key: None
                    if key == "color_rgb" and scene.get("appearance") and obj.get("selectable", True)
                    else obj.get(key)
                    for key in (
                        "name",
                        "geometry",
                        "color_rgb",
                        "mass_kg",
                        "marked_local_axis",
                        "kinematic",
                        "asset_usd",
                        "contact_offset_m",
                    )
                }
                for obj in scene["objects"]
            ],
            "mutable_appearance": bool(scene.get("appearance")),
            **({"appearance_version": 2} if scene.get("appearance", {}).get("version", 1) >= 2 else {}),
            **{
                key: scene.get(key)
                for key in (
                    "task",
                    "goal",
                    "settings_digest",
                    "observation_profile",
                    "randomization",
                    "reference_frame",
                    "rendering",
                )
            },
        }
    )


LAUNCH_MISMATCH = (
    "Persistent batches require one launch configuration: task, protocol, settings, rendering, and randomization"
)


def launch_key(scene):
    """Settings that must stay fixed for one simulator launch.

    Scenes that share this key can run in one worker; the worker rebuilds the
    stage in the running app when the ``compatibility_key`` (geometry) changes.
    """
    return digest(
        {
            key: scene.get(key)
            for key in (
                "category",
                "task",
                "goal",
                "settings_digest",
                "observation_profile",
                "randomization",
                "rendering",
            )
        }
    )


def measurement(obj, attribute):
    if attribute == "surface_area":
        return surface_area(obj["geometry"])
    return describe(obj["geometry"])["measurements"][attribute]


def render_instruction(scenario):
    query = scenario["query"]
    noun = scenario["requested_shape"].replace("_", " ") if scenario["requested_shape"] else "object"
    clauses = []
    for attribute, value in query.items():
        label = "total surface area" if attribute == "surface_area" else attribute
        unit = {"cm2": "cm²", "cm3": "cm³"}.get(value["unit"], value["unit"])
        clauses.append(f"{label} is {value['value']:g} {unit}")
    if "predicate" in scenario:
        from tasks.common.predicates import describe_predicate

        clauses = [describe_predicate(scenario["predicate"], table_relative=scenario["template_version"] >= 2)]
    selection = f"the {noun} whose " + " and ".join(clauses)
    template = TEMPLATES["templates"][scenario["template_id"]]
    if template["split"] != scenario["template_split"] or scenario["template_version"] not in (
        1,
        2,
        TEMPLATES["version"],
    ):
        raise ValueError("Instruction template version or split mismatch")
    text = template["text" if scenario["template_version"] >= 3 else "legacy_text"]
    instruction = text.format(selection=selection)
    return instruction


def identify(scenario):
    if "predicate" in scenario:
        from tasks.common.predicates import matches

        return [
            i
            for i, obj in enumerate(scenario["objects"])
            if obj.get("selectable", True) and matches(obj, scenario["predicate"], scenario)
        ]
    return [
        i
        for i, obj in enumerate(scenario["objects"])
        if (scenario["requested_shape"] is None or obj["category"] == scenario["requested_shape"])
        and all(equal(measurement(obj, key), q["value_si"]) for key, q in scenario["query"].items())
    ]


def seal(scenario):
    """Derive identities without including privileged target labels in scene identity."""
    scene_keys = ("objects", "scene_seed", "observation_profile", "goal", "settings_digest", "randomization")
    scene_payload = {key: scenario[key] for key in scene_keys}
    if "size_limits" in scenario:
        scene_payload["size_limits"] = scenario["size_limits"]
    if scenario.get("reference_frame"):
        scene_payload["reference_frame"] = scenario["reference_frame"]
    if "rendering" in scenario:
        scene_payload["rendering"] = scenario["rendering"]
    if "appearance" in scenario:
        scene_payload["appearance"] = scenario["appearance"]
    scenario["scene_id"] = digest(scene_payload)
    scenario["control_group_id"] = digest(
        sorted(
            json.dumps({"category": o["category"], "geometry": o["geometry"]}, sort_keys=True)
            for o in scenario["objects"]
        )
    )
    scenario["scenario_id"] = digest(
        {
            "scene": scenario["scene_id"],
            "query": scenario["query"],
            "shape": scenario["requested_shape"],
            "protocol": scenario["manipulation_protocol"],
            **({"predicate": scenario["predicate"]} if "predicate" in scenario else {}),
        }
    )
    scenario["instruction"] = render_instruction(scenario)
    scenario["episode_id"] = digest(
        {
            "scenario": scenario["scenario_id"],
            "instruction": scenario["instruction"],
            "template": scenario["template_id"],
            "version": scenario["template_version"],
            "split": scenario["split"],
        }
    )
    scenario["fingerprint"] = digest({k: v for k, v in scenario.items() if k != "fingerprint"})
    return scenario


def layout(objects, seed):
    """Pack conservative collision bounds with gripper access on the tabletop."""
    rng = np.random.default_rng(seed)
    points = [(x, y) for y in settings.CANDIDATE_LAYOUT["y_m"] for x in settings.CANDIDATE_LAYOUT["x_m"]]
    for _ in range(300):
        occupied = []
        available = rng.permutation(len(points)).tolist()
        for obj in objects:
            info = describe(obj["geometry"])
            for index in available:
                xy = np.asarray(points[index])
                pos = [*xy, settings.TABLE_HEIGHT + info["size_m"][2] / 2]
                lo, hi = world_aabb(obj["geometry"], pos, settings.ROBOT_ROTATION)
                if np.linalg.norm(xy - np.asarray(settings.ROBOT_POSITION[:2])) > 0.78:
                    continue
                if np.any(lo[:2] < -np.asarray(settings.TABLE_SIZE[:2]) / 2 + 0.01) or np.any(
                    hi[:2] > np.asarray(settings.TABLE_SIZE[:2]) / 2 - 0.01
                ):
                    continue
                if any(np.all(hi[:2] + 0.025 > a) and np.all(lo[:2] - 0.025 < b) for a, b in occupied):
                    continue
                obj.update(position_m=list(map(float, pos)), rotation_xyzw=list(settings.ROBOT_ROTATION))
                occupied.append((lo[:2], hi[:2]))
                available.remove(index)
                break
            else:
                break
        else:
            return
    raise InfeasibleScenario("No collision-free reachable layout at the requested count and geometry")


def validate(s):
    from tasks.common.appearance import validate_appearance

    validate_appearance(s)
    if s["schema_version"] != 2 or engine(s) != "selection" or subgroup(s["task"]) not in SELECTION_SUBGROUPS:
        raise ValueError("Unsupported versioned benchmark scenario")
    if "size_limits" in s:
        limits = s["size_limits"]
        keys = {"minimum_extent_m", "minimum_planar_span_m", "maximum_extent_m", "maximum_grasp_width_m"}
        if (
            not isinstance(limits, dict)
            or set(limits) != keys
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0
                for v in limits.values()
            )
        ):
            raise ValueError("Candidate size limits require four positive finite distances")
        if not (
            limits["minimum_extent_m"] <= limits["minimum_planar_span_m"] <= limits["maximum_extent_m"]
            and limits["minimum_extent_m"] <= limits["maximum_grasp_width_m"] <= limits["maximum_extent_m"]
        ):
            raise ValueError("Candidate size limits must have ordered minimum and maximum distances")
    protocol = PICK_PROTOCOL if s["template_version"] >= 3 else PROTOCOL
    if s["manipulation_protocol"] != protocol["id"] or s["goal"] != protocol:
        raise ValueError("Unsupported manipulation protocol or instruction version")
    if s.get("reference_frame"):
        expected_frame = table_reference_frame() if s["template_version"] >= 2 else "table_red_x_blue_y"
        if s["reference_frame"] != expected_frame:
            raise ValueError("Table reference frame differs from the instruction contract")
    if not np.isfinite(s["minimum_gap_si"]) or s["minimum_gap_si"] <= 0:
        raise ValueError("Distractor gap must be finite and positive")
    if s["split"] not in ("train", "eval", "diagnostic") or (
        s["split"] == "train" and s["template_split"] == "held_out"
    ):
        raise ValueError("Invalid dataset/language split")
    if "predicate" in s:
        from tasks.common.predicates import validate_predicate

        validate_predicate(s["predicate"], s)
    if s["candidate_count"] != sum(o.get("selectable", True) for o in s["objects"]):
        raise ValueError("Incorrect candidate count")
    for attr, q in s["query"].items():
        if not equal(q["value_si"], to_si(q["value"], q["unit"], attr)):
            raise ValueError("Inconsistent numerical units")
    for obj in s["objects"]:
        info = describe(obj["geometry"])
        if info["measurements"] != obj["physical_properties_si"] or not info["grasp_feasible"]:
            raise ValueError("Incorrect or infeasible object geometry")
        limits = s.get("size_limits")
        if limits and obj.get("selectable", True):
            size = info["size_m"]
            if (
                min(size) < limits["minimum_extent_m"] - 1e-9
                or max(size[:2]) < limits["minimum_planar_span_m"] - 1e-9
                or max(size) > limits["maximum_extent_m"] + 1e-9
                or info["grasp_width_m"] > limits["maximum_grasp_width_m"] + 1e-9
            ):
                raise InfeasibleScenario(f"{obj['object_id']}: outside primary candidate size limits")
        pose = np.asarray(obj["position_m"] + obj["rotation_xyzw"])
        if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1):
            raise ValueError("Object poses require finite XYZ and a unit XYZW quaternion")
    identities = [obj["object_id"] for obj in s["objects"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate object identity")
    for index, obj in enumerate(s["objects"]):
        if not placement_valid(obj, s["objects"][:index], goal=PROTOCOL if protocol == PROTOCOL else None):
            raise ValueError("Invalid initial layout, reach envelope, or goal intersection")
    matches = identify(s)
    if matches != [s["target_index"]] or s["target_id"] != s["objects"][matches[0]]["object_id"]:
        raise ValueError("Incorrect unique target")
    for obj in s["objects"]:
        for attr, q in s["query"].items():
            distance = abs(measurement(obj, attr) - q["value_si"])
            if distance > 1e-12 and distance < s["minimum_gap_si"] - 1e-12:
                raise ValueError("Numerical distractor is below the declared gap")
    rebuilt = seal(deepcopy(s))
    if rebuilt != s:
        raise ValueError("Scene, instruction, or identity mismatch")
    return {"objects": len(s["objects"]), "targets": 1, "control_group_id": s["control_group_id"]}


def placement_valid(obj, others, *, goal=None, minimum_pick_radius_m=0.0):
    lo, hi = world_aabb(obj["geometry"], obj["position_m"], obj["rotation_xyzw"])
    if np.any(lo[:2] < -np.asarray(settings.TABLE_SIZE[:2]) / 2 + 0.01) or np.any(
        hi[:2] > np.asarray(settings.TABLE_SIZE[:2]) / 2 - 0.01
    ):
        return False
    radius = np.linalg.norm(np.asarray(obj["position_m"][:2]) - settings.ROBOT_POSITION[:2])
    if radius > 0.78 or (obj.get("selectable", True) and radius < minimum_pick_radius_m):
        return False
    if goal is not None:
        goal_lo, goal_hi = goal_bounds(goal)
        if np.all(hi[:2] + 0.02 > goal_lo) and np.all(lo[:2] - 0.02 < goal_hi):
            return False
    for other in others:
        a, b = world_aabb(other["geometry"], other["position_m"], other["rotation_xyzw"])
        # A symmetric gap test stays invariant when candidates are shuffled.
        # Expanding opposite faces can disagree at exactly 2 cm due to roundoff.
        if np.all(np.maximum(a[:2] - hi[:2], lo[:2] - b[:2]) < 0.02 - 1e-12):
            return False
    return True
