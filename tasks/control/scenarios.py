# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable control scenes: sampled assets, independently solved goals, one instruction."""

from copy import deepcopy

import numpy as np
import yaml
from assets.geometry import describe, world_aabb
from core.profiles import render_settings
from environments import settings

from tasks.common.scenarios import ROOT, InfeasibleScenario, digest, table_reference_frame
from tasks.control.goals import CASES, action_text, horizontal_distance, quaternion, solve, vector, yaw
from tasks.registry import CORE_TASKS, engine, subgroup_of_case

PROTOCOL = yaml.safe_load((ROOT / "configs/protocols/control.yaml").read_text())
TEMPLATES = yaml.safe_load((ROOT / "configs/instructions/control.yaml").read_text())
GOAL_KEYS = {
    **{case: {"displacement_cm"} for case in ("ST-1A", "ST-1B", "ST-1C")},
    **{case: {"angle_deg"} for case in ("ST-2A", "ST-2B", "ST-2C")},
    "TP-1A": {"position_cm"},
    "TP-1B": {"offset_cm"},
    "TP-1C": {"offset_cm"},
    "TP-2A": {"axis", "sign", "gap_cm"},
    "TP-2B": {"axis", "sign", "gap_cm"},
    "TP-2C": {"gaps_cm"},
    "CM-1A": {"steps_cm"},
    "CM-1B": {"steps_cm"},
    "CM-1C": {"offsets_cm"},
    "CM-2A": {"displacement_cm", "minimum_clearance_cm"},
    "CM-2B": {"displacement_cm", "max_yaw_deviation_deg"},
    "CM-2C": {"displacement_cm", "minimum_clearance_cm", "detour_side"},
}


def make_object(identity, geometry, xy, angle, color, *, static=False):
    info = describe(geometry)
    return dict(
        object_id=identity,
        name=identity,
        geometry=deepcopy(geometry),
        category=geometry["kind"],
        shape=geometry["kind"],
        size_m=info["size_m"],
        physical_properties_si=info["measurements"],
        radius_m=info["measurements"]["radius"],
        grasp_width_m=info["grasp_width_m"],
        grasp_yaw_offset_rad=info["grasp_yaw_offset_rad"],
        grasp_offset_local_m=info["grasp_offset_local_m"],
        mass_kg=0.05,
        color_rgb=list(color),
        selectable=not static,
        kinematic=static,
        position_m=[*map(float, xy), settings.TABLE_HEIGHT + info["size_m"][2] / 2],
        rotation_xyzw=quaternion(np.deg2rad(angle)),
    )


def generate(cfg):
    if cfg.case not in CASES or cfg.task != subgroup_of_case(cfg.case):
        raise ValueError("Control case and task subgroup must agree")
    if set(cfg.numerical_goal) != GOAL_KEYS[cfg.case]:
        raise ValueError("Missing or unknown numerical goal fields")
    seeds = {
        name: int(child.generate_state(1)[0])
        for name, child in zip(
            ("geometry", "placement", "numerical_goal", "appearance", "camera", "language", "robot"),
            np.random.SeedSequence(cfg.seed).spawn(7),
        )
    }
    geometry_rng = np.random.default_rng(seeds["geometry"])
    placement_rng = np.random.default_rng(seeds["placement"])
    geometry = cfg.geometry_options[int(geometry_rng.integers(len(cfg.geometry_options)))]
    xy = vector(cfg.initial_xy_m, 2, "initial_xy_m")
    jitter = vector(cfg.initial_jitter_xy_m, 2, "initial_jitter_xy_m")
    if np.any(jitter < 0):
        raise ValueError("Position jitter ranges must be nonnegative")
    xy += placement_rng.uniform(-jitter, jitter)
    angle = float(placement_rng.choice(cfg.initial_yaw_deg))
    if not np.isfinite(angle):
        raise ValueError("Initial yaw must be finite")
    if cfg.case == "ST-2A" and (len(cfg.initial_yaw_deg) != 1 or angle != 0):
        raise ValueError("ST-2A uses canonical yaw zero; use ST-2B for varied initial yaw")
    objects = [make_object("movable", geometry, xy, angle, settings.CANDIDATE_COLOR)]
    for identity, ref in cfg.references.items():
        if identity not in ("A", "B") or set(ref) - {"geometry", "position_xy_m", "yaw_deg", "jitter_xy_m"}:
            raise ValueError("References support only A/B and geometry/position/yaw/jitter fields")
        offset = vector(ref.get("jitter_xy_m", [0, 0]), 2, "reference jitter")
        if np.any(offset < 0):
            raise ValueError("Reference jitter must be nonnegative")
        ref_xy = vector(ref["position_xy_m"], 2, "reference position") + placement_rng.uniform(-offset, offset)
        objects.append(
            make_object(
                identity,
                ref["geometry"],
                ref_xy,
                ref.get("yaw_deg", 0),
                (0.7, 0.005, 0.005) if identity == "A" else (0.005, 0.02, 0.7),
                static=True,
            )
        )
    if cfg.case == "TP-2C":
        # Solve the second reference from gap constraints, not an unrelated random location.
        a, b = objects[1:]
        gaps = vector(cfg.numerical_goal["gaps_cm"], 2, "gaps_cm") / 100
        b["position_m"][:2] = [
            float(a["position_m"][0] + a["size_m"][0] / 2 + sum(gaps) + objects[0]["size_m"][0] + b["size_m"][0] / 2),
            a["position_m"][1],
        ]
    templates = [k for k, v in TEMPLATES["templates"].items() if v["split"] == cfg.template_split]
    if not templates:
        raise ValueError("Unknown template split")
    template = cfg.template_id or templates[seeds["language"] % len(templates)]
    scenario = dict(
        schema_version=1,
        benchmark="Benchmark",
        category="control",
        task=CORE_TASKS[cfg.case[:-1]],
        case=cfg.case,
        case_id=cfg.case,
        family=subgroup_of_case(cfg.case),
        condition="numerical",
        level="explicit",
        seed=cfg.seed,
        scene_seed=cfg.seed,
        seeds=seeds,
        split=cfg.split,
        template_id=template,
        template_split=cfg.template_split,
        template_version=TEMPLATES["version"],
        candidate_count=1,
        target_index=0,
        target_id="movable",
        objects=objects,
        numerical_goal=deepcopy(cfg.numerical_goal),
        manipulation_protocol=PROTOCOL["id"],
        goal=deepcopy(PROTOCOL),
        size_limits=deepcopy(settings.CANDIDATE_LIMITS),
        settings_digest=digest(settings._SHARED_SETTINGS),
        randomization=deepcopy(cfg.randomization),
        render_profile=cfg.render_profile,
        rendering=render_settings(cfg.render_profile),
        reference_frame=table_reference_frame(),
        observation_profile="four_view_rgbd_256_calibrated_table_frame",
        observability_status="unverified",
        geometry_qualification="analytic envelope only; oracle and rendered review required",
    )
    from tasks.common.appearance import apply_appearance

    apply_appearance(scenario)
    scenario["resolved_goal"] = solve(cfg.case, scenario["numerical_goal"], objects, PROTOCOL)
    seal(scenario)
    validate(scenario)
    return scenario


def seal(s):
    scene_keys = (
        "objects",
        "scene_seed",
        "settings_digest",
        "randomization",
        "rendering",
        "reference_frame",
        "observation_profile",
        "size_limits",
    )
    scene_payload = {key: s[key] for key in scene_keys}
    if "appearance" in s:
        scene_payload["appearance"] = s["appearance"]
    s["scene_id"] = digest(scene_payload)
    # Conservative split grouping: all numerical/paraphrase variants of these assets stay together.
    s["control_group_id"] = digest([{"geometry": o["geometry"], "kinematic": o["kinematic"]} for o in s["objects"]])
    s["scenario_id"] = digest(
        dict(
            scene=s["scene_id"],
            case=s["case"],
            numerical_goal=s["numerical_goal"],
            resolved=s["resolved_goal"],
            protocol=s["goal"],
        )
    )
    template = TEMPLATES["templates"][s["template_id"]]
    if template["split"] != s["template_split"] or s["template_version"] not in (1, 2, 3, 4, TEMPLATES["version"]):
        raise ValueError("Template version/split mismatch")
    action = action_text(s["case"], s["numerical_goal"], table_relative=s["template_version"] >= 4)
    candidate = s.get("appearance", {}).get("candidate")
    if s["template_version"] >= 3 and candidate:
        # Version 5 also colors the "rectangular block" used in rotation tasks.
        # Keep older frozen instructions and episode identities reproducible.
        if s["template_version"] >= 5:
            action = action.replace("orange ", f"{candidate['name']} ")
        else:
            action = action.replace("orange block", f"{candidate['name']} block")
    if s["objects"][0]["geometry"]["kind"] == "cylinder":
        color_name = candidate["name"] if s["template_version"] >= 3 and candidate else "orange"
        action = action.replace(f"{color_name} block", f"{color_name} cylinder")
    frame = "The white origin and red +X/blue +Y strips define the table frame; each strip is 12 cm long."
    if s["template_version"] >= 4:
        frame = ""
    if s["case"].startswith("CM-1"):
        height = s["goal"]["transport_bottom_clearance_m"] * 100
        if s["template_version"] == 1:
            # Preserve pilot identities; that wording overstated the height check.
            frame += f" During the ordered motion keep the bottom {height:g} cm above the table."
        else:
            frame += f" At each waypoint, the bottom must be {height:g} cm above the table."
    text = template.get("legacy_text", template["text"]) if s["template_version"] < 4 else template["text"]
    s["instruction"] = text.format(action=action, action_lower=action[0].lower() + action[1:], frame=frame).strip()
    s["episode_id"] = digest(
        dict(
            scenario=s["scenario_id"],
            instruction=s["instruction"],
            template=s["template_id"],
            version=s["template_version"],
            split=s["split"],
        )
    )
    s["fingerprint"] = digest({k: v for k, v in s.items() if k != "fingerprint"})
    return s


def validate(s):
    from tasks.common.appearance import validate_appearance

    validate_appearance(s)
    if (
        engine(s) != "control"
        or s["schema_version"] != 1
        or s["case"] not in CASES
        or s["task"] != CORE_TASKS[s["case"][:-1]]
    ):
        raise ValueError("Unsupported spatial control schema/task")
    expected_protocol = deepcopy(PROTOCOL)
    if "evidence_version" not in s["goal"]:
        # Preserve the first engineering pilot's trace interpretation. Never
        # silently relabel its pre-lift interval or raw-velocity hold failures.
        expected_protocol.pop("evidence_version")
        expected_protocol.pop("pretransport_max_horizontal_drift_m")
    if s["manipulation_protocol"] != PROTOCOL["id"] or s["goal"] != expected_protocol:
        raise ValueError("Control protocol mismatch")
    expected_frame = (
        table_reference_frame()
        if s["template_version"] >= 4
        else dict(
            id="table",
            origin_m=[0.0, 0.0, settings.TABLE_HEIGHT],
            axes="+X right, +Y forward, +Z up",
            cue="red +X and blue +Y strips, each 12 cm long",
        )
    )
    if s["reference_frame"] != expected_frame:
        raise ValueError("Table frame must match its instruction and visual contract")
    if s["randomization"]["enabled"] and s["randomization"]["target_object"]["enabled"]:
        raise ValueError("Independent object randomization invalidates control goals")
    if s["split"] not in ("train", "eval", "diagnostic") or (
        s["split"] == "train" and s["template_split"] == "held_out"
    ):
        raise ValueError("Invalid split or held-out language in training")
    if set(s["numerical_goal"]) != GOAL_KEYS[s["case"]]:
        raise ValueError("Missing or unknown numerical goal fields")
    objs = s["objects"]
    if (
        s["candidate_count"] != 1
        or s["target_index"] != 0
        or s["target_id"] != "movable"
        or [o["object_id"] for o in objs] != ["movable", *("AB"[: len(objs) - 1])]
    ):
        raise ValueError("Control requires one movable followed by references A/B")
    expected_refs = (
        2
        if s["case"] in ("TP-1C", "TP-2C", "CM-1C")
        else 1
        if s["case"] in ("ST-2C", "TP-1B", "TP-2A", "TP-2B", "CM-2C")
        else 0
    )
    if len(objs) != 1 + expected_refs:
        raise ValueError("Incorrect number of references for case")
    for i, obj in enumerate(objs):
        if (
            obj["geometry"]["kind"] not in ("box", "cylinder")
            or obj["kinematic"] != (i > 0)
            or obj["selectable"] != (i == 0)
        ):
            raise ValueError("Only primitive dynamic movable and kinematic references are supported")
        info = describe(obj["geometry"])
        if info["measurements"] != obj["physical_properties_si"] or info["size_m"] != obj["size_m"]:
            raise ValueError("Geometry annotation mismatch")
        pose = vector(obj["position_m"] + obj["rotation_xyzw"], 7, "pose")
        if not np.isclose(np.linalg.norm(pose[3:]), 1) or abs(pose[3]) + abs(pose[4]) > 1e-9:
            raise ValueError("Initial objects must be upright with normalized XYZW quaternions")
        if abs(pose[2] - settings.TABLE_HEIGHT - info["size_m"][2] / 2) > 1e-8:
            raise ValueError("Initial object must be supported")
        if i == 0:
            limits = s["size_limits"]
            if (
                not info["grasp_feasible"]
                or min(info["size_m"]) < limits["minimum_extent_m"]
                or max(info["size_m"]) > limits["maximum_extent_m"]
                or max(info["size_m"][:2]) < limits["minimum_planar_span_m"]
                or info["grasp_width_m"] > limits["maximum_grasp_width_m"]
            ):
                raise InfeasibleScenario("Movable outside candidate size/grasp limits")
        check_pose(obj, obj["position_m"], yaw(obj["rotation_xyzw"]), objs[:i], margin=0.025)
    resolved = solve(s["case"], s["numerical_goal"], objs, s["goal"])
    if resolved != s["resolved_goal"]:
        raise ValueError("Numerical goal and derived physical target disagree")
    obj = objs[0]
    check_pose(obj, resolved["target_position_m"], resolved["target_yaw_rad"], objs[1:], margin=0.01)
    points = [np.asarray(obj["position_m"]), np.array([*obj["position_m"][:2], resolved["transport_z_m"]])]
    points.extend(map(np.asarray, resolved["oracle_route_m"]))
    points.append(np.asarray(resolved["target_position_m"]))
    if s["case"].startswith("ST-2"):
        # Bound the entire in-place yaw sweep, including intermediate orientations.
        for angle in np.linspace(resolved["initial_yaw_rad"], resolved["target_yaw_rad"], 25):
            check_pose(obj, points[1], angle, objs[1:], margin=0.015)
    for start, end in zip(points, points[1:]):
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm(end - start) / 0.005) + 1)):
            position = (1 - t) * start + t * end
            check_pose(obj, position, resolved["target_yaw_rad"], objs[1:], margin=0.015)
            if s["case"] == "CM-2C":
                lo, hi = world_aabb(obj["geometry"], position, quaternion(resolved["target_yaw_rad"]))
                ref = objs[1]
                a, b = world_aabb(ref["geometry"], ref["position_m"], ref["rotation_xyzw"])
                if horizontal_distance(lo, hi, a, b) < resolved["constraints"]["minimum_obstacle_clearance_m"]:
                    raise InfeasibleScenario("Oracle object route violates obstacle clearance")
    if s["case"] == "CM-2C":
        # The direct path must violate the requirement: otherwise the detour is trivial.
        ref = objs[1]
        a, b = world_aabb(ref["geometry"], ref["position_m"], ref["rotation_xyzw"])
        direct = []
        for t in np.linspace(0, 1, 41):
            position = (1 - t) * np.asarray(obj["position_m"]) + t * np.asarray(resolved["target_position_m"])
            lo, hi = world_aabb(obj["geometry"], position, obj["rotation_xyzw"])
            direct.append(horizontal_distance(lo, hi, a, b))
        if min(direct) >= resolved["constraints"]["minimum_obstacle_clearance_m"]:
            raise InfeasibleScenario("Obstacle does not constrain the direct transport path")
    if seal(deepcopy(s)) != s:
        raise ValueError("Control scene/instruction/identity mismatch")
    return dict(objects=len(objs), targets=1, control_group_id=s["control_group_id"])


def check_pose(obj, position, angle, others, margin):
    lo, hi = world_aabb(obj["geometry"], position, quaternion(angle))
    if (
        np.any(lo[:2] < -np.asarray(settings.TABLE_SIZE[:2]) / 2 + 0.01)
        or np.any(hi[:2] > np.asarray(settings.TABLE_SIZE[:2]) / 2 - 0.01)
        or lo[2] < settings.TABLE_HEIGHT - 1e-6
    ):
        raise InfeasibleScenario("Object or goal outside table support/workspace")
    reach = np.linalg.norm(np.asarray(position[:2]) - settings.ROBOT_POSITION[:2])
    if not 0.28 <= reach <= 0.75 or position[1] < settings.ROBOT_POSITION[1] + 0.25 or hi[2] > 1.25:
        raise InfeasibleScenario("Object pose outside conservative robot reach envelope")
    for ref in others:
        a, b = world_aabb(ref["geometry"], ref["position_m"], ref["rotation_xyzw"])
        if np.all(hi + margin > a) and np.all(lo - margin < b):
            raise InfeasibleScenario("Initial, final or oracle route geometry intersects a reference")
