# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solution-first multi-object scenes with physical baskets and frozen goals."""

from copy import deepcopy

import numpy as np
import yaml
from assets.geometry import describe, world_aabb
from core.profiles import render_settings
from environments import settings

from tasks.common.appearance import apply_appearance, validate_appearance
from tasks.common.scenarios import ROOT, InfeasibleScenario, digest, table_reference_frame
from tasks.control.goals import quaternion
from tasks.control.scenarios import make_object
from tasks.registry import engine, subgroup_module, subgroup_of_case

from .planning import CASES, TASKS, destination, solve

PROTOCOL = yaml.safe_load((ROOT / "configs/protocols/multi_object.yaml").read_text())
TEMPLATES = yaml.safe_load((ROOT / "configs/instructions/multi_object.yaml").read_text())


def instruction(s):
    """Combine the subgroup's action text with the shared placement requirements."""
    case = s["case"]
    action = subgroup_module(subgroup_of_case(case), "planning").describe(s)
    if s["template_version"] >= 2 and (case.startswith("CP") or case.startswith("FDP-1")):
        action += " Use at least two objects for each requested structure or collection."
    if s["template_version"] >= 3:
        return action + " All placements must be released and stable."
    return (
        action + " Distances use the white table origin and red +X/blue +Y 12 cm strips. "
        "All placements must be released and stable."
    )


def seal(s):
    keys = (
        "objects",
        "scene_seed",
        "appearance",
        "randomization",
        "settings_digest",
        "reference_frame",
        "rendering",
        "baskets",
    )
    s["scene_id"] = digest({k: s[k] for k in keys if k in s})
    s["control_group_id"] = digest([o["geometry"] for o in s["objects"]])
    s["scenario_id"] = digest(dict(scene=s["scene_id"], case=s["case"], goal=s["numerical_goal"], protocol=s["goal"]))
    if "diagnostic_intervention" in s:
        s["scenario_id"] = digest([s["scenario_id"], s["diagnostic_intervention"]])
    if "feedback_bounds_version" in s:
        s["scenario_id"] = digest([s["scenario_id"], s["feedback_bounds_version"]])
    template = TEMPLATES["templates"][s["template_id"]]
    if s["template_version"] not in (1, 2, TEMPLATES["version"]) or template["split"] != s["template_split"]:
        raise ValueError("Multi-object instruction template mismatch")
    action = instruction(s)
    s["instruction"] = template["text"].format(action=action, action_lower=action[0].lower() + action[1:])
    s["episode_id"] = digest(
        dict(scenario=s["scenario_id"], instruction=s["instruction"], split=s["split"], template=s["template_id"])
    )
    s["fingerprint"] = digest({k: v for k, v in s.items() if k != "fingerprint"})
    return s


def generate(cfg):
    if cfg.case not in CASES or cfg.task != subgroup_of_case(cfg.case):
        raise ValueError("Multi-object case/task mismatch")
    rng = np.random.default_rng(cfg.seed)
    if not cfg.problems:
        raise ValueError("Provide at least one solution-first problem")
    index = int(rng.integers(len(cfg.problems))) if cfg.problem_index is None else cfg.problem_index
    if type(index) is not int or not 0 <= index < len(cfg.problems):
        raise ValueError("problem_index must identify an existing problem")
    problem = deepcopy(cfg.problems[index])
    goal = problem["numerical_goal"]
    objects = []
    layout = np.asarray(cfg.initial_positions_xy_m, dtype=float)
    if len(problem["objects"]) > len(layout):
        raise InfeasibleScenario("Object count exceeds configured nonoverlapping spawn positions")
    layout = layout[rng.permutation(len(layout))[: len(problem["objects"])]]
    for i, geometry in enumerate(problem["objects"]):
        xy = layout[i] + rng.uniform(-cfg.position_jitter_m, cfg.position_jitter_m, 2)
        obj = make_object(chr(65 + i), geometry, xy, float(rng.choice(cfg.initial_yaw_deg)), settings.CANDIDATE_COLOR)
        objects.append(obj)
    n = len(objects)
    plan = solve(cfg.case, objects, goal)
    baskets = []
    reference_index = None
    if plan["mode"] == "basket":
        asset = yaml.safe_load((ROOT / cfg.basket_asset).read_text())
        count = max(plan["assignments"].values(), default=0) + 1
        xs = [0.0] if count == 1 else [-0.30, 0.30] if count == 2 else [-0.43, 0.0, 0.43]
        for b, x in enumerate(xs):
            xy = np.array([x, 0.16]) + rng.uniform(-cfg.basket_jitter_m, cfg.basket_jitter_m, 2)
            angle = float(rng.choice(cfg.basket_yaw_deg))
            geometry = dict(
                kind="hollow_box",
                outer_size_m=asset["outer_size_m"],
                inner_size_xy_m=asset["inner_size_xy_m"],
                bottom_m=asset["bottom_m"],
            )
            palette = np.asarray(cfg.basket_color_options, dtype=float)
            if (
                palette.ndim != 2
                or palette.shape[1] != 3
                or not len(palette)
                or not np.isfinite(palette).all()
                or np.any((palette < 0) | (palette > 1))
            ):
                raise ValueError("Basket colors must be a nonempty linear RGB palette in [0,1]")
            color_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 83, b]))
            color = palette[int(color_rng.integers(len(palette)))].tolist()
            obj = make_object(f"basket_{b}", geometry, xy, angle, color, static=True)
            obj.update(asset_usd=asset["usd_path"], asset_id=asset["id"])
            objects.append(obj)
            occupied = (
                sum(value == b for value in plan["assignments"].values())
                if plan["assignments"]
                else max(len(solution[0]) for solution in plan["solutions"])
            )
            slots = asset.get("slots_by_count", {}).get(occupied, asset["slots_xy_m"])
            baskets.append(
                dict(
                    object_index=len(objects) - 1,
                    center_xy_m=xy.tolist(),
                    yaw_rad=float(np.deg2rad(angle)),
                    **{k: asset[k] for k in ("inner_size_xy_m", "bottom_m", "outer_size_m")},
                    slots_xy_m=deepcopy(slots),
                )
            )
    if plan["mode"] == "chain":
        reference_index = len(objects)
        objects.append(
            make_object(
                "reference",
                dict(kind="box", size_m=[0.05, 0.05, 0.05]),
                cfg.reference_xy_m,
                0,
                (0.7, 0.005, 0.005),
                static=True,
            )
        )
    if not np.isfinite(cfg.contact_offset_m) or not 0 < cfg.contact_offset_m <= 0.005:
        raise ValueError("Multi-object contact offset must be between 0 and 5 mm")
    for obj in objects:
        obj["contact_offset_m"] = cfg.contact_offset_m
    templates = [k for k, v in TEMPLATES["templates"].items() if v["split"] == cfg.template_split]
    s = dict(
        schema_version=1,
        benchmark="Benchmark",
        category="multi_object",
        task=TASKS[cfg.case[:-1]],
        family=subgroup_of_case(cfg.case),
        case=cfg.case,
        case_id=cfg.case,
        condition="numerical",
        level="explicit",
        split=cfg.split,
        seed=cfg.seed,
        scene_seed=cfg.seed,
        candidate_count=n,
        target_index=0,
        target_id="multi_object",
        objects=objects,
        numerical_goal=goal,
        plan=plan,
        baskets=baskets,
        reference_index=reference_index,
        construction_xy_m=cfg.construction_xy_m,
        template_id=cfg.template_id or templates[0],
        template_split=cfg.template_split,
        template_version=TEMPLATES["version"],
        manipulation_protocol=PROTOCOL["id"],
        goal=deepcopy(PROTOCOL),
        settings_digest=digest(settings._SHARED_SETTINGS),
        randomization=deepcopy(cfg.randomization),
        render_profile=cfg.render_profile,
        rendering=render_settings(cfg.render_profile),
        reference_frame=table_reference_frame(),
        observation_profile="four_view_rgbd_256_calibrated_table_frame",
        observability_status="unverified",
    )
    if cfg.diagnostic_intervention is not None:
        s["diagnostic_intervention"] = deepcopy(cfg.diagnostic_intervention)
    if plan["mode"] == "chain":
        s["feedback_bounds_version"] = 2
    apply_appearance(s)
    seal(s)
    validate(s)
    return s


def validate(s):
    if engine(s) != "multi_object" or s["case"] not in CASES or s["goal"] != PROTOCOL:
        raise ValueError("Invalid Multi-object protocol")
    validate_appearance(s)
    expected_frame = (
        table_reference_frame()
        if s["template_version"] >= 3
        else dict(
            id="table",
            origin_m=[0.0, 0.0, settings.TABLE_HEIGHT],
            axes="+X right, +Y forward, +Z up",
            cue="red +X and blue +Y strips, each 12 cm long",
        )
    )
    if s["reference_frame"] != expected_frame:
        raise ValueError("Table frame must match its instruction and visual contract")
    if s.get("feedback_bounds_version", 2) != 2:
        raise ValueError("Unknown achieved-surface measurement contract")
    intervention = s.get("diagnostic_intervention")
    if intervention is not None:
        delta = np.asarray(intervention.get("offset_xy_m"), dtype=float)
        if s["split"] != "diagnostic" or not s["case"].startswith("FDP-2") or set(intervention) != {"offset_xy_m"}:
            raise ValueError("Controlled intervention is only allowed for diagnostic FDP-2 episodes")
        if delta.shape != (2,) or not np.isfinite(delta).all() or not 0.02 <= np.linalg.norm(delta) <= 0.03:
            raise ValueError("Diagnostic perturbation must be a 2–3 cm horizontal shift after A is released")
    objects = s["objects"]
    n = s["candidate_count"]
    if not 3 <= n <= 7 or s["plan"] != solve(s["case"], objects, s["numerical_goal"]):
        raise ValueError("Invalid numerical solution enumeration")
    if s["split"] not in ("train", "eval", "diagnostic") or s["split"] == "train" and s["template_split"] == "held_out":
        raise ValueError("Invalid language/dataset split")
    bounds = []
    for i, o in enumerate(objects):
        info = describe(o["geometry"])
        if o["size_m"] != info["size_m"] or o["physical_properties_si"] != info["measurements"]:
            raise ValueError("Object measurements must be derived from its complete geometry")
        if i < n and (min(o["size_m"]) < 0.035 or max(o["size_m"]) > 0.18 or o["grasp_width_m"] > 0.07):
            raise InfeasibleScenario("Multi-object candidate outside qualified size/grasp envelope")
        lo, hi = world_aabb(o["geometry"], o["position_m"], o["rotation_xyzw"])
        if np.any(
            np.abs(o["position_m"][:2]) + np.asarray(o["size_m"][:2]) / 2 > np.asarray(settings.TABLE_SIZE[:2]) / 2
        ):
            raise InfeasibleScenario("Object outside tabletop")
        if any(np.all(hi[:2] + 0.012 > a[:2]) and np.all(b[:2] + 0.012 > lo[:2]) for a, b in bounds):
            raise InfeasibleScenario("Initial objects overlap or lack finger clearance")
        if "asset_usd" in o and not (ROOT / o["asset_usd"]).is_file():
            raise ValueError("Basket asset path is missing")
        bounds.append((lo, hi))
    for basket in s["baskets"]:
        if n > len(basket["slots_xy_m"]) and not s["plan"]["assignments"]:
            # A cumulative task only needs the selected solution to fit.
            if max(len(solution[0]) for solution in s["plan"]["solutions"]) > len(basket["slots_xy_m"]):
                raise InfeasibleScenario("Basket slot capacity is insufficient")
    for solution in s["plan"]["solutions"]:
        validate_placement_geometry(s, bounds, solution)
    if seal(deepcopy(s)) != s:
        raise ValueError("Multi-object scene identity mismatch")
    return dict(objects=len(objects), targets=n, control_group_id=s["control_group_id"])


def validate_placement_geometry(s, bounds, groups):
    """Check each numerical solution fits physically, without simulating a grasp."""
    objects = s["objects"]
    poses = np.array([o["position_m"] + o["rotation_xyzw"] for o in objects])
    lo = np.array([v[0] for v in bounds])
    hi = np.array([v[1] for v in bounds])
    obs = dict(object_pose_xyzw=poses, object_aabb_min=lo, object_aabb_max=hi)
    plan = s["plan"]
    used = {}
    for group, indices in enumerate(groups):
        placed = []
        for ordinal, i in enumerate(indices):
            b = plan["assignments"].get(objects[i]["object_id"], group)
            slot = used.get(b, 0) if plan["mode"] == "basket" else ordinal
            if plan["mode"] == "basket" and slot >= len(s["baskets"][b]["slots_xy_m"]):
                raise InfeasibleScenario("Basket allocation capacity exceeded")
            point, angle = destination(s, i, b, slot, obs, placed)
            if np.linalg.norm(np.asarray(point[:2]) - settings.ROBOT_POSITION[:2]) > 0.76 or point[2] > 0.75 + 0.30:
                raise InfeasibleScenario("Planned placement outside conservative reach envelope")
            obs["object_pose_xyzw"][i] = [*point, *quaternion(angle)]
            obs["object_aabb_min"][i], obs["object_aabb_max"][i] = world_aabb(
                objects[i]["geometry"], point, quaternion(angle)
            )
            low, high = obs["object_aabb_min"][i], obs["object_aabb_max"][i]
            if np.any(low[:2] < -np.asarray(settings.TABLE_SIZE[:2]) / 2) or np.any(
                high[:2] > np.asarray(settings.TABLE_SIZE[:2]) / 2
            ):
                raise InfeasibleScenario("Construction extends outside the tabletop")
            if plan["mode"] == "basket":
                basket = s["baskets"][b]
                local = np.asarray(basket["slots_xy_m"][slot])
                if np.any(
                    np.abs(local) + np.asarray(objects[i]["size_m"][:2]) / 2 + 0.012
                    > np.asarray(basket["inner_size_xy_m"]) / 2
                ):
                    raise InfeasibleScenario("Basket placement lacks whole-object/finger clearance")
            for j in range(s["candidate_count"]):
                if j == i:
                    continue
                overlap = np.minimum(high, obs["object_aabb_max"][j]) - np.maximum(low, obs["object_aabb_min"][j])
                if np.all(overlap > 1e-6):
                    raise InfeasibleScenario("Planned components interpenetrate another object")
            if plan["mode"] == "tower" and placed:
                below = placed[-1]
                if np.any(low[:2] < obs["object_aabb_min"][below, :2] - 0.001) or np.any(
                    high[:2] > obs["object_aabb_max"][below, :2] + 0.001
                ):
                    raise InfeasibleScenario("Tower component lacks full support footprint")
            used[b] = slot + 1
            placed.append(i)
