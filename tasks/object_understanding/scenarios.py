# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit-count OU generation with immutable scene, query, and language identities."""

from copy import deepcopy

import numpy as np
from assets.geometry import construct, describe
from core.profiles import render_settings, validate_count
from core.units import equal, to_si
from environments import settings
from environments.randomization import resolve_randomization, sample_episode

from tasks.common.picking import PROTOCOL

ATTRIBUTES = ("length", "width", "height", "radius", "diameter", "surface_area", "volume")


from tasks.common.scenarios import TEMPLATES, InfeasibleScenario, digest, identify, layout, measurement, seal, validate


def generate(cfg):
    """Create one explicitly configured selection, pick, and instruction episode."""
    validate_count(cfg.candidate_count)
    if cfg.case not in ATTRIBUTES:
        raise ValueError("Unsupported object-understanding attribute or candidate count")
    if cfg.template_split not in ("seen", "held_out") or cfg.split not in ("train", "eval", "diagnostic"):
        raise ValueError("Invalid language or dataset split")
    if cfg.split == "train" and cfg.template_split == "held_out":
        raise ValueError("Held-out language cannot enter the training split")
    query = {cfg.case: {"value": cfg.value, "unit": cfg.unit, "value_si": to_si(cfg.value, cfg.unit, cfg.case)}}
    gap = to_si(cfg.gap, cfg.unit, cfg.case)
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("Distractor gap must be finite and positive")
    rank = cfg.target_rank if cfg.target_rank is not None else cfg.candidate_count // 2
    if not 0 <= rank < cfg.candidate_count:
        raise ValueError("target_rank must be within the candidate count")
    rng = np.random.default_rng(cfg.seed)
    shape_order = rng.permutation(cfg.shapes).tolist()
    objects = []
    for index in range(cfg.candidate_count):
        value = query[cfg.case]["value_si"] + (index - rank) * gap
        if value <= 0:
            raise InfeasibleScenario("Requested count, rank and gap produce a nonpositive measurement")
        category = shape_order[index % len(shape_order)]
        obj = construct(category, cfg.case, {cfg.case: value})
        if category == "cylinder" and cfg.case in ("radius", "diameter"):
            obj["geometry"]["height_m"] = settings.CANDIDATE_CYLINDER_HEIGHT
        info = describe(obj["geometry"])
        obj["physical_properties_si"] = info["measurements"]
        if not info["grasp_feasible"]:
            raise InfeasibleScenario(f"{category}/{cfg.case}/{value:g}: outside the supported grasp envelope")
        if not equal(measurement(obj, cfg.case), value):
            raise ValueError(f"{category} does not support exact {cfg.case} construction")
        obj.update(
            size_m=info["size_m"],
            shape=obj["geometry"]["kind"],
            radius_m=info["measurements"]["radius"],
            grasp_width_m=info["grasp_width_m"],
            grasp_yaw_offset_rad=info["grasp_yaw_offset_rad"],
            grasp_offset_local_m=info["grasp_offset_local_m"],
            mass_kg=0.05,
        )
        objects.append(obj)
    rng.shuffle(objects)
    for index, obj in enumerate(objects):
        obj.update(
            object_id=f"object_{index}",
            name=f"object_{index}",
            color_rgb=list(settings.CANDIDATE_COLOR),
        )
    layout(objects, cfg.seed)
    randomization = resolve_randomization(cfg.randomization or {"enabled": False})
    objects = sample_episode(
        objects, randomization, seed=cfg.seed, table_size=settings.TABLE_SIZE, colors=settings.COLORS
    )
    templates = [key for key, value in TEMPLATES["templates"].items() if value["split"] == cfg.template_split]
    template_id = cfg.template_id or templates[cfg.seed % len(templates)]
    if template_id not in templates:
        raise ValueError("Template does not belong to the requested language split")
    scenario = {
        "schema_version": 2,
        "benchmark": "Benchmark",
        "category": "selection",
        "task": "object_understanding",
        "case": cfg.case,
        "case_id": f"object_understanding/{cfg.case}",
        "family": "object_understanding",
        "condition": "numerical",
        "level": "explicit",
        "seed": cfg.seed,
        "scene_seed": cfg.seed,
        "split": cfg.split,
        "query": query,
        "requested_shape": None,
        "objects": objects,
        "candidate_count": cfg.candidate_count,
        "size_limits": deepcopy(settings.CANDIDATE_LIMITS),
        "minimum_gap_si": gap,
        "target_rank": rank,
        "template_id": template_id,
        "template_split": cfg.template_split,
        "template_version": TEMPLATES["version"],
        "manipulation_protocol": PROTOCOL["id"],
        "goal": deepcopy(PROTOCOL),
        "observability_status": "unverified",
        "observation_profile": "four_view_rgbd_256_calibrated",
        "settings_digest": digest(settings._SHARED_SETTINGS),
        "randomization": randomization if randomization["enabled"] else {"enabled": False},
    }
    if cfg.render_profile != "quality":
        scenario.update(render_profile=cfg.render_profile, rendering=render_settings(cfg.render_profile))
    matches = identify(scenario)
    if len(matches) != 1:
        raise ValueError("Instruction must select exactly one object")
    scenario["target_index"] = matches[0]
    scenario["target_id"] = objects[matches[0]]["object_id"]
    from tasks.common.appearance import apply_appearance

    apply_appearance(scenario)
    seal(scenario)
    validate(scenario)
    return scenario
