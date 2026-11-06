# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Balanced target and language sweeps over immutable benchmark scenes."""

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from core.profiles import CANDIDATE_COUNTS, RENDER_PROFILES, validate_count
from core.units import UNITS
from data_collection.config import load_config
from data_collection.manifest import validate_manifest, write_manifest

from tasks.common.scenarios import ROOT, TEMPLATES, digest, measurement, seal, validate
from tasks.common.task import configure
from tasks.registry import subgroup


def scene_episodes(base, *, all_templates=True):
    """Yield each eligible target/template pair once, interleaving target identities."""
    templates = [key for key, value in TEMPLATES["templates"].items() if value["split"] == base["template_split"]]
    if not all_templates:
        templates = [base["template_id"]]
    targets = [i for i, obj in enumerate(base["objects"]) if obj.get("selectable", True)]
    if subgroup(base["task"]) == "quantitative_constraint_grounding":
        targets = [base["target_index"]]
    rng = np.random.default_rng(base["scene_seed"])
    rng.shuffle(targets)
    rng.shuffle(templates)
    used = {target: set() for target in targets}
    template_counts = Counter()
    for _ in templates:
        for target in targets:
            template = min((t for t in templates if t not in used[target]), key=lambda t: template_counts[t])
            used[target].add(template)
            template_counts[template] += 1
            scene = deepcopy(base)
            obj = scene["objects"][target]
            for attr, q in scene["query"].items():
                q["value_si"] = measurement(obj, attr)
                q["value"] = round(q["value_si"] / UNITS[q["unit"]][1], 10)
            scene["target_index"], scene["target_id"] = target, obj["object_id"]
            if subgroup(base["task"]) == "relational_understanding":
                from tasks.common.predicates import quantity
                from tasks.relational_understanding.scenarios import constant

                pred = scene["predicate"]
                value_si = quantity(obj, pred["measure"], scene)
                factor = constant(1, pred["target"]["unit"])["value_si"]
                pred["target"] = constant(round(value_si / factor, 10), pred["target"]["unit"])
            if subgroup(base["task"]) == "object_understanding":
                scene["target_rank"] = sum(
                    measurement(other, scene["case"]) < measurement(obj, scene["case"]) - 1e-12
                    for other in scene["objects"]
                )
            scene["template_id"] = template
            seal(scene)
            validate(scene)
            yield scene


def create(
    cfg, scene_count=1, all_templates=True, *, episode_count=None, candidate_counts=None, random_candidate_counts=None
):
    """Plan an exact positive episode budget, or complete sweeps of scene_count scenes.

    Explicit candidate_counts divide a budget evenly, then interleave targets.
    random_candidate_counts draws a count and a new scene seed per episode,
    followed by one eligible target/template pair. Neither mode substitutes a
    different count, gap, or geometry when generation fails.
    """
    if random_candidate_counts is not None and (candidate_counts is not None or episode_count is None):
        raise ValueError(
            "Random candidate counts require an exact episode budget and cannot be combined with balanced counts"
        )
    counts = (
        list(random_candidate_counts if random_candidate_counts is not None else candidate_counts)
        if (random_candidate_counts is not None or candidate_counts is not None)
        else [cfg.candidate_count]
    )
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("Candidate counts must be nonempty and unique")
    for count in counts:
        validate_count(count)
    if type(scene_count) is not int or scene_count < 1:
        raise ValueError("scene_count must be a positive integer")
    if episode_count is not None and (type(episode_count) is not int or episode_count < 1):
        raise ValueError("episode_count must be a positive integer")
    if cfg.scenario_file or cfg.episode_files:
        raise ValueError("Plan generation requires a task configuration, not a frozen scenario or worker queue")
    episodes, exclusions = [], []
    if random_candidate_counts is not None:
        # Draw once per episode. Count distribution never depends on whether a
        # geometry succeeds: infeasible requested scenes fail instead of being
        # replaced with an easier count or silently resized.
        rng = np.random.default_rng(int(digest(dict(seed=cfg.seed, task=cfg.task, case=cfg.case))[:16], 16))
        drawn_counts = rng.choice(counts, size=episode_count)
        for index, count in enumerate(drawn_counts):
            current = deepcopy(cfg)
            current.num_episodes = current.max_attempts = 1
            current.candidate_count = int(count)
            current.seed = cfg.seed + index
            configure(current)
            variants = list(scene_episodes(current.scenario, all_templates=all_templates))
            scene = variants[int(rng.integers(len(variants)))]
            episodes.append(dict(episode_index=index, episode_id=scene["episode_id"], scenario=scene))
        result = dict(schema_version=3, benchmark="Benchmark", episodes=episodes, infeasible_cells=[])
        validate_manifest(result)
        return result
    for count_index, count in enumerate(counts):
        quota = (
            None
            if episode_count is None
            else episode_count // len(counts) + int(count_index < episode_count % len(counts))
        )
        emitted, offset = 0, 0
        while (offset < scene_count) if quota is None else (emitted < quota):
            current = deepcopy(cfg)
            current.num_episodes = current.max_attempts = 1
            current.candidate_count = count
            current.seed = cfg.seed + offset
            configure(current)
            base = current.scenario
            if subgroup(base["task"]) == "quantitative_constraint_grounding":
                exclusions.extend(
                    {
                        "scene_id": base["scene_id"],
                        "object_id": obj["object_id"],
                        "reason": "The fixed reasoning predicate does not uniquely select this candidate",
                    }
                    for i, obj in enumerate(base["objects"])
                    if obj.get("selectable", True) and i != base["target_index"]
                )
            for scene in scene_episodes(base, all_templates=all_templates):
                if quota is not None and emitted >= quota:
                    break
                episodes.append({"episode_index": len(episodes), "episode_id": scene["episode_id"], "scenario": scene})
                emitted += 1
            offset += 1
    result = {"schema_version": 3, "benchmark": "Benchmark", "episodes": episodes, "infeasible_cells": exclusions}
    validate_manifest(result)
    return result


def generate_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/collection/balanced.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/train/object_understanding/pilot")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--scene_count", type=int, help="Complete scene sweeps per candidate count")
    size.add_argument(
        "--episodes", "--num_episodes", dest="episodes", type=int, help="Exact number of planned episodes"
    )
    count_policy = parser.add_mutually_exclusive_group()
    count_policy.add_argument(
        "--candidate_counts",
        "--candidate_count",
        nargs="+",
        type=int,
        choices=CANDIDATE_COUNTS,
        help="Balance episodes across these fixed counts",
    )
    count_policy.add_argument(
        "--random_candidate_counts",
        nargs="+",
        type=int,
        choices=CANDIDATE_COUNTS,
        help="Sample uniformly per episode; Understanding episode budgets default to 3 4 5 6 7",
    )
    parser.add_argument("--render_profile", choices=RENDER_PROFILES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=("train", "eval", "diagnostic"))
    parser.add_argument("--template_split", choices=("seen", "held_out"))
    parser.add_argument("--one_template", action="store_true")
    args = parser.parse_args()
    if args.random_candidate_counts is not None and args.episodes is None:
        parser.error("--random_candidate_counts requires --episodes")
    overrides = {
        key: getattr(args, key)
        for key in ("render_profile", "seed", "split", "template_split")
        if getattr(args, key) is not None
    }
    import yaml

    from tasks.registry import SUBGROUPS, get_task, task_id

    paths = sorted(args.config.rglob("*.yaml")) if args.config.is_dir() else [args.config]
    if not paths:
        parser.error("No task YAML configurations found")
    # The episode budget is split evenly across case files. Each subgroup is
    # written as its own plan: under --output/<subgroup> when several subgroups
    # are configured, or directly to --output for one subgroup.
    quotas, plans = {}, {}
    for index, path in enumerate(paths):
        quotas[path] = (
            None if args.episodes is None else args.episodes // len(paths) + int(index < args.episodes % len(paths))
        )
        plans.setdefault(task_id(yaml.safe_load(path.read_text())["task"]), []).append(path)
    unknown = set(plans) - set(SUBGROUPS)
    if unknown:
        parser.error(f"Unknown task subgroups: {sorted(unknown)}")
    for name in (name for name in SUBGROUPS if name in plans):
        manifest = dict(schema_version=3, benchmark="Benchmark", episodes=[], infeasible_cells=[])
        for path in plans[name]:
            quota = quotas[path]
            if quota == 0:
                continue
            definition = get_task(name)
            cfg = load_config(path, overrides, definition.CONFIG_TYPE)
            builder = create
            if SUBGROUPS[name]["engine"] == "control":
                from tasks.control.manifest import create as builder
            elif SUBGROUPS[name]["engine"] == "multi_object":
                from tasks.multi_object.manifest import create as builder
            if args.random_candidate_counts is not None and builder is not create:
                parser.error("Random candidate counts apply only to selection subgroups")
            random_counts = (
                args.random_candidate_counts or [3, 4, 5, 6, 7]
                if builder is create and args.episodes is not None and args.candidate_counts is None
                else None
            )
            part = builder(
                cfg,
                1 if args.scene_count is None else args.scene_count,
                not args.one_template,
                episode_count=quota,
                candidate_counts=args.candidate_counts,
                **({"random_candidate_counts": random_counts} if builder is create else {}),
            )
            for entry in part["episodes"]:
                entry["episode_index"] = len(manifest["episodes"])
                manifest["episodes"].append(entry)
            manifest["infeasible_cells"].extend(part["infeasible_cells"])
        validate_manifest(manifest)
        scenes = [e["scenario"] for e in manifest["episodes"]]
        report = {
            "subgroup": name,
            "planned_episodes": len(scenes),
            "by_case": dict(Counter(s["case"] for s in scenes)),
            "scenes": len({s["scene_id"] for s in scenes}),
            "by_candidate_count": dict(Counter(s["candidate_count"] for s in scenes)),
            "by_template": dict(Counter(s["template_id"] for s in scenes)),
            "by_case_candidate_count": {
                case: dict(Counter(s["candidate_count"] for s in scenes if s["case"] == case))
                for case in sorted({s["case"] for s in scenes})
            },
            "collection_status": "planned; physical and visual qualification required",
        }
        write_manifest(manifest, args.output if len(plans) == 1 else args.output / name, report)
        print(report)
