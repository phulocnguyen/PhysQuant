# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent, reproducible Multi-object episodes and template sweeps."""

from copy import deepcopy

from data_collection.manifest import validate_manifest

from .planning import solve
from .scenarios import TEMPLATES, seal, validate
from .task import configure


def create(cfg, scene_count=1, all_templates=True, *, episode_count=None, candidate_counts=None):
    if candidate_counts is not None:
        raise ValueError("Multi-object object counts belong to solution-first problem definitions")
    templates = [k for k, v in TEMPLATES["templates"].items() if v["split"] == cfg.template_split]
    if not all_templates:
        templates = [cfg.template_id or templates[0]]
    if scene_count < 1 or episode_count is not None and episode_count < 1:
        raise ValueError("Episode count must be positive")
    episodes = []
    index = 0
    while len(episodes) < episode_count if episode_count is not None else index < scene_count:
        for template in templates:
            current = deepcopy(cfg)
            current.seed = cfg.seed + index
            if cfg.problem_index is None and cfg.problems:
                current.problem_index = index % len(cfg.problems)
            current.template_id = template
            current.num_episodes = current.max_attempts = 1
            configure(current)
            for goal in cfg.goal_sweep or [current.scenario["numerical_goal"]]:
                s = deepcopy(current.scenario)
                s["numerical_goal"] = deepcopy(goal)
                s["plan"] = solve(s["case"], s["objects"], goal)
                seal(s)
                validate(s)
                episodes.append(dict(episode_index=len(episodes), episode_id=s["episode_id"], scenario=s))
                if episode_count is not None and len(episodes) >= episode_count:
                    break
            if episode_count is not None and len(episodes) >= episode_count:
                break
        index += 1
    result = dict(schema_version=3, benchmark="Benchmark", episodes=episodes, infeasible_cells=[])
    validate_manifest(result)
    return result
