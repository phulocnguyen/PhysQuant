# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-balanced episodes, with separate numerical goals and paraphrases per reset."""

from copy import deepcopy

from data_collection.manifest import validate_manifest

from tasks.control.scenarios import TEMPLATES, InfeasibleScenario
from tasks.control.task import configure

MAX_CONSECUTIVE_INFEASIBLE_SCENES = 50


def create(cfg, scene_count=1, all_templates=True, *, episode_count=None, candidate_counts=None):
    if candidate_counts is not None:
        raise ValueError("Control has one movable; candidate_counts belongs to understanding tasks")
    if (
        type(scene_count) is not int
        or scene_count < 1
        or (episode_count is not None and (type(episode_count) is not int or episode_count < 1))
    ):
        raise ValueError("Scene and episode budgets must be positive integers")
    templates = [k for k, v in TEMPLATES["templates"].items() if v["split"] == cfg.template_split]
    if not all_templates:
        templates = [cfg.template_id or templates[0]]
    episodes = []
    exclusions = []
    scene_index = failed_scenes = 0
    while len(episodes) < episode_count if episode_count is not None else scene_index < scene_count:
        before = len(episodes)
        # Interleave numerical goals first to balance a partial language sweep.
        for template in templates:
            for goal in cfg.goal_sweep or [cfg.numerical_goal]:
                current = deepcopy(cfg)
                current.seed = cfg.seed + scene_index
                current.template_id = template
                current.numerical_goal = deepcopy(goal)
                current.num_episodes = current.max_attempts = 1
                try:
                    configure(current)
                except InfeasibleScenario as error:
                    exclusions.append(dict(seed=current.seed, case=cfg.case, numerical_goal=goal, reason=str(error)))
                    continue
                s = current.scenario
                episodes.append(dict(episode_index=len(episodes), episode_id=s["episode_id"], scenario=s))
                if episode_count is not None and len(episodes) >= episode_count:
                    break
            if episode_count is not None and len(episodes) >= episode_count:
                break
        # A scene seed whose jittered layout blocks every goal is skipped and reported in
        # infeasible_cells; many consecutive skips point to an infeasible configuration.
        failed_scenes = failed_scenes + 1 if len(episodes) == before else 0
        if failed_scenes >= MAX_CONSECUTIVE_INFEASIBLE_SCENES:
            raise InfeasibleScenario(f"No feasible control goals near seed {cfg.seed + scene_index}: {exclusions[-3:]}")
        scene_index += 1
    result = dict(schema_version=3, benchmark="Benchmark", episodes=episodes, infeasible_cells=exclusions)
    validate_manifest(result)
    return result
