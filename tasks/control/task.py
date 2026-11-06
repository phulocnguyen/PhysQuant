# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Control engine adapter for Spatial Transformation, Target Placement, and Constrained Motion."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from core.profiles import RENDER_PROFILES
from data_collection.config import CollectionConfig
from environments import settings
from environments.randomization import resolve_randomization

from tasks.common.scenarios import LAUNCH_MISMATCH, digest, launch_key
from tasks.common.task import resolve_settings as shared_settings
from tasks.control.goals import CASES
from tasks.control.scenarios import generate, validate
from tasks.registry import scenario_subgroup, task_id


@dataclass
class ControlConfig(CollectionConfig):
    task: str = "spatial_transformation"
    case: str = "ST-1A"
    geometry_options: list[dict] = field(default_factory=lambda: [{"kind": "box", "size_m": [0.10, 0.045, 0.05]}])
    initial_xy_m: list[float] = field(default_factory=lambda: [-0.12, 0.08])
    initial_jitter_xy_m: list[float] = field(default_factory=lambda: [0.015, 0.015])
    initial_yaw_deg: list[float] = field(default_factory=lambda: [0.0])
    references: dict = field(default_factory=dict)
    randomization_profile: str | None = None
    numerical_goal: dict = field(default_factory=lambda: {"displacement_cm": [10, 0, 0]})
    goal_sweep: list[dict] = field(default_factory=list)
    split: str = "diagnostic"
    template_split: str = "seen"
    template_id: str | None = None
    scenario_file: str | None = None
    episode_files: list[str] = field(default_factory=list)
    attempt_id: int = 0
    render_profile: str = "quality"
    compression: str = "lzf"
    isolated_episodes: bool = True
    num_episodes: int = 1
    max_attempts: int = 1
    # Empty means datasets/<subgroup>.
    output_dir: str = ""
    scenario: dict = field(default_factory=dict, init=False)
    scenarios: list[dict] = field(default_factory=list, init=False)


CONFIG_TYPE = ControlConfig
CLI_FIELDS = ("case", "split", "template_split", "template_id", "scenario_file", "render_profile", "compression")


def add_cli_args(parser):
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--split", choices=("train", "eval", "diagnostic"))
    parser.add_argument("--template_split", choices=("seen", "held_out"))
    parser.add_argument("--template_id")
    parser.add_argument("--scenario_file")
    parser.add_argument("--render_profile", choices=RENDER_PROFILES)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"))


def resolve_settings(cfg):
    result = shared_settings(cfg)
    result.TIMEOUT_SECONDS = cfg.scenario["goal"]["timeout_s"]
    return result


def configure(cfg):
    if cfg.task_prompt is not None or cfg.success_target is not None or cfg.task_settings:
        raise ValueError("Control episodes use generated instructions and versioned settings")
    if (
        cfg.compression not in ("lzf", "gzip", "none")
        or cfg.max_attempts < cfg.num_episodes
        or type(cfg.isolated_episodes) is not bool
    ):
        raise ValueError("Invalid compression/attempt count/isolation")
    cfg.task = task_id(cfg.task)
    cfg.output_dir = cfg.output_dir or f"datasets/native_hdf5/{cfg.task}"
    if type(cfg.attempt_id) is not int or cfg.attempt_id < 0:
        raise ValueError("attempt_id must be a nonnegative integer")
    # Goal-constrained geometry/pose sampling belongs to the generator. Baseline
    # physics, camera, appearance and robot remain fixed for matched goal sweeps.
    import yaml

    from tasks.common.scenarios import ROOT

    cfg.randomization = resolve_randomization(
        yaml.safe_load((ROOT / cfg.randomization_profile).read_text())
        if cfg.randomization_profile
        else {"enabled": False}
    )
    if cfg.randomization["enabled"] and cfg.randomization["target_object"]["enabled"]:
        raise ValueError(
            "Control object pose/geometry are solved at generation; independent object randomization is forbidden"
        )
    cfg.scenario = json.loads(Path(cfg.scenario_file).read_text()) if cfg.scenario_file else generate(cfg)
    validate(cfg.scenario)
    if cfg.task != scenario_subgroup(cfg.scenario) or cfg.case != cfg.scenario["case"]:
        raise ValueError("Configured task/case differs from frozen scenario")
    if cfg.scenario["settings_digest"] != digest(settings._SHARED_SETTINGS):
        raise ValueError("Scenario uses different shared settings; generate a new plan")
    cfg.randomization = deepcopy(cfg.scenario["randomization"])
    cfg.scenarios = [json.loads(Path(p).read_text()) for p in cfg.episode_files]
    for s in cfg.scenarios:
        validate(s)
        if launch_key(s) != launch_key(cfg.scenario):
            raise ValueError(LAUNCH_MISMATCH)
    if cfg.scenarios and len(cfg.scenarios) != cfg.num_episodes:
        raise ValueError("Queue length differs from requested episode count")
    resolve_settings(cfg)


def run(args, cfg):
    from data_collection.simulation import run as collect

    s = cfg.scenario
    return collect(
        args,
        cfg,
        config=resolve_settings(cfg),
        specs=s["objects"],
        target=0,
        prompt=s["instruction"],
        scenario=s,
        episode_scenarios=cfg.scenarios,
    )


def validate_recorded_scenario(scenario):
    return validate(scenario)


def collection_settings(s):
    return dict(
        task=scenario_subgroup(s),
        case=s["case"],
        seed=s["seed"],
        split=s["split"],
        template_id=s["template_id"],
        template_split=s["template_split"],
        render_profile=s["render_profile"],
    )
