# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-object task adapter for the common persistent collector."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from data_collection.config import CollectionConfig
from environments import settings
from environments.randomization import resolve_randomization

from tasks.common.scenarios import LAUNCH_MISMATCH, ROOT, digest, launch_key
from tasks.common.task import resolve_settings as shared_settings
from tasks.registry import scenario_subgroup, task_id

from .scenarios import CASES, generate, validate


@dataclass
class MultiObjectConfig(CollectionConfig):
    task: str = "sequencing_allocation_planning"
    case: str = "SAP-1A"
    problems: list[dict] = field(default_factory=list)
    problem_index: int | None = None
    goal_sweep: list[dict] = field(default_factory=list)
    diagnostic_intervention: dict | None = None
    contact_offset_m: float = 0.001
    initial_positions_xy_m: list = field(
        default_factory=lambda: [[-0.26, -0.09], [0.0, -0.09], [0.26, -0.09], [-0.34, -0.20], [0.34, -0.20]]
    )
    position_jitter_m: float = 0.005
    initial_yaw_deg: list[float] = field(default_factory=lambda: [0.0])
    basket_asset: str = "configs/assets/basket.yaml"
    basket_jitter_m: float = 0.003
    basket_yaw_deg: list[float] = field(default_factory=lambda: [-3.0, 0.0, 3.0])
    basket_color_options: list = field(
        default_factory=lambda: [[0.08, 0.16, 0.22], [0.18, 0.08, 0.03], [0.12, 0.12, 0.12]]
    )
    reference_xy_m: list[float] = field(default_factory=lambda: [-0.25, 0.20])
    construction_xy_m: list = field(default_factory=lambda: [[-0.15, 0.20], [0.15, 0.20]])
    randomization_profile: str | None = None
    split: str = "diagnostic"
    template_split: str = "seen"
    template_id: str | None = None
    scenario_file: str | None = None
    episode_files: list[str] = field(default_factory=list)
    attempt_id: int = 0
    render_profile: str = "balanced"
    compression: str = "lzf"
    isolated_episodes: bool = True
    num_episodes: int = 1
    max_attempts: int = 1
    # Empty means datasets/<subgroup>.
    output_dir: str = ""
    scenario: dict = field(default_factory=dict, init=False)
    scenarios: list = field(default_factory=list, init=False)


CONFIG_TYPE = MultiObjectConfig
CLI_FIELDS = ("case", "split", "template_split", "template_id", "scenario_file", "render_profile", "compression")


def add_cli_args(parser):
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--split", choices=("train", "eval", "diagnostic"))
    parser.add_argument("--template_split", choices=("seen", "held_out"))
    for name in ("template_id", "scenario_file", "render_profile", "compression"):
        parser.add_argument("--" + name)


def resolve_settings(cfg):
    result = shared_settings(cfg)
    result.TIMEOUT_SECONDS = cfg.scenario["goal"]["timeout_s"]
    return result


def configure(cfg):
    if cfg.task_settings or cfg.task_prompt is not None or cfg.success_target is not None:
        raise ValueError("Multi-object uses frozen task instructions and shared settings")
    cfg.task = task_id(cfg.task)
    cfg.output_dir = cfg.output_dir or f"datasets/native_hdf5/{cfg.task}"
    cfg.randomization = resolve_randomization(
        yaml.safe_load((ROOT / cfg.randomization_profile).read_text())
        if cfg.randomization_profile
        else {"enabled": False}
    )
    if cfg.randomization["enabled"] and cfg.randomization["target_object"]["enabled"]:
        raise ValueError("Independent pose/physics randomization would invalidate the construction plan")
    cfg.scenario = json.loads(Path(cfg.scenario_file).read_text()) if cfg.scenario_file else generate(cfg)
    validate(cfg.scenario)
    if (
        cfg.task != scenario_subgroup(cfg.scenario)
        or cfg.case != cfg.scenario["case"]
        or cfg.scenario["settings_digest"] != digest(settings._SHARED_SETTINGS)
    ):
        raise ValueError("Task/settings mismatch; regenerate the plan")
    cfg.randomization = deepcopy(cfg.scenario["randomization"])
    cfg.scenarios = [json.loads(Path(p).read_text()) for p in cfg.episode_files]
    for s in cfg.scenarios:
        validate(s)
        if launch_key(s) != launch_key(cfg.scenario):
            raise ValueError(LAUNCH_MISMATCH)
    if cfg.scenarios and len(cfg.scenarios) != cfg.num_episodes:
        raise ValueError("Episode queue length mismatch")
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


def validate_recorded_scenario(s):
    return validate(s)
