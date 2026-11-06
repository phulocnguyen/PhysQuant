# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared versioned configuration and simulator entry point for all spatial-understanding families."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from core.profiles import CANDIDATE_COUNTS, RENDER_PROFILES, render_settings
from data_collection.config import CollectionConfig
from environments import settings
from environments.randomization import resolve_randomization

from tasks.common.picking import PROTOCOL
from tasks.common.scenarios import LAUNCH_MISMATCH, digest, launch_key, validate
from tasks.object_understanding.scenarios import ATTRIBUTES
from tasks.registry import SELECTION_SUBGROUPS, scenario_subgroup, subgroup_module, task_id


@dataclass
class BenchmarkConfig(CollectionConfig):
    task: str = "object_understanding"
    case: str = "length"
    operator: str = "gt"
    candidate_count: int = 3
    value: float = 12
    unit: str = "cm"
    gap: float = 2
    tolerance: float = 0.5
    interval_bounds: list[float] | None = None
    interval_closed: list[bool] = field(default_factory=lambda: [True, True])
    target_rank: int | None = None
    shapes: list[str] = field(default_factory=lambda: ["box"])
    split: str = "diagnostic"
    template_split: str = "seen"
    template_id: str | None = None
    scenario_file: str | None = None
    episode_files: list[str] = field(default_factory=list)
    attempt_id: int = 0
    randomization_profile: str | None = None
    render_profile: str = "quality"
    compression: str = "lzf"
    isolated_episodes: bool = True
    num_episodes: int = 1
    max_attempts: int = 1
    # Empty means datasets/<subgroup>.
    output_dir: str = ""
    scenario: dict = field(default_factory=dict, init=False)
    scenarios: list[dict] = field(default_factory=list, init=False)


CONFIG_TYPE = BenchmarkConfig
CLI_FIELDS = (
    "case",
    "operator",
    "candidate_count",
    "value",
    "unit",
    "gap",
    "tolerance",
    "target_rank",
    "split",
    "template_split",
    "template_id",
    "scenario_file",
    "render_profile",
    "compression",
)


def add_cli_args(parser):
    from tasks.quantitative_constraint_grounding.scenarios import CASES as QCG_CASES
    from tasks.relational_understanding.scenarios import CASES as RU_CASES

    parser.add_argument("--case", choices=(*ATTRIBUTES, *RU_CASES, *QCG_CASES))
    parser.add_argument("--operator", choices=("gt", "ge", "lt", "le"))
    parser.add_argument("--candidate_count", type=int, choices=CANDIDATE_COUNTS)
    parser.add_argument("--value", type=float)
    parser.add_argument("--unit")
    parser.add_argument("--gap", type=float)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--target_rank", type=int)
    parser.add_argument("--split", choices=("train", "eval", "diagnostic"))
    parser.add_argument("--template_split", choices=("seen", "held_out"))
    parser.add_argument("--template_id")
    parser.add_argument("--scenario_file")
    parser.add_argument("--render_profile", choices=RENDER_PROFILES)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"))


def resolve_settings(cfg):
    import yaml

    from tasks.common.scenarios import ROOT

    if cfg.task_settings:
        raise ValueError(
            "Versioned tasks use shared settings and manipulation protocols; task_settings overrides are forbidden"
        )
    resolved = SimpleNamespace(
        **{k: deepcopy(v) for k, v in vars(settings).items() if k.isupper() and not k.startswith("_")}
    )
    protocol = cfg.scenario.get("goal", PROTOCOL)
    resolved.TIMEOUT_SECONDS = protocol["timeout_s"]
    resolved.MAX_DISTRACTOR_MOVEMENT = protocol["max_distractor_movement_m"]
    resolved.OBJECT_START_ASLEEP = False
    resolved.RENDER_SETTINGS = deepcopy(cfg.scenario.get("rendering", render_settings(cfg.render_profile)))
    resolved.PLACEMENT_CONTROL = yaml.safe_load((ROOT / "configs/controllers/pick_place.yaml").read_text())
    settings.validate_settings(resolved)
    return resolved


def configure(cfg):
    cfg.task = task_id(cfg.task)
    if cfg.task not in SELECTION_SUBGROUPS:
        raise ValueError(f"{cfg.task!r} is not a selection subgroup")
    cfg.output_dir = cfg.output_dir or f"datasets/native_hdf5/{cfg.task}"
    render_settings(cfg.render_profile)
    if cfg.compression not in ("lzf", "gzip", "none") or type(cfg.isolated_episodes) is not bool:
        raise ValueError("Invalid lossless compression or episode isolation setting")
    if cfg.max_attempts < cfg.num_episodes:
        raise ValueError("max_attempts must cover num_episodes; use a manifest for distinct episode collection")
    if cfg.task_prompt is not None or cfg.success_target is not None:
        raise ValueError("Versioned episodes have one generated instruction and an explicit attempt count")
    if not cfg.shapes:
        raise ValueError("At least one object shape is required")
    import yaml

    from tasks.common.scenarios import ROOT

    cfg.randomization = resolve_randomization(
        yaml.safe_load((ROOT / cfg.randomization_profile).read_text())
        if cfg.randomization_profile
        else {"enabled": False}
    )
    group = cfg.randomization["target_object"]
    if (
        cfg.task != "object_understanding"
        and cfg.randomization["enabled"]
        and group["enabled"]
        and (group["position_range_m"] or group["yaw_range_deg"])
    ):
        raise ValueError(
            "Relational Understanding and Quantitative Constraint Grounding positions and orientations are"
            " solved by the predicate; independent pose randomization is forbidden"
        )
    cfg.scenario = json.loads(Path(cfg.scenario_file).read_text()) if cfg.scenario_file else _generate(cfg)
    validate(cfg.scenario)
    if scenario_subgroup(cfg.scenario) != cfg.task:
        raise ValueError("Configured task subgroup differs from the scenario")
    if cfg.scenario["settings_digest"] != digest(settings._SHARED_SETTINGS):
        raise ValueError("The scenario was authored for different shared settings")
    resolve_settings(cfg)
    cfg.randomization = resolve_randomization(cfg.scenario["randomization"])
    cfg.scenarios = [json.loads(Path(path).read_text()) for path in cfg.episode_files]
    for scenario in cfg.scenarios:
        validate(scenario)
        if launch_key(scenario) != launch_key(cfg.scenario):
            raise ValueError(LAUNCH_MISMATCH)
    if cfg.scenarios and len(cfg.scenarios) != cfg.num_episodes:
        raise ValueError("Episode queue length differs from num_episodes")
    if isinstance(cfg.attempt_id, bool) or not isinstance(cfg.attempt_id, int) or cfg.attempt_id < 0:
        raise ValueError("attempt_id must be a nonnegative integer")


def run(args, cfg):
    from data_collection.simulation import run as collect

    s = cfg.scenario
    return collect(
        args,
        cfg,
        config=resolve_settings(cfg),
        specs=s["objects"],
        target=s["target_index"],
        prompt=s["instruction"],
        scenario=s,
        episode_scenarios=cfg.scenarios,
    )


def validate_recorded_scenario(scenario):
    return validate(scenario)


def collection_settings(scenario):
    return {
        "task": scenario_subgroup(scenario),
        "case": scenario["case"],
        "seed": scenario["seed"],
        "split": scenario["split"],
        "candidate_count": scenario["candidate_count"],
        "template_id": scenario["template_id"],
        "template_split": scenario["template_split"],
        "render_profile": scenario.get("render_profile", "quality"),
    }


def _generate(cfg):
    """Generate a scene with the configured subgroup's ``scenarios.generate``."""
    return subgroup_module(cfg.task, "scenarios").generate(cfg)
