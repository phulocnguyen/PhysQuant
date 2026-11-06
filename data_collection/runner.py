# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared command-line configuration, task dispatch, validation, and export."""

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import yaml
from tasks.registry import SUBGROUPS, get_task

from isaaclab.app import add_launcher_args

from data_collection.config import load_config


def main() -> None:
    """Resolve one registered task and invoke its collection interface."""
    project_root = Path(__file__).resolve().parents[1]
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/tasks/object_understanding/length.yaml",
    )
    selector.add_argument("--task")
    selector.add_argument("--list_tasks", action="store_true")
    selected, _ = selector.parse_known_args()
    if selected.list_tasks:
        print("\n".join(SUBGROUPS))
        return
    raw = yaml.safe_load(selected.config.read_text()) or {}
    if not isinstance(raw, dict):
        selector.error("YAML configuration must be a mapping")
    name = selected.task or raw.get("task")
    try:
        definition = get_task(name)
    except ValueError as error:
        selector.error(str(error))
    parser = argparse.ArgumentParser(description="Collect demonstrations for a configured task.", parents=[selector])
    parser.add_argument("--check_config", action="store_true")
    parser.add_argument("--episodes", "--num_episodes", type=int, default=None)
    parser.add_argument("--success_target", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--export_lerobot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lerobot_hz", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--capture_profile", choices=("full", "training"), default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save MP4s for each episode's recorded camera views",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    definition.add_cli_args(parser)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args = parser.parse_args()
    fields = ("seed", "success_target", "max_attempts", "capture_profile", "video", "export_lerobot", "lerobot_hz")
    overrides = {key: getattr(args, key) for key in (*fields, *definition.CLI_FIELDS) if getattr(args, key) is not None}
    overrides["task"] = name
    if args.episodes is not None:
        overrides["num_episodes"] = args.episodes
        if args.max_attempts is None:
            overrides["max_attempts"] = args.episodes
        if args.success_target is None:
            overrides["success_target"] = None
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)
    try:
        resolved = load_config(args.config, overrides, config_type=definition.CONFIG_TYPE)
        definition.configure(resolved)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    if args.check_config:
        print(yaml.safe_dump(asdict(resolved), sort_keys=False))
        return
    if resolved.video and not args.preview and shutil.which("ffmpeg") is None:
        parser.error("--video requires ffmpeg on PATH")
    for key in ("task", *fields, *definition.CLI_FIELDS):
        setattr(args, key, getattr(resolved, key))
    args.episodes = resolved.num_episodes
    args.output_dir = Path(resolved.output_dir).expanduser()
    if not args.output_dir.is_absolute():
        args.output_dir = project_root / args.output_dir
    if resolved.export_lerobot:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401

    # Kit can terminate Python while closing; runtime finalizes inside its launch context.
    definition.run(args, resolved)
