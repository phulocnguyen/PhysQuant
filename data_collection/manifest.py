# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable episode plans: reading, writing, validation, and the collection command."""

import argparse
import csv
import json
from pathlib import Path

import yaml
from tasks.registry import get_task, scenario_subgroup


def load_manifest(path):
    """Load an episode plan whose scenario files are relative to it."""
    path = Path(path)
    if path.is_dir():
        path = path / "plan.yaml"
    manifest = yaml.safe_load(path.read_text())
    if manifest.get("schema_version") != 3:
        raise ValueError(f"{path}: unsupported plan schema {manifest.get('schema_version')!r}; expected 3")
    manifest["infeasible_cells"] = yaml.safe_load((path.parent / "reports" / "excluded_cells.yaml").read_text())
    for entry in manifest["episodes"]:
        scenario = json.loads((path.parent / entry["scenario_file"]).read_text())
        if scenario["fingerprint"] != entry["fingerprint"]:
            raise ValueError("Scene fingerprint differs from the plan")
        for key in ("case", "condition", "level", "split"):
            if entry[key] != scenario[key]:
                raise ValueError(f"{key} differs from the scene")
        if entry["prompt"] != scenario["instruction"]:
            raise ValueError("Prompt differs from the scene")
        entry["scenario"] = scenario
    return manifest


def write_manifest(manifest, directory, report):
    """Separate the readable collection plan from exact generated scene records."""
    directory = Path(directory)
    if (directory / "plan.yaml").exists():
        raise FileExistsError("Use a fresh directory for an immutable episode plan")
    directory.mkdir(parents=True, exist_ok=True)
    plan = {key: value for key, value in manifest.items() if key not in ("episodes", "infeasible_cells")}
    plan["schema_version"] = 3
    plan["episodes"] = []
    for entry in manifest["episodes"]:
        scenario = entry["scenario"]
        relative = Path("scenarios") / scenario["case"] / f"episode_{entry['episode_index']:06d}.json"
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(scenario, indent=2) + "\n")
        plan["episodes"].append(
            dict(
                episode_index=entry["episode_index"],
                episode_id=entry["episode_id"],
                case=scenario["case"],
                condition=scenario["condition"],
                level=scenario["level"],
                split=scenario["split"],
                prompt=scenario["instruction"],
                scenario_file=relative.as_posix(),
                fingerprint=scenario["fingerprint"],
            )
        )
    (directory / "plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=False, allow_unicode=True))
    with (directory / "episodes.csv").open("w", newline="") as stream:
        columns = [
            "episode_index",
            "episode_id",
            "case",
            "condition",
            "level",
            "split",
            "prompt",
            "scenario_file",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plan["episodes"])
    reports = directory / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "coverage.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    excluded = manifest.get("infeasible_cells", [])
    (reports / "excluded_cells.yaml").write_text(yaml.safe_dump(excluded, sort_keys=False))


def validate_manifest(manifest):
    identities = set()
    indices = set()
    splits = {}
    for entry in manifest["episodes"]:
        s = entry["scenario"]
        info = get_task(scenario_subgroup(s)).validate_recorded_scenario(s)
        identity = entry["episode_id"]
        if identity in identities:
            raise ValueError("Duplicate planned episode")
        identities.add(identity)
        if identity != s["episode_id"]:
            raise ValueError("Episode identity differs from the scenario")
        index = entry["episode_index"]
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError("Episode indices must be unique nonnegative integers")
        indices.add(index)
        key = info["control_group_id"]
        if splits.setdefault(key, s["split"]) != s["split"]:
            raise ValueError("Matched geometry leaks across splits")


def collect_main():
    parser = argparse.ArgumentParser(
        description="Collect a Benchmark manifest with videos, saved logs, and bounded retries"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_episodes", type=int)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--attempts_per_episode", type=int, default=3)
    parser.add_argument("--mode", choices=("subprocess", "persistent"), default="persistent")
    parser.add_argument("--attempt_timeout_s", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker_episodes", type=int, default=32, help="Recycle after at most this many episodes")
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Encode review videos during collection (default: off)",
    )
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument(
        "--capture_profile",
        choices=("full", "training"),
        default="full",
        help="full records RGB and depth; training records RGB only and skips the depth render pass",
    )
    parser.add_argument(
        "--camera_hz",
        type=float,
        help="Image capture rate, for example 10; must divide the 120 Hz physics rate (default: settings.yaml, 20)",
    )
    parser.add_argument(
        "--workers_per_gpu", type=int, default=1, help="Persistent workers sharing each renderer GPU (default: 1)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Cloned environments per simulator, stepped together (default: 1). Raise --worker_episodes to fill them",
    )
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--gpu", type=int, help="RTX renderer GPU index (as listed by nvidia-smi)")
    devices.add_argument("--gpus", nargs="+", type=int, help="Run one persistent worker per listed renderer GPU")
    parser.add_argument(
        "--cpus_per_worker", type=int, help="Disjoint CPU budget per worker; parallel default: up to 8 available CPUs"
    )
    args = parser.parse_args()
    if (
        args.worker_episodes < 1
        or args.attempts_per_episode < 1
        or args.start_episode < 0
        or args.attempt_timeout_s <= 0
        or (args.gpu is not None and args.gpu < 0)
        or (args.gpus is not None and (min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus)))
        or (args.cpus_per_worker is not None and args.cpus_per_worker < 1)
        or args.workers_per_gpu < 1
        or args.num_envs < 1
        or (args.camera_hz is not None and args.camera_hz <= 0)
        or args.max_episodes is not None
        and args.max_episodes < 1
    ):
        parser.error("Invalid attempt/count/timeout limits")
    manifest = load_manifest(args.manifest)
    from environments import settings
    from tasks.common.scenarios import digest

    settings_digest = digest(settings._SHARED_SETTINGS)
    if any(entry["scenario"]["settings_digest"] != settings_digest for entry in manifest["episodes"]):
        parser.error(
            "The manifest was authored for different shared settings. "
            "Generate a new plan in a fresh directory before collection; keep existing plans immutable."
        )
    validate_manifest(manifest)
    from data_collection.coordinator import collect

    collect(manifest, args)
