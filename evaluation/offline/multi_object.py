# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay and summarize multi-object attempts by subgroup and case from their recorded physical evidence."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
from data_collection.metadata import find_datasets, read_metadata, write_json_atomic
from data_collection.validate import validate_dataset
from tasks.registry import SUBGROUPS, subgroup_of_case


def evaluate(root):
    rows = []
    incomplete = []
    datasets = find_datasets(root)
    for path in datasets:
        with h5py.File(path, "r") as data:
            names = []
            for name, group in data["data"].items():
                meta = read_metadata(group)
                if meta.get("manipulation_protocol") != "long_horizon_v1":
                    continue
                if not group.attrs.get("finalized", False):
                    incomplete.append(
                        dict(
                            dataset=str(path),
                            episode=name,
                            episode_id=meta["episode_id"],
                            attempt_id=meta["attempt_id"],
                            case=meta["case"],
                            reason="Recording was not finalized; no verified outcome",
                        )
                    )
                else:
                    names.append(name)
        for name in names:
            validate_dataset(path, episode_name=name)
            with h5py.File(path, "r") as data:
                meta = read_metadata(data["data"][name])
            rows.append(
                dict(
                    dataset=str(path),
                    episode=name,
                    episode_id=meta["episode_id"],
                    attempt_id=meta["attempt_id"],
                    case=meta["case"],
                    success=meta["success"],
                    accepted=meta["accepted_demonstration"],
                    termination_reason=meta["termination_reason"],
                    duration_s=meta["episode_duration_s"],
                    metrics=meta["long_horizon_metrics"],
                )
            )
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"]].append(row)
    return dict(
        attempted_recordings=len(rows) + len(incomplete),
        finalized_recordings=len(rows),
        incomplete_recordings=incomplete,
        successful=sum(r["success"] for r in rows),
        accepted=sum(r["accepted"] for r in rows),
        by_subgroup={
            name: dict(attempts=len(values), successes=sum(v["success"] for v in values))
            for name in SUBGROUPS
            if (values := [r for r in rows if subgroup_of_case(r["case"]) == name])
        },
        by_case={
            case: dict(
                attempts=len(values),
                successes=sum(v["success"] for v in values),
                numerical_successes=sum(v["metrics"]["numerical_success"] for v in values),
                sequence_successes=sum(v["metrics"]["sequence_success"] for v in values),
                correct_stops=sum(v["metrics"]["correct_stopping"] for v in values),
            )
            for case, values in groups.items()
        },
        episodes=rows,
        note="Oracle attempts, not VLA accuracy. Accepted data also require an explicit observability review.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.root)
    if not result["attempted_recordings"]:
        parser.error("No multi-object recordings found")
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
