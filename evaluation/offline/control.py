# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate and summarize recorded control trajectories by subgroup and case, not selection guesses."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from data_collection.metadata import find_datasets, read_metadata, write_json_atomic
from data_collection.validate import validate_dataset
from tasks.registry import SUBGROUPS, subgroup_of_case


def evaluate(root):
    datasets = find_datasets(root)
    rows = []
    incomplete = []
    for path in datasets:
        with h5py.File(path, "r") as data:
            names = []
            for name, group in data["data"].items():
                meta = read_metadata(group)
                if meta.get("manipulation_protocol") != "spatial_control_v1":
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
            for name in names:
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
                        metrics=meta["control_metrics"],
                    )
                )
    groups = defaultdict(list)
    for row in rows:
        groups[row["case"]].append(row)
    summary = {
        case: dict(
            attempts=len(values),
            successes=sum(v["success"] for v in values),
            mean_position_error_m=float(np.mean([v["metrics"]["position_error_m"] for v in values])),
            maximum_constraint_violation=max(v["metrics"]["maximum_constraint_violation"] for v in values),
        )
        for case, values in groups.items()
    }
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
        by_case=summary,
        episodes=rows,
        note=(
            "Physics-rate replay of recorded attempts. Oracle success is not VLA accuracy; "
            "retries are separate attempts."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.root)
    if not report["attempted_recordings"]:
        parser.error("No control recordings found")
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
