# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Review real camera streams and separate collection outcomes from VLA accuracy."""

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from data_collection.metadata import episode_directory, find_episode_metadata, write_json_atomic


def accepted(meta):
    base = bool(meta.get("finalized") and meta.get("success") and meta.get("observability_status") == "verified")
    gates = meta.get("validation_gates", {})
    return (
        base
        and meta.get("manipulation_protocol")
        in ("pick_place_v1", "pick_hold_v1", "spatial_control_v1", "long_horizon_v1")
        and all(
            gates.get(key) == "passed" for key in ("numerical", "spatial", "physical", "observability", "integrity")
        )
    )


def statistics(root):
    counts = Counter(
        dict(
            attempted=0,
            successful_manipulations=0,
            failed_attempts=0,
            incomplete_or_running=0,
            accepted_demonstrations=0,
        )
    )
    cases = Counter()
    conditions = Counter()
    levels = Counter()
    for path in find_episode_metadata(root):
        meta = json.loads(path.read_text())
        if meta.get("benchmark") != "Benchmark":
            continue
        scenario = meta
        counts["attempted"] += 1
        counts["successful_manipulations"] += int(meta["success"])
        counts["failed_attempts"] += int(meta.get("finalized", False) and not meta["success"])
        counts["incomplete_or_running"] += int(not meta.get("finalized", False))
        counts["accepted_demonstrations"] += int(accepted(meta))
        cases[scenario["case"]] += 1
        conditions[scenario["condition"]] += 1
        levels[scenario["level"]] += 1
    return dict(
        counts=dict(counts),
        by_case=dict(cases),
        by_condition=dict(conditions),
        by_level=dict(levels),
        identification_accuracy=None,
        note=(
            "Oracle manipulation collection does not measure VLA identification accuracy; "
            "use evaluate.py with policy predictions"
        ),
    )


def contact_sheet(dataset, episode, output):
    from PIL import Image, ImageDraw

    with h5py.File(dataset, "r") as file:
        group = file[f"data/{episode}"]
        views = list(group["sensors"])
        sheet = Image.new("RGB", (768, 280 * len(views)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, view in enumerate(views):
            stream = group[f"sensors/{view}"]
            n = len(stream["rgb"])
            if not n:
                raise ValueError("No recorded frames")
            for col, index in enumerate(np.linspace(0, n - 1, 3, dtype=int)):
                sheet.paste(Image.fromarray(stream["rgb"][index]).resize((256, 256)), (col * 256, row * 280 + 24))
                draw.text((col * 256 + 4, row * 280 + 4), f"{view} t={stream['timestamps'][index]:.2f}s", fill="black")
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--episode", default="episode_0000")
    parser.add_argument("--contact_sheet", type=Path)
    parser.add_argument("--visibility", choices=["verified", "rejected"])
    parser.add_argument("--reason")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.visibility:
        if not args.dataset or not args.reason:
            parser.error("Visibility review requires --dataset and --reason")
        with h5py.File(args.dataset, "r+") as file:
            group = file[f"data/{args.episode}"]
            meta = json.loads(group.attrs["metadata_json"])
            if meta.get("benchmark") != "Benchmark":
                raise ValueError("Not a Benchmark recording; do not relabel it")
            meta["observability_status"] = args.visibility
            meta["validation_gates"]["observability"] = "passed" if args.visibility == "verified" else "failed"
            meta["visibility_review_reason"] = args.reason
            meta["accepted_demonstration"] = accepted(meta)
            group.attrs["metadata_json"] = json.dumps(meta)
            write_json_atomic(episode_directory(args.dataset, args.episode) / "metadata.json", meta)
    if args.contact_sheet:
        if not args.dataset:
            parser.error("--contact_sheet requires --dataset")
        contact_sheet(args.dataset, args.episode, args.contact_sheet)
    report = statistics(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
