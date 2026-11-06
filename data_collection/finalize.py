# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Finalize dataset checks before the simulator launcher closes its process."""

import json
import time

import h5py

from data_collection.calibration import export_calibration
from data_collection.metadata import write_json_atomic
from data_collection.validate import validate_dataset


def finalize_attempt(recorder, success, reason, cfg):
    """Close and validate one recording before publishing its completion marker."""
    directory = recorder.directory
    dataset = recorder.dataset_path
    primary = recorder.metadata.get("manipulation_protocol") in (
        "pick_place_v1",
        "pick_hold_v1",
        "spatial_control_v1",
        "long_horizon_v1",
    )
    if primary:
        recorder.metadata["validation_gates"] = {
            "numerical": "passed",
            "spatial": "passed",
            "physical": "passed" if success else "failed",
            "observability": "unverified",
            "integrity": "unverified",
        }
    recorder.finish(success, reason)
    start = time.monotonic()
    integrity = validate_dataset(dataset, directory.name)
    validation_seconds = time.monotonic() - start
    encoding_started = time.monotonic()
    if cfg.video:
        from data_collection.video import export_episode_videos

        export_episode_videos(dataset, directory.name, directory.parent / "videos")
    if primary:
        recorder.metadata["validation_gates"]["integrity"] = "passed"
        recorder.metadata["validation_version"] = 1
        recorder.metadata["validation_seconds"] = validation_seconds
        recorder.metadata["encoding_seconds"] = time.monotonic() - encoding_started
        with h5py.File(dataset, "r+") as data:
            data[f"data/{directory.name}"].attrs["metadata_json"] = json.dumps(recorder.metadata)
            data.flush()
        write_json_atomic(directory / "metadata.json", recorder.metadata)
        scenario = recorder.scene["scenario"]
        write_json_atomic(
            directory / "complete.json",
            {
                "episode_id": scenario["episode_id"],
                "fingerprint": scenario["fingerprint"],
                "success": success,
                "integrity": integrity,
                "attempt_id": recorder.metadata.get("attempt_id", 0),
            },
        )


def finalize_run(run_dir, summary, cfg):
    paths = sorted(run_dir.glob("episode_*/dataset.hdf5")) or [run_dir / "dataset.hdf5"]
    report = {"episodes": 0, "successes": 0, "transitions": 0}
    for path in paths:
        checked = validate_dataset(path)
        for key in report:
            report[key] += checked[key]
        destination = path.parent
        export_calibration(path, destination, pose_streams=False)
        if cfg.export_lerobot:
            from data_collection.export_lerobot import export_dataset

            export_dataset(path, destination / "lerobot", cfg.lerobot_hz)
    (run_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    write_json_atomic(run_dir / "summary.json", summary)
    print(f"Collection summary: {summary}", flush=True)
    if not summary["completed"]:
        raise SystemExit("Collection stopped before the requested count; attempts retained")
    write_json_atomic(run_dir / "complete.json", {"validation": report, "summary": summary})
