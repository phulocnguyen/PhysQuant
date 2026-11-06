# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export successful native episodes to LeRobot using causal image alignment.

Requires LeRobot with the create/add_frame/save_episode/finalize writer API.
No upload is performed. Depth and full ground truth remain in native HDF5.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from data_collection.calibration import camera_pose_wxyz, export_calibration
from data_collection.metadata import find_datasets, read_metadata, read_scene
from data_collection.validate import validate_dataset


def export_dataset(path: Path, output: Path, fps: int = 30, *, split: str | None = None, videos: bool = False) -> int:
    """Export accepted episodes from one native file or a collection directory.

    ``videos`` stores each camera stream as encoded video instead of image files,
    which is much smaller but lossy.

    Retries of a planned episode are deduplicated. One export must contain one
    dataset split and one robot/camera contract. Multi-object datasets append an
    explicit stop-request component to the joint-position action vector.
    """
    if type(fps) is not int or fps < 1:
        raise ValueError("LeRobot FPS must be a positive integer")
    if split not in (None, "train", "eval", "diagnostic"):
        raise ValueError("Unknown dataset split")
    path, output = Path(path), Path(output)
    paths = find_datasets(path)
    if not paths:
        raise ValueError(f"No native datasets found in {path}")
    selected, identities, contracts, splits = [], set(), set(), set()
    for source in paths:
        with h5py.File(source, "r") as native:
            for name, episode in native["data"].items():
                metadata = read_metadata(episode)
                if not episode.attrs.get("finalized", False) or not episode.attrs.get("success", False):
                    continue
                if metadata.get("benchmark") == "Benchmark" and not metadata.get("accepted_demonstration", False):
                    continue
                scene = read_scene(episode)
                if (scene.get("scenario") or {}).get("diagnostic_intervention"):
                    continue
                if split is not None and metadata.get("split") != split:
                    continue
                identity = metadata.get("episode_id", (str(source.resolve()), name))
                if identity in identities:
                    continue
                ratio = 1 / (metadata["control_dt_s"] * fps)
                if not np.isclose(ratio, round(ratio)) or ratio < 1:
                    raise ValueError("LeRobot FPS must divide the recorded control rate")
                if metadata.get("manipulation_protocol") == "long_horizon_v1":
                    if not np.any(episode["phase"].asstr()[:: round(ratio)] == "stop"):
                        raise ValueError("Export FPS drops every stop action; use a higher FPS")
                views = metadata["camera_views"]
                contracts.add(
                    (
                        tuple(metadata["joint_names"]),
                        tuple(views),
                        tuple(episode[f"sensors/{view}/rgb"].shape[1:] for view in views),
                        metadata["action_schema"],
                        metadata["control_dt_s"],
                    )
                )
                splits.add(metadata.get("split"))
                selected.append((source, name, metadata, scene, round(ratio)))
                identities.add(identity)
    if len(splits) > 1:
        raise ValueError("Export train/eval/diagnostic splits separately; select --split")
    if len(contracts) > 1:
        raise ValueError("Export requires consistent joint order, camera shapes and action/control contracts")
    if not selected:
        return 0
    # Validate everything before creating a partial training dataset.
    for source, name, _, _, _ in selected:
        validate_dataset(source, name)
    include_stop = any(m.get("manipulation_protocol") == "long_horizon_v1" for _, _, m, _, _ in selected)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    writer = None
    sources = []
    try:
        for source, name, metadata, scene, stride in selected:
            with h5py.File(source, "r") as native:
                episode = native["data"][name]
                views = metadata["camera_views"]
                if writer is None:
                    joints = episode["actions"].shape[1]
                    features = {
                        "observation.state": {"dtype": "float32", "shape": (2 * joints,), "names": None},
                        "action": {
                            "dtype": "float32",
                            "shape": (joints + int(include_stop),),
                            "names": metadata["joint_names"] + (["stop_request"] if include_stop else []),
                        },
                    }
                    for view in views:
                        shape = episode[f"sensors/{view}/rgb"].shape[1:]
                        features[f"observation.images.{view}"] = {
                            "dtype": "video" if videos else "image",
                            "shape": tuple(shape),
                            "names": ["height", "width", "channels"],
                        }
                        features[f"observation.intrinsics.{view}"] = {"dtype": "float32", "shape": (9,), "names": None}
                        features[f"observation.camera_pose.{view}"] = {
                            "dtype": "float32",
                            "shape": (7,),
                            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
                        }
                        features[f"observation.camera_timestamp.{view}"] = {
                            "dtype": "float32",
                            "shape": (1,),
                            "names": None,
                        }
                    writer = LeRobotDataset.create(
                        repo_id="local/Benchmark",
                        root=output,
                        fps=fps,
                        robot_type="franka_panda",
                        features=features,
                        use_videos=videos,
                        image_writer_threads=4,
                        **({"video_backend": "pyav"} if videos else {}),
                    )
                phases = episode["phase"].asstr()[:] if include_stop else None
                for i in range(0, len(episode["actions"]), stride):
                    action = episode["actions"][i].astype(np.float32)
                    if include_stop:
                        action = np.append(action, np.float32(phases[i] == "stop"))
                    frame = {
                        "task": metadata["prompt"],
                        "observation.state": np.concatenate(
                            (episode["obs/joint_position"][i], episode["obs/joint_velocity"][i])
                        ).astype(np.float32),
                        "action": action,
                    }
                    for view in views:
                        sensor = episode[f"sensors/{view}"]
                        index = int(episode[f"obs/camera_index/{view}"][i])
                        frame[f"observation.images.{view}"] = sensor["rgb"][index]
                        frame[f"observation.intrinsics.{view}"] = (
                            sensor["intrinsics"][index].reshape(-1).astype(np.float32)
                        )
                        frame[f"observation.camera_pose.{view}"] = camera_pose_wxyz(
                            sensor["position"][index], sensor["quaternion_ros_xyzw"][index]
                        )
                        frame[f"observation.camera_timestamp.{view}"] = np.array(
                            [sensor["timestamps"][index]], np.float32
                        )
                    writer.add_frame(frame)
                writer.save_episode()
                sources.append(
                    {
                        "source_dataset": str(source.resolve()),
                        "source_episode": name,
                        "metadata": metadata,
                        "scene": scene,
                    }
                )
    finally:
        if writer is not None:
            writer.finalize()
    if writer is not None:
        (output / "ground_truth.json").write_text(json.dumps(sources, indent=2) + "\n")
        (output / "meta/action_contract.json").write_text(
            json.dumps(
                {
                    "action_schema": selected[0][2]["action_schema"],
                    "joint_names": selected[0][2]["joint_names"],
                    "stop_request": "last component; 0=continue, 1=request stop" if include_stop else None,
                    "state_schema": "joint positions followed by joint velocities in joint_names order",
                    "split": next(iter(splits)),
                    "source_control_dt_s": selected[0][2]["control_dt_s"],
                    "export_fps": fps,
                    "image_alignment": "latest recorded frame at or before each action; images may repeat",
                    "camera_pose_schema": "world XYZ metres; WXYZ; camera axes +X forward, +Z up",
                    "privileged_data": "ground_truth.json is evaluation metadata, never a policy input",
                },
                indent=2,
            )
            + "\n"
        )
        if len(paths) == 1:
            export_calibration(paths[0], output, [item["source_episode"] for item in sources])
        else:
            for index, (source, name, _, _, _) in enumerate(selected):
                export_calibration(source, output / "native_calibration" / f"episode_{index:06d}", [name])
    return len(sources)


def main() -> None:
    """Run the dataset tool from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--split", choices=("train", "eval", "diagnostic"))
    parser.add_argument("--videos", action="store_true", help="Store camera streams as encoded video (smaller, lossy)")
    parser.add_argument(
        "--calibration_only",
        action="store_true",
        help="Write reference-format calibration and poses without simulation or LeRobot",
    )
    args = parser.parse_args()
    if args.calibration_only:
        output = args.output or args.dataset.parent
        count = export_calibration(args.dataset, output)
        print(f"Exported calibration for {count} episodes to {output / 'meta'}")
        return
    if args.output is None:
        parser.error("--output is required for LeRobot export")
    count = export_dataset(args.dataset, args.output, args.fps, split=args.split, videos=args.videos)
    print(f"Exported {count} accepted episodes")


if __name__ == "__main__":
    main()
