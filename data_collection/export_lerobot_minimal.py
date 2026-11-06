# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert successful recordings to a minimal LeRobot dataset: top + wrist RGB, state, action, task.

Each frame holds only:
- ``observation.images.top`` and ``observation.images.wrist``: RGB, stored as video (no depth)
- ``observation.state``: joint positions (7 arm joints and 2 fingers)
- ``action``: joint-position targets, plus a stop flag for multi-object tasks
- ``task``: the episode's language instruction

Images use the latest camera frame at or before each action. Only one successful
attempt per planned episode is exported, in plan order.

Run in the LeRobot environment, for example:
    /tmp/spatial-export-env/bin/python PhysQuant/export_lerobot_minimal.py \\
        /media/Seagate16T/locnp/PhysQuant/object_understanding_7000_20261103 \\
        --output /media/Seagate16T/locnp/PhysQuant/lerobot/object_understanding_7000 --fps 10
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

VIEWS = ("top", "wrist")


def recordings(root):
    """(dataset.hdf5, group name) of each successful recording, in plan order.

    Collection outputs list each planned episode's attempts in ``episode_*/attempts.json``.
    Other folders fall back to every successful ``complete.json`` below ``root``.
    """
    index = sorted(root.glob("episode_*/attempts.json"))
    if index:
        for path in index:
            attempts = json.loads(path.read_text())
            done = next((a for a in attempts if a["successful_rollout"] and a["finalized"]), None)
            if done is not None:
                folder = Path(done["completion_file"]).parent
                yield folder / "dataset.hdf5", folder.name
        return
    for marker in sorted(root.rglob("complete.json")):
        if json.loads(marker.read_text()).get("success"):
            yield marker.parent / "dataset.hdf5", marker.parent.name


def read_episode(path, name, stride):
    """Frames for one episode, subsampled to the export rate."""
    with h5py.File(path, "r") as native:
        recording = json.loads(native.attrs["recording_json"])
        episode = native["data"][name]
        metadata = json.loads(episode.attrs["metadata_json"])
        steps = np.arange(0, len(episode["actions"]), stride)
        state = episode["obs/joint_position"][:][steps].astype(np.float32)
        action = episode["actions"][:][steps].astype(np.float32)
        if metadata.get("manipulation_protocol") == "long_horizon_v1":
            stop = episode["phase"].asstr()[:][steps] == "stop"
            action = np.concatenate([action, stop[:, None].astype(np.float32)], axis=1)
        images = {}
        for view in VIEWS:
            frames = episode[f"obs/camera_index/{view}"][:][steps]
            images[view] = episode[f"sensors/{view}/rgb"][:][frames]
    return recording, metadata["prompt"], state, action, images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Collection output folder, or any folder of recordings")
    parser.add_argument("--output", type=Path, required=True, help="New LeRobot dataset folder")
    parser.add_argument("--fps", type=int, default=10, help="Export rate; must divide the 30 Hz control rate")
    parser.add_argument("--repo_id", default="local/spatial_understanding")
    parser.add_argument("--limit", type=int, help="Export at most this many episodes, for a quick check")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a new folder")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    writer, count = None, 0
    try:
        for path, name in recordings(args.root):
            if args.limit is not None and count >= args.limit:
                break
            with h5py.File(path, "r") as native:
                control_hz = 1 / json.loads(native.attrs["recording_json"])["control_dt_s"]
            stride = control_hz / args.fps
            if stride < 1 or not np.isclose(stride, round(stride)):
                parser.error(f"--fps must divide the {control_hz:g} Hz control rate")
            recording, task, state, action, images = read_episode(path, name, round(stride))
            if writer is None:
                joints = recording["joint_names"]
                stop = ["stop_request"] if action.shape[1] > len(joints) else []
                features = {
                    f"observation.images.{view}": {
                        "dtype": "video",
                        "shape": tuple(images[view].shape[1:]),
                        "names": ["height", "width", "channels"],
                    }
                    for view in VIEWS
                }
                features["observation.state"] = {"dtype": "float32", "shape": (len(joints),), "names": joints}
                features["action"] = {"dtype": "float32", "shape": (action.shape[1],), "names": joints + stop}
                writer = LeRobotDataset.create(
                    repo_id=args.repo_id,
                    root=args.output,
                    fps=args.fps,
                    robot_type="franka_panda",
                    features=features,
                    use_videos=True,
                    video_backend="pyav",
                    image_writer_threads=4,
                )
            for i in range(len(action)):
                frame = {"task": task, "observation.state": state[i], "action": action[i]}
                frame.update({f"observation.images.{view}": images[view][i] for view in VIEWS})
                writer.add_frame(frame)
            writer.save_episode()
            count += 1
            if count % 100 == 0:
                print(f"Exported {count} episodes", flush=True)
    finally:
        if writer is not None:
            writer.finalize()
    print(f"Exported {count} episodes to {args.output}")


if __name__ == "__main__":
    main()
