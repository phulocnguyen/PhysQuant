# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export camera calibration in the Extrinsic Dexterity recording conventions."""

import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from data_collection.metadata import read_metadata, read_settings


def camera_quaternion_wxyz(ros_xyzw, convention: str = "world") -> np.ndarray:
    """Convert ROS camera-to-world rotations to WXYZ in the requested camera axes.

    Camera axes are ROS (+Z forward, -Y up), world (+X forward, +Z up),
    or OpenGL (-Z forward, +Y up). The reference position remains in world space.
    """
    offsets = {
        "ros": (0.0, 0.0, 0.0, 1.0),
        "opengl": (1.0, 0.0, 0.0, 0.0),
        "world": (0.5, -0.5, 0.5, 0.5),
    }
    q = np.asarray(ros_xyzw, dtype=np.float64)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all():
        raise ValueError("Camera quaternion must contain four finite XYZW components")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Camera quaternion cannot have zero norm")
    q = q / norm
    offset = np.asarray(offsets[convention])
    # Postmultiply: change the local camera basis, not its world reference frame.
    xyz = q[..., 3:] * offset[:3] + offset[3] * q[..., :3] + np.cross(q[..., :3], offset[:3])
    w = q[..., 3:] * offset[3] - np.sum(q[..., :3] * offset[:3], axis=-1, keepdims=True)
    result = np.concatenate((w, xyz), axis=-1)
    return np.where(result[..., :1] < 0, -result, result)


def camera_pose_wxyz(position, ros_xyzw) -> np.ndarray:
    """Return reference-format XYZ + WXYZ, with +X forward and +Z up camera axes."""
    return np.concatenate((position, camera_quaternion_wxyz(ros_xyzw)), axis=-1).astype(np.float32)


def export_calibration(path: Path, output: Path, episode_names=None, *, pose_streams: bool = True) -> int:
    """Write calibration JSON and causally aligned reference-format pose streams.

    Reads saved settings, never the current scene config. Native images, depth,
    and ROS/XYZW poses are untouched. No simulator or LeRobot dependency is needed.
    With pose_streams, a companion HDF5 uses the reference recorder's
    obs/camera_pose_<sensor> names. Collection needs only the JSON snapshot.
    """
    path, output = Path(path), Path(output)
    calibrations = {}
    meta = output / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    temporary = meta / "camera_poses.hdf5.tmp"
    try:
        with ExitStack() as stack:
            source = stack.enter_context(h5py.File(path, "r"))
            settings = read_settings(source)
            if pose_streams:
                poses = stack.enter_context(h5py.File(temporary, "w"))
                poses.attrs["pose_schema"] = "world XYZ metres; WXYZ; camera axes +X forward, +Z up"
                poses.attrs["source_dataset"] = str(path.resolve())
                data = poses.create_group("data")
            names = list(source["data"]) if episode_names is None else episode_names
            for name in names:
                episode = source[f"data/{name}"]
                if not episode.attrs.get("finalized", False):
                    raise ValueError(f"{name}: finalize the recording before converting calibration")
                if not episode.attrs["num_samples"]:
                    continue
                metadata = read_metadata(episode)
                calibration = {
                    "note": (
                        "Snapshot at the first recorded camera frame of this episode. "
                        "Per-frame intrinsics and ROS/XYZW poses are in dataset.hdf5 sensors/<view>; "
                        "obs/camera_index aligns them to actions. "
                        + ("Reference WXYZ poses are also in camera_poses.hdf5. " if pose_streams else "")
                        + "Optical dimensions are spawn settings; intrinsic_matrix is measured."
                    ),
                    "rendering": settings.get("RENDER_SETTINGS", {"note": "Not recorded in this run"}),
                    "cameras": {},
                }
                if pose_streams:
                    group = data.create_group(name)
                    group.create_dataset("timestamps", data=episode["timestamps"][:])
                    obs = group.create_group("obs")
                for view in metadata["camera_views"]:
                    sensor = episode[f"sensors/{view}"]
                    indices = episode[f"obs/camera_index/{view}"][:]
                    position = sensor["position"][:]
                    rotation = sensor["quaternion_ros_xyzw"][:]
                    first = int(indices[0])
                    height, width = sensor["rgb"].shape[1:3]
                    mount = settings["CAMERAS"][view]
                    mounted = "body" in mount
                    prefix = "WRIST_CAMERA" if mounted else "CAMERA"
                    sensor_name = f"camera_{view}"
                    calibration["cameras"][view] = {
                        "sensor_name": sensor_name,
                        "width": int(width),
                        "height": int(height),
                        "intrinsic_matrix": sensor["intrinsics"][first].tolist(),
                        "focal_length_mm": settings[f"{prefix}_FOCAL_LENGTH"],
                        "horizontal_aperture_mm": settings[f"{prefix}_HORIZONTAL_APERTURE"],
                        "clipping_range_m": settings[f"{prefix}_CLIPPING_RANGE"],
                        "prim_path": f"/World/envs/env_0/Camera_{view}",
                        # The standalone wrist prim follows panda_hand through pose synchronization.
                        "mounted_on_robot": mounted,
                        "pose_world": {
                            "position": position[first].tolist(),
                            **{
                                f"quaternion_wxyz_{convention}": camera_quaternion_wxyz(
                                    rotation[first], convention
                                ).tolist()
                                for convention in ("world", "ros", "opengl")
                            },
                        },
                    }
                    if pose_streams:
                        obs.create_dataset(
                            f"camera_pose_{sensor_name}",
                            data=camera_pose_wxyz(position[indices], rotation[indices]),
                        )
                        obs.create_dataset(f"camera_timestamp_{sensor_name}", data=sensor["timestamps"][:][indices])
                calibrations[name] = calibration
        if pose_streams:
            temporary.replace(meta / "camera_poses.hdf5")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if calibrations:
        # Keep the reference's first-snapshot file and retain episode-specific randomization.
        documents = {"camera_calibration.json": next(iter(calibrations.values()))}
        if len(calibrations) > 1:
            documents["camera_calibration_episodes.json"] = calibrations
        for name, document in documents.items():
            temp = meta / f"{name}.tmp"
            temp.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
            temp.replace(meta / name)
    return len(calibrations)
