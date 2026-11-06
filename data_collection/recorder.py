# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Incremental HDF5 episodes with causal camera streams and explicit outcomes."""

import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

from data_collection.metadata import write_json_atomic


class EpisodeRecorder:
    """Stream an episode to disk: observations before actions, states afterward."""

    def __init__(
        self,
        directory: Path,
        metadata: dict,
        dataset_path: Path | None = None,
        *,
        scene: dict | None = None,
        recording: dict | None = None,
        run_config: dict | None = None,
        compression: str = "gzip",
    ):
        self.directory = Path(directory)
        if compression not in ("gzip", "lzf", "none"):
            raise ValueError("Recording compression must be gzip, lzf or none")
        self.compression = compression
        self.directory.mkdir(parents=True, exist_ok=False)
        version = (
            4 if scene is not None and (scene.get("scenario") or {}).get("scene_id") else 3 if scene is not None else 2
        )
        self.metadata = dict(metadata, schema_version=version, success=False, finalized=False)
        self.file = h5py.File(dataset_path or self.directory / "episode.hdf5", "a")
        self.dataset_path = Path(self.file.filename)
        self.file.attrs["lossless_compression"] = compression
        self.file.attrs["schema_version"] = version
        for key, value in (("recording_json", recording), ("run_config_json", run_config)):
            if value is not None:
                self.file.attrs[key] = json.dumps(value)
        self.group = self.file.require_group("data").create_group(self.directory.name)
        self.scene = scene
        if scene is not None:
            self._write_scene()
        self.group.attrs["metadata_json"] = json.dumps(self.metadata)
        self.group.attrs["finalized"] = False
        self.group.attrs["success"] = False
        self.frames = {}
        self.datasets = {}
        self.lengths = {}
        self.count = 0
        self.pending = False
        self.before_paths = []
        self._write_metadata()
        self.file.flush()

    def _write_metadata(self):
        write_json_atomic(self.directory / "metadata.json", self.metadata)

    def _write_scene(self):
        payload = deepcopy(self.scene)
        scenario = payload.get("scenario") or {}
        if "scene_id" in scenario:
            identity = scenario["scene_id"]
            shared = self.file.require_group("scene_specs").require_group(identity)
            objects = scenario.pop("objects")
            encoded = json.dumps(objects, sort_keys=True)
            if "objects_json" in shared.attrs and shared.attrs["objects_json"] != encoded:
                raise ValueError("Shared scene identity has conflicting object specifications")
            shared.attrs["objects_json"] = encoded
            payload["shared_scene_id"] = identity
            payload["initial_object_poses"] = [
                {"position_m": obj["position_m"], "rotation_xyzw": obj["rotation_xyzw"]}
                for obj in payload.pop("objects")
            ]
        self.group.attrs["scene_json"] = json.dumps(payload)

    def _append(self, path, value):
        value = np.asarray(value)
        if value.dtype.kind in "UO":
            value = np.asarray(str(value), dtype=h5py.string_dtype("utf-8"))
        dataset = self.datasets.get(path)
        if dataset is None:
            parent, _, name = path.rpartition("/")
            group = self.group.require_group(parent) if parent else self.group
            options = {}
            if value.ndim >= 2 and self.compression != "none":
                options["compression"] = self.compression
                if self.compression == "gzip":
                    options["compression_opts"] = 1
            dataset = group.create_dataset(
                name,
                shape=(0, *value.shape),
                maxshape=(None, *value.shape),
                chunks=(1, *value.shape),
                dtype=value.dtype,
                **options,
            )
            self.datasets[path] = dataset
            self.lengths[path] = 0
        index = self.lengths[path]
        dataset.resize(index + 1, axis=0)
        dataset[index] = value
        self.lengths[path] = index + 1

    def before(self, action, phase, time_s: float, observation: dict, cameras: dict):
        """Record state_t, action_t, and fresh camera samples with causal frame indices."""
        if self.pending:
            raise RuntimeError("after() must complete the preceding transition")
        for view, sample in cameras.items():
            frame = int(sample["frame"])
            if frame == self.frames.get(view):
                continue
            clock = float(sample["timestamp_s"])
            if clock > time_s + 1e-8:
                raise ValueError("A camera sample cannot be newer than its action")
            if self.group.get(f"sensors/{view}/timestamps") is not None:
                if clock <= self.group[f"sensors/{view}/timestamps"][-1]:
                    raise ValueError("Camera timestamps must increase")
            rgb = np.asarray(sample["rgb"])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("RGB must be uint8 with shape (height, width, 3)")
            for key, shape in (("intrinsics", (3, 3)), ("position", (3,)), ("quaternion_ros_xyzw", (4,))):
                value = np.asarray(sample[key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid camera calibration: {view}/{key}")
            if "distance_to_image_plane" in sample:
                depth = np.asarray(sample["distance_to_image_plane"])
                if depth.shape not in (rgb.shape[:2], (*rgb.shape[:2], 1)):
                    raise ValueError("Depth dimensions must match RGB")
            for key, value in sample.items():
                self._append(f"sensors/{view}/" + ("timestamps" if key == "timestamp_s" else key), value)
            self.frames[view] = frame
        if not self.frames:
            raise ValueError("The first action requires at least one camera frame")
        if "initial_state" not in self.group:
            initial = self.group.create_group("initial_state")
            for key, value in observation.items():
                initial.create_dataset(key, data=np.asarray(value))
        rows = {"actions": action, "timestamps": time_s, "phase": phase}
        rows.update({f"obs/{key}": value for key, value in observation.items()})
        for view in self.frames:
            stream = self.group[f"sensors/{view}/timestamps"]
            if stream[-1] > time_s + 1e-8:
                raise ValueError("Latest camera frame is newer than its action")
            rows[f"obs/camera_index/{view}"] = len(stream) - 1
        self.before_paths = list(rows)
        for key, value in rows.items():
            self._append(key, value)
        self.pending = True

    def after(self, state: dict, success: bool):
        """Record the resulting state_t+1 after applying the preceding action."""
        if not self.pending:
            raise RuntimeError("before() must precede after()")
        for key, value in state.items():
            self._append(f"states/{key}", value)
        self._append("success", bool(success))
        self.count += 1
        self.pending = False
        if self.count % 60 == 0:
            self.file.flush()

    def finish(self, success: bool, reason: str):
        """Finalize the episode; truncate an action interrupted before its next state."""
        if self.pending:
            for path in self.before_paths:
                self.group[path].resize(self.count, axis=0)
        self.metadata.update(
            success=bool(success), termination_reason=reason, num_transitions=self.count, finalized=True
        )
        shared = json.loads(self.file.attrs.get("recording_json", "{}"))
        dt = shared.get("control_dt_s", self.metadata.get("control_dt_s"))
        if dt is not None:
            self.metadata["episode_duration_s"] = self.count * dt
        if self.scene is not None:
            self._write_scene()
        self.group.attrs["metadata_json"] = json.dumps(self.metadata)
        self.group.attrs["success"] = bool(success)
        self.group.attrs["finalized"] = True
        self.group.attrs["num_samples"] = self.count
        self.file.flush()
        self._write_metadata()
        self.file.close()

    def control_trace(self, samples):
        """Store physics-rate control evidence separately from action-rate observations."""
        if not samples:
            return
        trace = self.group.create_group("control_trace")
        trace.attrs["schema_version"] = 1
        for key in samples[0]:
            values = np.asarray([sample[key] for sample in samples])
            trace.create_dataset(key, data=values, compression=None if self.compression == "none" else self.compression)
