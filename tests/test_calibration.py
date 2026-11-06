# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check reference camera axes, moving-camera alignment, and episode calibration."""

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_collection.calibration import export_calibration
from data_collection.recorder import EpisodeRecorder


class CalibrationContracts(unittest.TestCase):
    def test_camera_capture_owns_a_snapshot_when_backend_buffers_are_reused(self):
        from types import SimpleNamespace

        import torch
        from environments.cameras import CameraRig

        def array(value):
            return SimpleNamespace(torch=torch.as_tensor(value))

        from data_collection.parallel import ParallelCameraRig, _slice_images

        for rig_type, count in ((CameraRig, 1), (ParallelCameraRig, 2)):
            with self.subTest(rig=rig_type.__name__):
                origins = np.array([[10 * i, 20 * i, 0] for i in range(count)], dtype=np.float32)
                pixels = np.full((count, 2, 2, 3), 127, dtype=np.uint8)
                pixels[-1] = 231
                data = SimpleNamespace(
                    output={
                        "rgb": array(pixels),
                        "distance_to_image_plane": array(np.ones((count, 2, 2, 1), dtype=np.float32)),
                    },
                    intrinsic_matrices=array(np.repeat(np.eye(3)[None], count, axis=0)),
                    pos_w=array(origins + [0.2, 0.3, 1.0]),
                    quat_w_ros=array([[0.0, 0.0, 0.0, 1.0]] * count),
                )
                camera = SimpleNamespace(data=data, frame=array([1] * count), update=lambda *args, **kwargs: None)
                rig = rig_type.__new__(rig_type)
                rig.mounts = []
                rig.sim = SimpleNamespace(render=lambda: None)
                rig.cameras = {"top": camera}
                rig.origins = torch.as_tensor(origins)
                rig.record_depth = True
                # Only the native render operation is stubbed; real capture and
                # environment slicing must detach samples from reusable buffers.
                with patch("isaaclab.sensors.SensorBase.update_batch"):
                    frames = rig.capture(0.05)
                sample = (frames if count == 1 else _slice_images(frames, count - 1))["top"]
                for buffer in (*data.output.values(), data.intrinsic_matrices, data.pos_w, data.quat_w_ros):
                    buffer.torch.zero_()
                np.testing.assert_array_equal(sample["rgb"], np.full((2, 2, 3), 231, dtype=np.uint8))
                np.testing.assert_array_equal(sample["distance_to_image_plane"], np.ones((2, 2, 1)))
                np.testing.assert_array_equal(sample["intrinsics"], np.eye(3))
                np.testing.assert_allclose(sample["position"], [0.2, 0.3, 1.0])
                np.testing.assert_array_equal(sample["quaternion_ros_xyzw"], [0, 0, 0, 1])

    def test_reference_axes_and_causal_moving_camera_poses_preserve_native_data(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / "dataset.hdf5"
            settings = {
                "RENDER_SETTINGS": {"mode": "path_traced"},
                "CAMERAS": {"wrist": {"body": "panda_hand"}},
                "WRIST_CAMERA_FOCAL_LENGTH": 24.0,
                "WRIST_CAMERA_HORIZONTAL_APERTURE": 20.955,
                "WRIST_CAMERA_CLIPPING_RANGE": [0.01, 10.0],
            }
            (root / "run_config.yaml").write_text(json.dumps({"task_settings": settings}))
            for episode_index, focal_pixels in enumerate((100.0, 120.0)):
                recorder = EpisodeRecorder(
                    root / f"episode_{episode_index:04d}",
                    {"camera_views": ["wrist"], "control_dt_s": 1 / 30},
                    dataset,
                )
                for step in range(3):
                    sample = {}
                    if step in (0, 2):
                        sample["wrist"] = {
                            "frame": step // 2,
                            "timestamp_s": 0.0 if step == 0 else 0.05,
                            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                            "intrinsics": np.array([[focal_pixels, 0, 1], [0, focal_pixels, 1], [0, 0, 1]]),
                            "position": np.array([step, 0.0, 1.0]),
                            "quaternion_ros_xyzw": np.array(
                                [0, 0, 0, 1] if step == 0 else [0, 0, np.sqrt(0.5), np.sqrt(0.5)],
                                dtype=np.float64,
                            ),
                        }
                    recorder.before(np.zeros(9), "approach", step / 30, {"joint_position": np.zeros(9)}, sample)
                    recorder.after({"joint_position": np.zeros(9)}, False)
                recorder.finish(False, "timeout")
            original_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
            self.assertEqual(export_calibration(dataset, root / "converted"), 2)
            self.assertEqual(hashlib.sha256(dataset.read_bytes()).hexdigest(), original_hash)
            meta = root / "converted/meta"
            calibration = json.loads((meta / "camera_calibration.json").read_text())
            camera = calibration["cameras"]["wrist"]
            self.assertTrue(camera["mounted_on_robot"])
            self.assertEqual(camera["sensor_name"], "camera_wrist")
            self.assertEqual(camera["intrinsic_matrix"][0][0], 100.0)
            np.testing.assert_allclose(camera["pose_world"]["quaternion_wxyz_ros"], [1, 0, 0, 0])
            np.testing.assert_allclose(camera["pose_world"]["quaternion_wxyz_opengl"], [0, 1, 0, 0])
            # ROS identity looks along world +Z, with screen-up along world -Y.
            np.testing.assert_allclose(camera["pose_world"]["quaternion_wxyz_world"], [0.5, 0.5, -0.5, 0.5])
            episodes = json.loads((meta / "camera_calibration_episodes.json").read_text())
            self.assertEqual(episodes["episode_0001"]["cameras"]["wrist"]["intrinsic_matrix"][0][0], 120.0)
            with h5py.File(meta / "camera_poses.hdf5", "r") as poses:
                obs = poses["data/episode_0000/obs"]
                np.testing.assert_allclose(
                    obs["camera_pose_camera_wrist"][:],
                    [
                        [0, 0, 1, 0.5, 0.5, -0.5, 0.5],
                        [0, 0, 1, 0.5, 0.5, -0.5, 0.5],
                        [2, 0, 1, 0, np.sqrt(0.5), 0, np.sqrt(0.5)],
                    ],
                    atol=1e-7,
                )
                np.testing.assert_allclose(obs["camera_timestamp_camera_wrist"][:], [0, 0, 0.05])


if __name__ == "__main__":
    unittest.main(verbosity=2)
