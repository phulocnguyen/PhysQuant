# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reject recordings whose timestamps conceal invalid policy observations."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_collection.recorder import EpisodeRecorder
from data_collection.validate import validate_dataset


class RecordingIntegrity(unittest.TestCase):
    def test_corrupted_calibration_and_shifted_observations_are_rejected(self):
        # These files have valid lengths and causal indices. Those checks alone
        # cannot catch a shifted proprioceptive stream or unusable calibration.
        corruptions = (
            ("sensors/top/intrinsics", (0, 0, 0), np.nan, "calibration"),
            ("sensors/top/quaternion_ros_xyzw", (0, 3), 0.0, "quaternion"),
            ("obs/joint_position", (1, 0), 4.0, "continuity"),
            ("control_trace/joint_position", (3, 0), 4.0, "physics trace"),
        )
        for field, index, value, message in corruptions:
            with self.subTest(field=field), TemporaryDirectory() as folder:
                root = Path(folder)
                recorder = EpisodeRecorder(
                    root / "episode_0000",
                    {
                        "camera_views": ["top"],
                        "control_dt_s": 1 / 30,
                        "camera_hz": 30,
                        "control_sampling_hz": 120,
                    },
                )
                for step in range(2):
                    sample = {
                        "frame": step,
                        "timestamp_s": step / 30,
                        "rgb": np.full((2, 2, 3), 127, np.uint8),
                        "intrinsics": np.eye(3),
                        "position": np.zeros(3),
                        "quaternion_ros_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
                    }
                    recorder.before(
                        np.zeros(2),
                        "approach",
                        step / 30,
                        {"joint_position": np.full(2, step, dtype=float)},
                        {"top": sample},
                    )
                    recorder.after({"joint_position": np.full(2, step + 1, dtype=float)}, False)
                recorder.control_trace(
                    [
                        {"timestamp_s": (tick + 1) / 120, "joint_position": np.full(2, (tick + 1) / 4)}
                        for tick in range(8)
                    ]
                )
                recorder.finish(False, "timeout")
                self.assertEqual(validate_dataset(recorder.dataset_path)["transitions"], 2)
                with h5py.File(recorder.dataset_path, "r+") as data:
                    data[f"data/episode_0000/{field}"][index] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_dataset(recorder.dataset_path)


if __name__ == "__main__":
    unittest.main()
