# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise dataset assembly through the actual LeRobot writer and loader."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_collection.export_lerobot import export_dataset
from data_collection.recorder import EpisodeRecorder


class ExportContracts(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "Run in the documented LeRobot export environment")
    def test_collection_round_trip_deduplicates_retries_preserves_stop_and_separates_splits(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from torch.utils.data import DataLoader

        with TemporaryDirectory() as folder:
            root = Path(folder)
            for index, (identity, split) in enumerate((("first", "train"), ("first", "train"), ("second", "eval"))):
                recorder = EpisodeRecorder(
                    root / "native" / f"run_{index}" / "episode_0000",
                    {
                        "episode_id": identity,
                        "split": split,
                        "prompt": "Place both blocks, then stop.",
                        "camera_views": ["top"],
                        "control_dt_s": 1 / 30,
                        "joint_names": ["arm", "finger"],
                        "action_schema": "joint_position_target",
                        "manipulation_protocol": "long_horizon_v1",
                    },
                    root / "native" / f"run_{index}" / "dataset.hdf5",
                    run_config={
                        "task_settings": {
                            "CAMERAS": {"top": {}},
                            "CAMERA_FOCAL_LENGTH": 25,
                            "CAMERA_HORIZONTAL_APERTURE": 20,
                            "CAMERA_CLIPPING_RANGE": [0.01, 10.0],
                        }
                    },
                )
                for step in range(3):
                    camera = {
                        "frame": step,
                        "timestamp_s": step / 30,
                        "rgb": np.full((8, 8, 3), 40 + 40 * step, np.uint8),
                        "intrinsics": np.array([[10, 0, 4], [0, 10, 4], [0, 0, 1]]),
                        "position": np.array([0.0, 0.0, 1.0]),
                        "quaternion_ros_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
                    }
                    recorder.before(
                        np.array([0.2, 0.04], np.float32),
                        "stop" if step == 2 else "transport",
                        step / 30,
                        {"joint_position": np.zeros(2), "joint_velocity": np.zeros(2)},
                        {"top": camera},
                    )
                    recorder.after({"joint_position": np.zeros(2), "joint_velocity": np.zeros(2)}, step == 2)
                recorder.finish(True, "success")
            # Collection archives are linked back into the dataset tree.
            archive = root / "archive"
            (root / "native/run_2").rename(archive)
            (root / "native/run_2").symlink_to(archive, target_is_directory=True)
            (archive / "cycle").symlink_to(root / "native", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "splits separately"):
                export_dataset(root / "native", root / "mixed")
            self.assertFalse((root / "mixed").exists())
            output = root / "training"
            self.assertEqual(export_dataset(root / "native", output, split="train"), 1)
            dataset = LeRobotDataset("local/Benchmark", root=output)
            batch = next(iter(DataLoader(dataset, batch_size=3)))
            self.assertEqual(tuple(batch["action"].shape), (3, 3))
            np.testing.assert_array_equal(batch["action"][:, -1].numpy(), [0, 0, 1])
            np.testing.assert_allclose(batch["action"][:, :2].numpy(), [[0.2, 0.04]] * 3)
            self.assertEqual(tuple(batch["observation.images.top"].shape), (3, 3, 8, 8))
            np.testing.assert_allclose(
                batch["observation.images.top"][:, 0, 0, 0].numpy(), np.array([40, 80, 120]) / 255
            )
            self.assertEqual(len(json.loads((output / "ground_truth.json").read_text())), 1)
            contract = json.loads((output / "meta/action_contract.json").read_text())
            self.assertEqual(contract["split"], "train")


if __name__ == "__main__":
    unittest.main()
