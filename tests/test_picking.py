# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick terminal conditions, replay integrity, and per-episode count sampling."""

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assets.geometry import world_aabb
from data_collection.finalize import finalize_attempt
from data_collection.metadata import episode_info, recording_info
from data_collection.recorder import EpisodeRecorder
from data_collection.validate import validate_dataset
from environments import settings
from tasks.common.manifest import create
from tasks.common.picking import PickEvaluator
from tasks.common.task import BenchmarkConfig
from tasks.object_understanding.scenarios import generate


def observation(scene, lift=0.0):
    poses = np.asarray([o["position_m"] + o["rotation_xyzw"] for o in scene["objects"]])
    poses[scene["target_index"], 2] += lift
    bounds = [world_aabb(o["geometry"], pose[:3], pose[3:]) for o, pose in zip(scene["objects"], poses)]
    hand = np.r_[poses[scene["target_index"], :3] + [0, 0, settings.HAND_TO_GRASP], [1, 0, 0, 0]]
    width = scene["objects"][scene["target_index"]]["grasp_width_m"]
    return dict(
        object_pose_xyzw=poses,
        object_aabb_min=np.asarray([b[0] for b in bounds]),
        object_aabb_max=np.asarray([b[1] for b in bounds]),
        object_velocity=np.zeros((len(poses), 6)),
        hand_pose_xyzw=hand,
        joint_position=np.full(2, width / 2),
        joint_velocity=np.zeros(2),
    )


class PickingContracts(unittest.TestCase):
    def test_bounded_contact_jitter_can_hold_but_continuous_motion_cannot(self):
        scene = generate(BenchmarkConfig())
        for motion in ("jitter", "translation", "rotation"):
            with self.subTest(motion=motion):
                evaluator = PickEvaluator(scene, observation(scene))
                for tick in range(120):
                    lift = 0.07 + (0.001 * tick if motion == "translation" else 0.001 * (-1) ** tick)
                    sample = observation(scene, lift=lift)
                    sample["object_velocity"][scene["target_index"], 2] = (
                        0.12 if motion == "translation" else 0.24 * (-1) ** tick
                    )
                    if motion == "rotation":
                        angle = tick * 0.01
                        sample["object_pose_xyzw"][scene["target_index"], 3:] = [
                            0,
                            0,
                            np.sin(angle / 2),
                            np.cos(angle / 2),
                        ]
                    success = evaluator.update(sample, sample["joint_position"], "hold", 1 / 120)
                    if motion != "jitter":
                        self.assertFalse(success)
                self.assertEqual(success, motion == "jitter")

    def test_success_requires_current_sustained_grasp_and_latches_wrong_selection(self):
        scene = generate(BenchmarkConfig())
        initial, lifted = observation(scene), observation(scene, lift=0.07)
        evaluator = PickEvaluator(scene, initial)
        for _ in range(4):
            self.assertFalse(evaluator.update(lifted, lifted["joint_position"], "hold", 0.1))
        self.assertTrue(evaluator.update(lifted, lifted["joint_position"], "hold", 0.1))
        # A peak lift followed by release or a drop must never be a terminal success.
        self.assertFalse(evaluator.update(lifted, [0.04, 0.04], "hold", 0.1))
        self.assertFalse(evaluator.update(initial, initial["joint_position"], "hold", 0.1))
        distant_hand = deepcopy(lifted)
        distant_hand["hand_pose_xyzw"][0] += 0.3
        for _ in range(8):
            self.assertFalse(evaluator.update(distant_hand, lifted["joint_position"], "hold", 0.1))
        for _ in range(5):
            success = evaluator.update(lifted, lifted["joint_position"], "hold", 0.1)
        self.assertTrue(success)
        wrong_index = (scene["target_index"] + 1) % len(scene["objects"])
        wrong = deepcopy(lifted)
        wrong["object_pose_xyzw"][wrong_index, 2] += 0.04
        self.assertFalse(evaluator.update(wrong, lifted["joint_position"], "hold", 0.1))
        for _ in range(8):
            self.assertFalse(evaluator.update(lifted, lifted["joint_position"], "hold", 0.1))
        self.assertEqual(evaluator.metrics["wrong_object_indices"], [wrong_index])

    def test_physics_replay_rejects_changed_grasp_evidence(self):
        from evaluation.offline.review import accepted

        scene = generate(BenchmarkConfig(split="train"))
        before = observation(scene)
        evaluator = PickEvaluator(scene, before)
        recording = recording_info(settings, ["panda_finger_joint1", "panda_finger_joint2"], "training", "pick_hold_v1")
        recording["camera_views"] = ["top"]
        camera = dict(
            frame=0,
            timestamp_s=0.0,
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            intrinsics=np.eye(3),
            position=np.zeros(3),
            quaternion_ros_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
        )
        with TemporaryDirectory() as folder:
            writer = EpisodeRecorder(
                Path(folder) / "episode_0000",
                episode_info("object_understanding", scene, scene["seed"], scene["target_index"]),
                scene=dict(scenario=scene, objects=scene["objects"]),
                recording=recording,
            )
            for step in range(40):
                writer.before(before["joint_position"], "hold", step / 30, before, {"top": camera} if step == 0 else {})
                for tick in range(4):
                    after = observation(scene, min(0.07, (step * 4 + tick + 1) * 0.001))
                    success = evaluator.update(after, after["joint_position"], "hold", 1 / 120)
                writer.after(after, success)
                before = after
                if success:
                    break
            self.assertTrue(success)
            writer.control_trace(evaluator.trace)
            writer.metadata.update(pick_metrics=evaluator.metrics, pick_sampling_hz=120)
            finalize_attempt(writer, success, "success", SimpleNamespace(video=False))
            self.assertEqual(validate_dataset(writer.dataset_path)["successes"], 1)
            reviewed = json.loads((writer.directory / "metadata.json").read_text())
            self.assertFalse(accepted(reviewed))
            reviewed["observability_status"] = "verified"
            reviewed["validation_gates"]["observability"] = "passed"
            self.assertTrue(accepted(reviewed))
            with h5py.File(writer.dataset_path, "r+") as data:
                trace = data["data/episode_0000/control_trace"]
                trace["finger_positions"][-1] = [0.04, 0.04]
            with self.assertRaisesRegex(ValueError, "finger evidence"):
                validate_dataset(writer.dataset_path)

    def test_random_counts_are_reproducible_per_episode_and_do_not_hide_infeasible_geometry(self):
        from tasks.common.scenarios import InfeasibleScenario

        cfg = BenchmarkConfig(case="diameter", value=5.5, gap=0.4, shapes=["cylinder"])
        plan = create(cfg, episode_count=40, random_candidate_counts=[3, 4, 5, 6, 7])
        entries = plan["episodes"]
        counts = [e["scenario"]["candidate_count"] for e in entries]
        self.assertEqual(len(entries), 40)
        self.assertEqual(set(counts), {3, 4, 5, 6, 7})
        self.assertNotEqual(counts, sorted(counts))
        self.assertEqual(len({e["scenario"]["scene_seed"] for e in entries}), 40)
        self.assertEqual(len({e["episode_id"] for e in entries}), 40)
        self.assertEqual(plan, create(cfg, episode_count=40, random_candidate_counts=[3, 4, 5, 6, 7]))
        with self.assertRaisesRegex(ValueError, "exact episode budget"):
            create(cfg, random_candidate_counts=[3, 7])
        cfg.value, cfg.gap = 5, 0.4
        with self.assertRaises(InfeasibleScenario):
            create(cfg, episode_count=2, random_candidate_counts=[7])


if __name__ == "__main__":
    unittest.main()
