# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable bundle and policy-boundary contracts independent of simulator startup."""

import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.failure_analysis import analyze
from evaluation.metrics import summarize
from evaluation.policies.gr00t import Gr00tPolicy
from evaluation.policies.llm_policy import LLMPolicy
from evaluation.policies.pi05 import Pi05Policy
from evaluation.protocol import decode, encode
from tools.bundle import build_bundle


class EvaluationContracts(unittest.TestCase):
    def test_policy_outputs_preserve_joint_order_stop_and_observation_buffers(self):
        """Checkpoint adapters must return physical targets without erasing a stop dimension."""
        observation = {"observation.state": np.arange(9, dtype=np.float32), "task": "Build a tower."}
        raw = np.arange(20, dtype=np.float32).reshape(2, 10)

        class NativePolicy:
            def infer(self, value):
                return {"actions": raw}

            def get_action(self, value):
                return {"joints": raw}, {"inference_time": 0.01}

        resets = []
        adapters = (
            Pi05Policy(
                NativePolicy(),
                observation_transform=dict,
                action_transform=lambda value, obs: value,
                reset_model=resets.append,
            ),
            Gr00tPolicy(
                NativePolicy(),
                observation_transform=dict,
                action_transform=lambda value, obs: value["joints"],
                reset_model=resets.append,
            ),
            LLMPolicy(
                lambda obs: json.dumps({"actions": raw.tolist()}),
                observation_transform=dict,
                action_transform=lambda value, obs: value,
                reset_model=resets.append,
            ),
        )
        for adapter in adapters:
            with self.subTest(adapter=type(adapter).__name__):
                adapter.reset({"action_dim": 10})
                reply = decode(encode({"action": adapter.act(observation)}))["action"]
                np.testing.assert_array_equal(reply, np.arange(20).reshape(2, 10))
                self.assertEqual(reply.dtype, np.float32)
                adapter.reset({"action_dim": 9})
                with self.assertRaisesRegex(ValueError, "joint targets"):
                    adapter.act(observation)
        self.assertEqual(len(resets), 6)
        np.testing.assert_array_equal(observation["observation.state"], np.arange(9))
        bad = LLMPolicy(
            lambda obs: '{"actions": []}', observation_transform=dict, action_transform=lambda value, obs: value
        )
        bad.reset({"action_dim": 9})
        with self.assertRaises(ValueError):
            bad.act(observation)

    def test_reports_keep_execution_errors_separate_from_measured_task_failures(self):
        records = [
            {"suite": "object_understanding", "tier": "t1", "case": "width", "status": "completed", "success": True},
            {
                "suite": "object_understanding",
                "tier": "t1",
                "case": "width",
                "status": "completed",
                "success": False,
                "metrics": {"wrong_object_indices": [1], "distractors_preserved": False},
            },
            {
                "suite": "object_understanding",
                "tier": "t1",
                "case": "height",
                "status": "error",
                "success": False,
                "error": "Connection lost",
            },
        ]
        report = summarize(records)["object_understanding"]["t1"]
        self.assertEqual(
            (report["episodes"], report["completed"], report["errors"], report["success_rate"]), (3, 2, 1, 0.5)
        )
        failures = analyze(records)
        self.assertEqual(failures["outcomes"], {"success": 1, "task_failure": 1, "execution_error": 1})
        self.assertEqual(failures["evidence"], {"wrong_object_lifted": 1, "distractor_moved": 1})
        self.assertEqual(failures["failures"][1]["error"], "Connection lost")

    def test_bundle_resolves_background_and_texture_after_source_is_removed(self):
        """Resolve a USD composition after moving the bundle away from all source paths."""
        from pxr import Usd

        with TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "project"
            for folder in (
                "configs",
                "controllers",
                "core",
                "data_collection",
                "environments",
                "evaluation",
                "tasks",
                "tools",
                "assets/scenes/backgrounds",
            ):
                (root / folder).mkdir(parents=True, exist_ok=True)
            source = temporary / "InteriorAgent/room"
            source.mkdir(parents=True)
            (source / "room.usda").write_text(
                '#usda 1.0\n(defaultPrim = "Room")\ndef Xform "Room" {\n custom asset texture = @texture.bin@\n}\n'
            )
            (source / "texture.bin").write_bytes(b"portable texture")
            (root / "assets/textures").symlink_to(source, target_is_directory=True)
            layer = root / "assets/scenes/backgrounds/room.usda"
            layer.write_text(
                '#usda 1.0\ndef Xform "Background" {\n'
                f' def Xform "Scene" (prepend references = @{source}/room.usda@) {{}}\n'
                "}\n"
            )
            manifests = root / "manifests/benchmark"
            plan = manifests / "object_understanding/t1_in_distribution"
            (plan / "scenarios/width").mkdir(parents=True)
            (plan / "plan.yaml").write_text("schema_version: 3\nepisodes: []\n")
            (plan / "scenarios/width/episode.json").write_text(
                json.dumps({"appearance": {"background": {"usd": "assets/scenes/backgrounds/room.usda"}}})
            )
            output = temporary / "bundle"
            build_bundle(root, manifests, output)
            relocated = temporary / "relocated"
            output.rename(relocated)
            shutil.rmtree(root)
            shutil.rmtree(source)
            stage = Usd.Stage.Open(str(relocated / "assets/scenes/backgrounds/room.usda"))
            texture = stage.GetPrimAtPath("/Background/Scene").GetAttribute("texture").Get()
            self.assertEqual(Path(texture.resolvedPath).read_bytes(), b"portable texture")
            self.assertEqual((relocated / "assets/textures/texture.bin").read_bytes(), b"portable texture")
            self.assertFalse(any(path.is_symlink() for path in relocated.rglob("*")))


if __name__ == "__main__":
    unittest.main()
