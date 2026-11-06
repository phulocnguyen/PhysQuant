# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical-control contracts: exact goals, immutable scenes, and path-sensitive scoring."""

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assets.geometry import world_aabb
from data_collection.config import load_config
from data_collection.manifest import load_manifest, validate_manifest, write_manifest
from tasks.common.scenarios import ROOT, compatibility_key
from tasks.control.evaluator import ControlEvaluator
from tasks.control.goals import angle_error, quaternion
from tasks.control.manifest import create
from tasks.control.scenarios import seal, validate
from tasks.control.task import ControlConfig, configure
from tasks.registry import SUBGROUPS, subgroup_of_case


def configured(case):
    cfg = load_config(ROOT / "configs/tasks" / subgroup_of_case(case) / f"{case}.yaml", {}, ControlConfig)
    cfg.initial_jitter_xy_m = [0, 0]
    configure(cfg)
    return cfg


def observation(scene, position=None, *, released=False, angle=None):
    poses = np.asarray([obj["position_m"] + obj["rotation_xyzw"] for obj in scene["objects"]])
    if position is not None:
        poses[0, :3] = position
    if angle is not None:
        poses[0, 3:] = quaternion(angle)
    bounds = [world_aabb(obj["geometry"], pose[:3], pose[3:]) for obj, pose in zip(scene["objects"], poses)]
    return dict(
        object_pose_xyzw=poses,
        object_velocity=np.zeros((len(poses), 6)),
        object_aabb_min=np.array([b[0] for b in bounds]),
        object_aabb_max=np.array([b[1] for b in bounds]),
        hand_pose_xyzw=np.r_[poses[0, :3] + [0, 0, 0.30 if released else 0.1034], [1, 0, 0, 0]],
    )


def feed(evaluator, scene, position, *, released=False, seconds=0.6, angle=None):
    obs = observation(scene, position, released=released, angle=angle)
    result = False
    for _ in range(round(seconds * 120)):
        result = evaluator.update(obs, [0.04, 0.04] if released else [0.02, 0.02], "arbitrary_policy_label", 1 / 120)
    return result


class SpatialControl(unittest.TestCase):
    def test_table_frame_migration_preserves_motion_and_separates_visual_workers(self):
        current = configured("ST-1A").scenario
        self.assertIn("move it 10 cm right from its starting center", current["instruction"])
        self.assertNotIn("+X", current["instruction"])
        self.assertNotIn("robot-side", current["instruction"])
        self.assertNotIn("strips", current["instruction"])
        absolute_cfg = configured("TP-1A")
        absolute_cfg.numerical_goal = {"position_cm": [-12, 8]}
        configure(absolute_cfg)
        absolute = absolute_cfg.scenario
        self.assertIn("12 cm left and 8 cm forward from the center of the tabletop", absolute["instruction"])
        np.testing.assert_allclose(absolute["resolved_goal"]["target_position_m"][:2], [-0.12, 0.08])
        legacy = deepcopy(current)
        legacy["template_version"] = 3
        legacy["reference_frame"] = dict(
            id="table",
            origin_m=[0.0, 0.0, 0.75],
            axes="+X right, +Y forward, +Z up",
            cue="red +X and blue +Y strips, each 12 cm long",
        )
        seal(legacy)
        validate(legacy)
        self.assertEqual(current["objects"], legacy["objects"])
        self.assertEqual(current["resolved_goal"], legacy["resolved_goal"])
        self.assertNotEqual(compatibility_key(current), compatibility_key(legacy))
        self.assertNotEqual(current["episode_id"], legacy["episode_id"])
        self.assertIn("each strip is 12 cm long", legacy["instruction"])
        shifted = deepcopy(current)
        shifted["reference_frame"]["origin_m"][0] = 0.05
        seal(shifted)
        with self.assertRaisesRegex(ValueError, "Table frame"):
            validate(shifted)

    def test_appearance_changes_between_scenes_but_not_matched_goals_or_retries(self):
        from tasks.common.appearance import apply_appearance

        cfg = configured("ST-1A")
        first = deepcopy(cfg.scenario)
        configure(cfg)
        self.assertEqual(cfg.scenario, first)
        cfg.numerical_goal = {"displacement_cm": [14, 0, 0]}
        configure(cfg)
        self.assertEqual(cfg.scenario["appearance"], first["appearance"])
        self.assertEqual(cfg.scenario["scene_id"], first["scene_id"])
        colors, tables, backgrounds = set(), set(), set()
        for seed in (712, 713, 717):
            cfg.seed = seed
            configure(cfg)
            s = cfg.scenario
            appearance = s["appearance"]
            colors.add(appearance["candidate"]["name"])
            tables.add(appearance["table"]["name"])
            backgrounds.add(appearance["background"]["name"])
            self.assertIn(f"{appearance['candidate']['name']} block", s["instruction"])
            self.assertEqual(compatibility_key(s), compatibility_key(first))
        self.assertGreater(len(colors), 1)
        self.assertGreater(len(tables), 1)
        self.assertGreater(len(backgrounds), 1)
        altered = deepcopy(cfg.scenario)
        altered["objects"][0]["color_rgb"] = [1, 1, 1]
        with self.assertRaisesRegex(ValueError, "share the frozen scene color"):
            validate(altered)
        disabled = deepcopy(first)
        disabled["randomization"] = {"enabled": False, "target_object": {"enabled": True}, "light": {"enabled": True}}
        apply_appearance(disabled)
        self.assertTrue(disabled["randomization"]["table"]["enabled"])
        self.assertFalse(disabled["randomization"]["target_object"]["enabled"])
        self.assertFalse(disabled["randomization"]["light"]["enabled"])

        rotation = configured("ST-2A").scenario
        rotation["appearance"]["candidate"] = dict(name="purple", color_rgb=[0.4, 0.01, 0.7])
        rotation["objects"][0]["color_rgb"] = [0.4, 0.01, 0.7]
        seal(rotation)
        self.assertIn("purple rectangular block", rotation["instruction"])
        legacy_rotation = deepcopy(rotation)
        legacy_rotation["template_version"] = 4
        seal(legacy_rotation)
        validate(legacy_rotation)
        self.assertIn("orange rectangular block", legacy_rotation["instruction"])
        self.assertEqual(rotation["objects"], legacy_rotation["objects"])
        self.assertEqual(rotation["resolved_goal"], legacy_rotation["resolved_goal"])
        self.assertNotEqual(rotation["episode_id"], legacy_rotation["episode_id"])

    def test_measured_grasp_width_prevents_close_stall_after_contact_rotation(self):
        from controllers.single_object import SingleObjectController
        from tasks.control.task import resolve_settings

        cfg = configured("ST-1A")
        s = cfg.scenario
        obj = s["objects"][0]
        initial = np.asarray(obj["position_m"] + obj["rotation_xyzw"])
        motion = SingleObjectController(
            initial, 0.045, s["resolved_goal"], s["goal"], resolve_settings(cfg), 0.0, obj["size_m"]
        )
        motion.phase = "close"
        turned = initial.copy()
        turned[3:] = quaternion(np.deg2rad(-7))
        hand = np.r_[motion.positions["close"], [1.0, 0.0, 0.0, 0.0]]
        _, _, phase = motion.command(hand, np.array([0.024, 0.024]), np.zeros(2), object_pose=turned)
        self.assertEqual(phase, "lift")

    def test_shipped_families_validate_and_preserve_instruction_through_manifest(self):
        episodes = []
        paths = sorted(ROOT.glob("configs/tasks/*/*.yaml"))
        for path in (path for path in paths if SUBGROUPS[path.parent.name]["engine"] == "control"):
            with self.subTest(config=path.stem):
                cfg = load_config(path, {}, ControlConfig)
                configure(cfg)
                scene = json.loads(json.dumps(cfg.scenario))
                self.assertEqual(validate(scene)["targets"], 1)
                altered = deepcopy(scene)
                altered["resolved_goal"]["target_position_m"][0] += 0.1
                with self.assertRaisesRegex(ValueError, "derived physical target"):
                    validate(altered)
                episodes.append(dict(episode_index=len(episodes), episode_id=scene["episode_id"], scenario=scene))
        manifest = dict(schema_version=3, benchmark="Benchmark", episodes=episodes, infeasible_cells=[])
        with TemporaryDirectory() as d:
            write_manifest(manifest, d, {"planned": len(episodes)})
            restored = load_manifest(d)
            validate_manifest(restored)
            self.assertEqual(
                [e["scenario"]["instruction"] for e in restored["episodes"]],
                [e["scenario"]["instruction"] for e in episodes],
            )

    def test_goal_sweeps_reuse_scene_but_change_targets_and_preserve_split_groups(self):
        cfg = configured("ST-1A")
        cfg.goal_sweep = [{"displacement_cm": [6, 0, 0]}, {"displacement_cm": [14, 0, 0]}]
        m = create(cfg)
        scenes = [e["scenario"] for e in m["episodes"]]
        self.assertEqual(len(scenes), 4)
        self.assertEqual(len({s["scene_id"] for s in scenes}), 1)
        self.assertEqual(len({compatibility_key(s) for s in scenes}), 1)
        self.assertEqual(len({s["scenario_id"] for s in scenes}), 2)
        np.testing.assert_allclose(scenes[0]["resolved_goal"]["target_position_m"], [-0.06, 0.08, 0.775])
        np.testing.assert_allclose(scenes[1]["resolved_goal"]["target_position_m"], [0.02, 0.08, 0.775])
        cfg.split = "eval"
        other = create(cfg, all_templates=False)
        mixed = deepcopy(m)
        mixed["episodes"].append(dict(other["episodes"][0], episode_index=4))
        with self.assertRaisesRegex(ValueError, "leaks across splits"):
            validate_manifest(mixed)

    def test_reference_and_size_changes_move_clearance_goal_without_changing_gap(self):
        cfg = configured("TP-2A")
        base = cfg.scenario
        self.assertAlmostEqual(base["resolved_goal"]["target_position_m"][0], 0.07)
        cfg.references["A"]["geometry"]["size_m"][0] = 0.10
        cfg.references["A"]["position_xy_m"][0] = -0.10
        configure(cfg)
        self.assertAlmostEqual(cfg.scenario["resolved_goal"]["target_position_m"][0], 0.06)
        self.assertEqual(cfg.scenario["numerical_goal"]["gap_cm"], 6)
        cfg = configured("ST-1C")
        cfg.numerical_goal = {"displacement_cm": [0, 0, -10]}
        with self.assertRaisesRegex(ValueError, "positive vertical"):
            configure(cfg)
        cfg = configured("ST-2A")
        cfg.geometry_options = [{"kind": "cylinder", "radius_m": 0.025, "height_m": 0.10}]
        with self.assertRaisesRegex(ValueError, "elongated"):
            configure(cfg)
        self.assertAlmostEqual(angle_error(np.deg2rad(225), np.deg2rad(45)), 0)
        self.assertGreater(angle_error(np.deg2rad(135), np.deg2rad(45)), 1.5)

    def test_vertical_hold_requires_actual_height_and_retention_not_open_gripper(self):
        s = configured("ST-1C").scenario
        e = ControlEvaluator(s, observation(s))
        target = s["resolved_goal"]["target_position_m"]
        self.assertFalse(feed(e, s, np.asarray(target) + [0, 0, 0.04]))
        self.assertFalse(feed(e, s, target, released=True))
        self.assertTrue(feed(e, s, target))
        # Actual pose is stationary despite a nonzero contact-solver velocity.
        held = observation(s, target)
        held["object_velocity"][0, 3] = 0.65
        for _ in range(70):
            result = e.update(held, [0.02, 0.02], "hold", 1 / 120)
        self.assertTrue(result)
        # Measurable orientation oscillation must still fail held stability.
        for i in range(70):
            moving = observation(s, target, angle=0.02 if i % 2 else -0.02)
            result = e.update(moving, [0.02, 0.02], "hold", 1 / 120)
        self.assertFalse(result)

    def test_waypoints_reject_diagonal_shortcut_and_record_order(self):
        s = configured("CM-1A").scenario
        bad = ControlEvaluator(s, observation(s))
        route = s["resolved_goal"]["waypoints_m"]
        feed(bad, s, route[-1])
        self.assertFalse(feed(bad, s, s["resolved_goal"]["target_position_m"], released=True))
        self.assertTrue(bad.metrics["waypoint_order_violation"])
        good = ControlEvaluator(s, observation(s))
        for point in route:
            feed(good, s, point, seconds=0.05)
        self.assertTrue(feed(good, s, s["resolved_goal"]["target_position_m"], released=True))
        self.assertEqual(good.metrics["waypoints_reached"], 2)

    def test_transport_violations_cannot_be_erased_by_endpoint_success(self):
        for case in ("CM-2A", "CM-2B", "CM-2C"):
            with self.subTest(case=case):
                s = configured(case).scenario
                e = ControlEvaluator(s, observation(s))
                start = np.array(s["objects"][0]["position_m"])
                start[2] = s["resolved_goal"]["transport_z_m"]
                feed(e, s, start, seconds=0.05)
                bad = start.copy()
                bad[0] += 0.02
                angle = None
                if case == "CM-2A":
                    bad[2] -= 0.10
                if case == "CM-2B":
                    angle = np.deg2rad(40)
                if case == "CM-2C":
                    bad[:2] = s["objects"][1]["position_m"][:2]
                feed(e, s, bad, seconds=1 / 120, angle=angle)
                for point in s["resolved_goal"]["oracle_route_m"]:
                    feed(e, s, point)
                self.assertFalse(feed(e, s, s["resolved_goal"]["target_position_m"], released=True))
                self.assertGreater(e.metrics["maximum_constraint_violation"], 0)
                self.assertGreater(e.metrics["violation_duration_s"], 0)
                # Clean oracle route has zero violation and placement succeeds.
                good = ControlEvaluator(s, observation(s))
                pregrasp = np.asarray(s["objects"][0]["position_m"]) + [0, -0.006, 0]
                feed(good, s, pregrasp, seconds=0.1)
                self.assertEqual(good.transport_samples, 0)
                feed(good, s, start)
                for point in s["resolved_goal"]["oracle_route_m"]:
                    feed(good, s, point)
                self.assertTrue(feed(good, s, s["resolved_goal"]["target_position_m"], released=True))
                self.assertEqual(good.metrics["maximum_constraint_violation"], 0)


if __name__ == "__main__":
    unittest.main()
