# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned scene, language, and placement contracts independent of the simulator."""

import math
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assets.geometry import describe, make, surface_area
from data_collection.manifest import load_manifest, validate_manifest, write_manifest
from selection_fixtures import placement_plan
from tasks.common.manifest import create
from tasks.common.placement import PROTOCOL, PlacementEvaluator
from tasks.common.predicates import matches, quantity, validate_predicate
from tasks.common.scenarios import validate
from tasks.common.task import BenchmarkConfig, configure
from tasks.object_understanding.scenarios import generate


class GeometryAndIdentity(unittest.TestCase):
    def test_candidate_size_bounds_reject_instead_of_rescaling_queries(self):
        from tasks.common.scenarios import InfeasibleScenario

        for cfg in (BenchmarkConfig(case="width", value=3, gap=0.2), BenchmarkConfig(value=19, gap=1)):
            with self.subTest(case=cfg.case), self.assertRaisesRegex(InfeasibleScenario, "primary candidate size"):
                generate(cfg)
        scene = generate(
            BenchmarkConfig(case="diameter", shapes=["cylinder"], value=55, unit="mm", gap=3, candidate_count=9)
        )
        target = scene["objects"][scene["target_index"]]
        self.assertAlmostEqual(target["physical_properties_si"]["diameter"], 0.055)
        self.assertAlmostEqual(target["physical_properties_si"]["height"], 0.12)
        self.assertEqual(scene["query"]["diameter"]["value"], 55)

    def test_candidates_share_color_across_families_and_reference_colors_remain_distinct(self):
        for family, case, value in (
            ("object_understanding", "length", 12),
            ("relational_understanding", "reference_frame", 20),
            ("quantitative_constraint_grounding", "arithmetic", 10),
        ):
            for seed in (712, 713):
                with self.subTest(family=family, seed=seed):
                    cfg = BenchmarkConfig(task=family, case=case, value=value, seed=seed)
                    configure(cfg)
                    candidates = [o for o in cfg.scenario["objects"] if o.get("selectable", True)]
                    references = [o for o in cfg.scenario["objects"] if not o.get("selectable", True)]
                    colors = {tuple(o["color_rgb"]) for o in candidates}
                    self.assertEqual(len(colors), 1)
                    self.assertTrue(all(tuple(o["color_rgb"]) not in colors for o in references))
                    self.assertEqual(len({tuple(o["color_rgb"]) for o in references}), len(references))

    def test_closed_surface_area_is_distinct_from_footprint(self):
        examples = [
            ({"kind": "box", "size_m": [0.1, 0.05, 0.03]}, 0.019),
            ({"kind": "cylinder", "radius_m": 0.02, "height_m": 0.08}, 0.004 * math.pi),
            ({"kind": "sphere", "radius_m": 0.03}, 0.0036 * math.pi),
        ]
        for geometry, expected in examples:
            with self.subTest(kind=geometry["kind"]):
                self.assertAlmostEqual(surface_area(geometry), expected)
                self.assertGreater(surface_area(geometry), describe(geometry)["measurements"]["footprint_area"])
        for shape in ("box", "cylinder", "sphere"):
            self.assertAlmostEqual(surface_area(make(shape, "surface_area", 0.016)), 0.016)
        with self.assertRaisesRegex(ValueError, "not qualified"):
            surface_area(make("rectangular_container", "capacity", 0.0002))

    def test_counts_gaps_and_language_splits_are_independent(self):
        for count in range(3, 10):
            s = generate(BenchmarkConfig(candidate_count=count, case="diameter", shapes=["cylinder"], value=5, gap=0.2))
            self.assertEqual(len(s["objects"]), count)
            self.assertAlmostEqual(s["minimum_gap_si"], 0.002)
            self.assertNotIn("marked goal zone", s["instruction"])
            self.assertEqual(s["manipulation_protocol"], "pick_hold_v1")
            self.assertNotIn("untouched", s["instruction"])
        seen = generate(BenchmarkConfig())
        held = generate(BenchmarkConfig(template_split="held_out", split="eval"))
        self.assertEqual(seen["scene_id"], held["scene_id"])
        self.assertEqual(seen["scenario_id"], held["scenario_id"])
        self.assertNotEqual(seen["episode_id"], held["episode_id"])
        with self.assertRaisesRegex(ValueError, "training"):
            generate(BenchmarkConfig(template_split="held_out", split="train"))
        for count in (2, 10, 12, True):
            with self.assertRaisesRegex(ValueError, "3 through 9"):
                generate(BenchmarkConfig(candidate_count=count))

    def test_exact_episode_budget_balances_counts_and_preserves_unique_queries(self):
        from collections import Counter

        cfg = BenchmarkConfig(
            case="diameter", value=55, unit="mm", gap=3, shapes=["cylinder"], render_profile="balanced"
        )
        plan = create(cfg, episode_count=101, candidate_counts=list(range(3, 10)))
        entries = plan["episodes"]
        self.assertEqual(len(entries), 101)
        self.assertEqual(len({e["episode_id"] for e in entries}), 101)
        counts = Counter(e["scenario"]["candidate_count"] for e in entries)
        self.assertEqual(set(counts), set(range(3, 10)))
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(entries, create(cfg, episode_count=101, candidate_counts=list(range(3, 10)))["episodes"])
        for budget in (0, -1, True):
            with self.assertRaisesRegex(ValueError, "episode_count"):
                create(cfg, episode_count=budget)

    def test_render_profile_changes_identity_without_changing_truth_or_physics(self):
        from tasks.common.task import resolve_settings

        quality, balanced = BenchmarkConfig(), BenchmarkConfig(render_profile="balanced")
        configure(quality)
        configure(balanced)
        self.assertEqual(quality.scenario["objects"], balanced.scenario["objects"])
        self.assertEqual(quality.scenario["query"], balanced.scenario["query"])
        self.assertNotEqual(quality.scenario["scene_id"], balanced.scenario["scene_id"])
        for key in ("PHYSICS_DT", "CONTROL_STEPS", "CAMERAS", "IMAGE_WIDTH", "IMAGE_HEIGHT", "CAMERA_HZ"):
            self.assertEqual(getattr(resolve_settings(quality), key), getattr(resolve_settings(balanced), key))

    def test_manifest_sweeps_every_target_and_rejects_split_leakage(self):
        plan = create(BenchmarkConfig())
        scenes = [e["scenario"] for e in plan["episodes"]]
        self.assertEqual(len(plan["episodes"]), 9)
        self.assertEqual(len({s["scene_id"] for s in scenes}), 1)
        self.assertEqual(len({s["target_id"] for s in scenes}), 3)
        self.assertEqual(len({s["template_id"] for s in scenes}), 3)
        with TemporaryDirectory() as folder:
            write_manifest(plan, folder, {})
            restored = load_manifest(Path(folder) / "plan.yaml")
            validate_manifest(restored)
            self.assertTrue((Path(folder) / "episodes.csv").exists())
            self.assertEqual([e["scenario"] for e in restored["episodes"]], scenes)
        leaked = deepcopy(plan)
        leaked["episodes"][-1]["scenario"]["split"] = "eval"
        with self.assertRaises(ValueError):
            validate_manifest(leaked)

    def test_abbreviated_family_plans_still_load_and_resume(self):
        import json

        from tasks.common.scenarios import seal
        from tasks.common.task import collection_settings

        cfg = BenchmarkConfig(task="relational_understanding", case="reference_frame", value=20)
        configure(cfg)
        current = cfg.scenario
        self.assertEqual(current["case_id"], "relational_understanding/reference_frame")
        # Frozen plans written before the rename store the abbreviated family.
        old = seal({**deepcopy(current), "task": "ru", "family": "ru", "case_id": "RU/reference_frame"})
        self.assertEqual(old["episode_id"], current["episode_id"])
        validate_manifest({"episodes": [{"episode_index": 0, "episode_id": old["episode_id"], "scenario": old}]})
        with TemporaryDirectory() as folder:
            path = Path(folder) / "scenario.json"
            path.write_text(json.dumps(old))
            restored = BenchmarkConfig(task="spatial_understanding/ru", scenario_file=str(path))
            configure(restored)
        self.assertEqual(restored.task, "relational_understanding")
        self.assertEqual(restored.scenario["fingerprint"], old["fingerprint"])
        self.assertEqual(collection_settings(old)["task"], restored.task)

    def test_tampered_geometry_prompt_and_persistent_scene_are_rejected(self):
        s = generate(BenchmarkConfig())
        for field, value in (("instruction", "Pick anything."), ("target_id", "missing"), ("scene_id", "changed")):
            broken = deepcopy(s)
            broken[field] = value
            with self.assertRaises(ValueError):
                validate(broken)
        with TemporaryDirectory() as folder:
            import json

            path = Path(folder) / "scenario.json"
            # Different geometry shares a launch; the worker rebuilds the stage in place.
            path.write_text(json.dumps(generate(BenchmarkConfig(seed=713))))
            configure(BenchmarkConfig(episode_files=[str(path)]))
            # A different launch configuration still needs its own worker.
            path.write_text(json.dumps(generate(BenchmarkConfig(seed=713, render_profile="balanced"))))
            with self.assertRaisesRegex(ValueError, "one launch configuration"):
                configure(BenchmarkConfig(episode_files=[str(path)]))


class WorkerRecoveryContracts(unittest.TestCase):
    def test_stale_shared_settings_are_rejected_before_collection_starts(self):
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from data_collection.manifest import collect_main
        from tasks.common.scenarios import seal

        scene = generate(BenchmarkConfig())
        scene["settings_digest"] = "previous-camera-settings"
        seal(scene)
        plan = dict(
            schema_version=3,
            benchmark="Benchmark",
            episodes=[dict(episode_index=0, episode_id=scene["episode_id"], scenario=scene)],
            infeasible_cells=[],
        )
        # Historical plans remain inspectable, but cannot run against new cameras.
        validate_manifest(plan)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_manifest(plan, root / "plan", {})
            output = root / "recordings"
            argv = ["collect_manifest.py", str(root / "plan/plan.yaml"), "--output", str(output)]
            stderr = io.StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch("data_collection.coordinator.collect", side_effect=AssertionError("stale plan reached workers")),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as error,
            ):
                collect_main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("different shared settings", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_lossless_completed_episode_survives_partial_next_file_and_recovers_once(self):
        from types import SimpleNamespace

        import h5py
        from data_collection.coordinator import _record_results
        from data_collection.finalize import finalize_attempt
        from data_collection.metadata import episode_info, recording_info, write_json_atomic
        from data_collection.recorder import EpisodeRecorder
        from data_collection.validate import validate_dataset
        from environments import settings

        planned = placement_plan(BenchmarkConfig(), episode_count=2)["episodes"]
        entries = {e["episode_id"]: e for e in planned}
        keys = list(entries)
        scene = planned[0]["scenario"]
        rng = np.random.default_rng(17)
        rgb = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        depth = rng.uniform(0.1, 1.2, (8, 8)).astype(np.float32)
        depth[0, 0] = np.nan
        with TemporaryDirectory() as folder:
            root = Path(folder)
            worker = root / "worker_000000"
            directory = worker / "data/run/episode_0000"
            writer = EpisodeRecorder(
                directory,
                episode_info("object_understanding", scene, 712, scene["target_index"]),
                directory / "dataset.hdf5",
                scene={"scenario": scene, "objects": scene["objects"]},
                recording=recording_info(settings, ["joint_0", "joint_1"], "full", "pick_place_v1"),
                compression="lzf",
            )
            sample = dict(
                frame=0,
                timestamp_s=0.0,
                rgb=rgb,
                intrinsics=np.eye(3),
                position=np.zeros(3),
                quaternion_ros_xyzw=np.array([0, 0, 0, 1]),
                distance_to_image_plane=depth,
                depth_valid=np.isfinite(depth),
            )
            writer.before(
                np.zeros(2),
                "approach",
                0.0,
                {"joint_position": np.zeros(2)},
                {view: sample for view in settings.CAMERAS},
            )
            writer.after({"joint_position": np.zeros(2)}, False)
            finalize_attempt(writer, False, "timeout", SimpleNamespace(video=False))
            damaged = worker / "data/run/episode_0001/dataset.hdf5"
            damaged.parent.mkdir()
            damaged.write_bytes(b"interrupted HDF5 write")
            write_json_atomic(worker / "assigned.json", {"episode_ids": keys, "attempt_id": 0})
            write_json_atomic(worker / "data/run/progress.json", {"episode": 1})
            histories = {key: [] for key in keys}
            directories = {key: root / f"planned_{i}" for i, key in enumerate(keys)}
            for _ in range(2):
                _record_results(worker, entries, histories, directories, final=True)
            self.assertEqual([len(histories[k]) for k in keys], [1, 1])
            self.assertTrue(histories[keys[0]][0]["finalized"])
            self.assertFalse(histories[keys[1]][0]["finalized"])
            self.assertEqual(histories[keys[1]][0]["failure_reason"], "incomplete_worker")
            with h5py.File(directory / "dataset.hdf5") as data:
                sensor = data["data/episode_0000/sensors/top"]
                np.testing.assert_array_equal(sensor["rgb"][0], rgb)
                np.testing.assert_array_equal(sensor["distance_to_image_plane"][0], depth)
            with h5py.File(directory / "dataset.hdf5", "r+") as data:
                del data["data/episode_0000/sensors/top/distance_to_image_plane"]
            with self.assertRaisesRegex(ValueError, "requires depth"):
                validate_dataset(directory / "dataset.hdf5")


class PlacementControllerContracts(unittest.TestCase):
    def test_supported_release_tolerates_contact_jitter_but_requires_safe_support(self):
        from controllers.pick_place import PickPlaceController
        from tasks.common.task import resolve_settings

        cfg = BenchmarkConfig(case="diameter", value=5, gap=0.4, shapes=["cylinder"])
        configure(cfg)
        settings = resolve_settings(cfg)
        motion = PickPlaceController(
            [0.1, 0.22, 0.8],
            0.05,
            goal_xy=PROTOCOL["goal_center_xy_m"],
            clearance_z=1.2,
            protocol=PROTOCOL,
            settings=settings,
        )
        hand = np.asarray([0.1, 0.22, 1.3, 1.0, 0.0, 0.0, 0.0])
        fingers = np.asarray([0.04, 0.04])
        for _ in range(round(20 / motion.control_dt)):
            goal, aperture, phase = motion.command(hand, fingers, np.zeros(2))
            if phase in ("lift", "transfer"):
                self.assertLessEqual(
                    np.linalg.norm(goal[:3] - hand[:3]),
                    settings.PLACEMENT_CONTROL["motion_speed_m_s"] * motion.control_dt + 1e-10,
                )
            hand = goal
            fingers = np.full(2, max(aperture, 0.025))
            if phase == "lower":
                break
        self.assertEqual(phase, "lower")
        # Supported pose with vertical/rotational contact jitter measured in stalled grasps.
        hand[:3] = [0.30, -0.15, 0.942]
        supported = (np.asarray([0.275, -0.175, 0.75]), np.asarray([0.325, -0.125, 0.85]))
        jitter = np.asarray([0.012, 0.0, 0.025, 0.6, 0.0, 0.0])
        suspended = tuple(bound + [0, 0, 0.05] for bound in supported)
        outside = tuple(bound + [0.3, 0, 0] for bound in supported)
        sliding = jitter.copy()
        sliding[0] = 0.1
        for bounds, velocity in [(suspended, jitter), (outside, jitter), (supported, sliding)]:
            for _ in range(12):
                _, aperture, phase = motion.command(
                    hand, fingers, np.zeros(2), object_bounds=bounds, object_velocity=velocity
                )
                self.assertEqual(phase, "lower")
                self.assertEqual(aperture, 0.0)
        for _ in range(12):
            _, aperture, phase = motion.command(
                hand, fingers, np.zeros(2), object_bounds=supported, object_velocity=jitter
            )
            if phase == "release":
                break
        self.assertEqual(phase, "release")
        self.assertEqual(aperture, 0.04)
        from environments.settings import validate_settings

        for invalid in (0, float("nan")):
            settings.PLACEMENT_CONTROL["motion_speed_m_s"] = invalid
            with self.assertRaisesRegex(ValueError, "PLACEMENT_CONTROL.motion_speed_m_s"):
                validate_settings(settings)


class PlacementContracts(unittest.TestCase):
    def setUp(self):
        self.initial = np.asarray([[0, 0.1, 0.79], [-0.2, 0.1, 0.79]])
        self.evaluator = PlacementEvaluator(self.initial, 0, 0.75)

    def observation(self, target=(0.30, -0.15, 0.79)):
        centres = self.initial.copy()
        centres[0] = target
        return {
            "object_pose_xyzw": np.column_stack((centres, np.tile([0, 0, 0, 1], (2, 1)))),
            "object_aabb_min": centres - [0.03, 0.04, 0.04],
            "object_aabb_max": centres + [0.03, 0.04, 0.04],
            "object_velocity": np.zeros((2, 6)),
            "hand_pose_xyzw": np.asarray([0.30, -0.15, 1.20, 1, 0, 0, 0]),
        }

    def lift(self):
        self.evaluator.update(self.observation((0, 0.1, 0.86)), [0.02, 0.02], "lift", 0.1)

    def test_lift_alone_and_opening_without_separation_cannot_succeed(self):
        self.lift()
        obs = self.observation()
        obs["hand_pose_xyzw"][:3] = [0.30, -0.15, 0.90]
        for _ in range(20):
            self.assertFalse(self.evaluator.update(obs, [0.04, 0.04], "release", 0.1))
        self.assertFalse(self.evaluator.metrics["released"])

    def test_release_support_containment_and_continuous_dwell(self):
        self.lift()
        valid = self.observation()
        for component, speed in ((0, 0.1), (2, 0.025), (3, 0.6)):
            for _ in range(4):
                self.assertFalse(self.evaluator.update(valid, [0.04, 0.04], "settled", 0.1))
            moving = deepcopy(valid)
            moving["object_velocity"][0, component] = speed
            self.assertFalse(self.evaluator.update(moving, [0.04, 0.04], "settled", 0.1))
            self.assertEqual(self.evaluator.dwell, 0)
        for _ in range(4):
            self.assertFalse(self.evaluator.update(valid, [0.04, 0.04], "settled", 0.1))
        self.assertTrue(self.evaluator.update(valid, [0.04, 0.04], "settled", 0.1))

    def test_suspended_or_overhanging_target_is_rejected(self):
        self.lift()
        for obs in (self.observation((0.30, -0.15, 0.80)), self.observation((0.405, -0.15, 0.79))):
            for _ in range(10):
                self.assertFalse(self.evaluator.update(obs, [0.04, 0.04], "settled", 0.1))

    def test_distractor_excursion_cannot_be_hidden_by_returning_it(self):
        self.lift()
        moved = self.observation()
        moved["object_pose_xyzw"][1, 0] += PROTOCOL["max_distractor_movement_m"] + 0.001
        self.evaluator.update(moved, [0.04, 0.04], "settled", 0.1)
        for _ in range(10):
            self.assertFalse(self.evaluator.update(self.observation(), [0.04, 0.04], "settled", 0.1))
        self.assertFalse(self.evaluator.metrics["distractors_preserved"])


class PredicateContracts(unittest.TestCase):
    def test_exact_clearance_layout_survives_candidate_permutation(self):
        from environments import settings

        for count, seed in ((6, 721), (7, 712)):
            with self.subTest(count=count, seed=seed):
                cfg = BenchmarkConfig(
                    task="relational_understanding",
                    case="directional_offset",
                    value=20,
                    gap=4,
                    candidate_count=count,
                    seed=seed,
                )
                configure(cfg)
                scene = cfg.scenario
                target = scene["objects"][scene["target_index"]]
                reference = next(o for o in scene["objects"] if o["object_id"] == "reference_red")
                self.assertAlmostEqual(target["position_m"][0] - reference["position_m"][0], 0.20)
                candidates = [o for o in scene["objects"] if o.get("selectable", True)]
                self.assertEqual(len(candidates), count)
                for obj in candidates:
                    radius = np.linalg.norm(np.asarray(obj["position_m"][:2]) - settings.ROBOT_POSITION[:2])
                    self.assertGreaterEqual(radius, 0.32)

    def test_reasoning_tolerance_does_not_change_with_distractor_gap(self):
        predicates = []
        for gap in (2, 3):
            cfg = BenchmarkConfig(
                task="quantitative_constraint_grounding", case="tolerance", value=10, gap=gap, tolerance=0.25
            )
            configure(cfg)
            predicates.append(cfg.scenario["predicate"])
        self.assertEqual(predicates[0], predicates[1])
        self.assertAlmostEqual(predicates[0]["tolerance_si"], 0.0025)

    def test_policy_evaluation_keeps_paraphrases_and_rejects_mismatched_protocol(self):
        from evaluation.offline.selection import score, summarize

        planned = create(BenchmarkConfig())["episodes"]
        identity = planned[0]["scenario"]["scenario_id"]
        episodes = [e for e in planned if e["scenario"]["scenario_id"] == identity]
        rows = []
        for entry in episodes:
            s = entry["scenario"]
            prediction = {"selected_object_id": s["target_id"], "source": "policy"}
            rows.append(score(s, prediction))
            self.assertTrue(
                score(s, dict(prediction, manipulation_success=True, success_protocol="pick_hold_v1"))[
                    "manipulation_success"
                ]
            )
            with self.assertRaisesRegex(ValueError, "pick_hold_v1"):
                score(s, dict(prediction, manipulation_success=True, success_protocol="pick_lift_hold_v1"))
        report = summarize(rows)
        self.assertEqual(len({r["episode_id"] for r in rows}), 3)
        self.assertEqual(report["paired_scenarios"], 1)
        self.assertEqual(report["paraphrase_selection_consistency"], 1)
        self.assertEqual(report["by_factor"]["candidate_count"]["3"]["count"], 3)
        self.assertIsNone(summarize([dict(r, source="oracle") for r in rows])["target_identification_accuracy"])

    def test_reference_frame_rotation_and_clearance_are_distinct_from_center_distance(self):
        obj = {
            "object_id": "candidate",
            "geometry": {"kind": "box", "size_m": [0.04, 0.04, 0.04]},
            "position_m": [0, 0.20, 0.02],
            "rotation_xyzw": [0, 0, 0, 1],
        }
        ref = dict(
            obj,
            object_id="reference_red",
            selectable=False,
            position_m=[0, 0, 0.02],
            rotation_xyzw=[0, 0, math.sqrt(0.5), math.sqrt(0.5)],
        )
        scene = {"objects": [obj, ref]}
        local = {"kind": "offset", "reference": "reference_red", "axis": 0, "frame": "reference"}
        self.assertAlmostEqual(quantity(obj, local, scene), 0.20)
        self.assertAlmostEqual(quantity(obj, dict(local, frame="table"), scene), 0)
        self.assertAlmostEqual(
            quantity(obj, {"kind": "distance", "reference": "reference_red", "planar": True}, scene), 0.20
        )
        self.assertAlmostEqual(quantity(obj, {"kind": "clearance", "reference": "reference_red"}, scene), 0.16)

    def test_threshold_endpoints_interval_tolerance_and_nearest_ties(self):
        from tasks.relational_understanding.scenarios import constant

        objects = [
            {"object_id": str(i), "geometry": {"kind": "box", "size_m": [0.06, 0.04, h]}}
            for i, h in enumerate((0.08, 0.10, 0.12))
        ]
        scene = {"objects": objects}
        predicate = {"op": "gt", "measure": {"kind": "attribute", "attribute": "height"}, "target": constant(10, "cm")}
        self.assertEqual([matches(o, predicate, scene) for o in objects], [False, False, True])
        predicate["op"] = "ge"
        self.assertEqual([matches(o, predicate, scene) for o in objects], [False, True, True])
        predicate.update(op="interval", bounds=[8, 10], bounds_si=[0.08, 0.10], closed=[False, True], unit="cm")
        self.assertEqual([matches(o, predicate, scene) for o in objects], [False, True, False])
        predicate = dict(
            op="tolerance",
            measure={"kind": "attribute", "attribute": "height"},
            target=constant(10, "cm"),
            tolerance=2,
            tolerance_si=0.02,
            unit="cm",
        )
        self.assertTrue(all(matches(o, predicate, scene) for o in objects))
        predicate["op"] = "closest"
        tied = {"objects": [objects[0], objects[2]]}
        self.assertFalse(any(matches(o, predicate, tied) for o in tied["objects"]))
        broken = deepcopy(predicate)
        broken["target"]["value_si"] = 10
        with self.assertRaisesRegex(ValueError, "units"):
            validate_predicate(broken, scene)

    def test_reasoning_distractors_match_only_part_of_a_conjunction(self):
        cfg = BenchmarkConfig(task="quantitative_constraint_grounding", case="multi_attribute", value=10, gap=2)
        configure(cfg)
        s = cfg.scenario
        terms = s["predicate"]["terms"]
        rows = [(matches(obj, terms[0], s), matches(obj, terms[1], s)) for obj in s["objects"]]
        self.assertCountEqual(rows, [(True, True), (True, False), (False, True)])
        for case in ("arithmetic", "ratio", "derived_property"):
            cfg = BenchmarkConfig(
                task="quantitative_constraint_grounding",
                case=case,
                value=200 if case == "derived_property" else 10,
                unit="cm3" if case == "derived_property" else "cm",
                gap=25 if case == "derived_property" else 2,
            )
            configure(cfg)
            s = cfg.scenario
            target = s["objects"][s["target_index"]]
            self.assertAlmostEqual(
                quantity(target, s["predicate"]["target"], s), 0.0002 if case == "derived_property" else 0.10
            )
            self.assertEqual(sum(matches(o, s["predicate"], s) for o in s["objects"] if o["selectable"]), 1)

    def test_versioned_recordings_share_geometry_but_keep_query_and_reset_state(self):
        import h5py
        from data_collection.metadata import episode_info, read_scene
        from data_collection.recorder import EpisodeRecorder
        from data_collection.validate import validate_dataset

        scenes = [e["scenario"] for e in placement_plan(BenchmarkConfig(), all_templates=False)["episodes"]]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for index, s in enumerate(scenes):
                realized = deepcopy(s["objects"])
                realized[0]["position_m"][0] += 0.0001 * index
                recorder = EpisodeRecorder(
                    root / f"episode_{index:04d}",
                    episode_info("object_understanding", s, 712, s["target_index"]),
                    root / "dataset.hdf5",
                    scene={"scenario": s, "objects": realized},
                )
                recorder.finish(False, "interrupted")
            self.assertEqual(validate_dataset(root / "dataset.hdf5")["episodes"], 3)
            with h5py.File(root / "dataset.hdf5") as data:
                self.assertEqual(data.attrs["schema_version"], 4)
                self.assertEqual(len(data["scene_specs"]), 1)
                for index, s in enumerate(scenes):
                    restored = read_scene(data[f"data/episode_{index:04d}"])
                    self.assertEqual(restored["scenario"], s)
                    self.assertAlmostEqual(
                        restored["objects"][0]["position_m"][0], s["objects"][0]["position_m"][0] + 0.0001 * index
                    )


if __name__ == "__main__":
    unittest.main()
