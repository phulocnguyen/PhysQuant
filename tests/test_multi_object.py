# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-contract checks; synthetic state sequences are not physical qualification."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assets.geometry import world_aabb
from data_collection.config import load_config
from tasks.common.scenarios import ROOT
from tasks.multi_object.evaluator import MultiObjectEvaluator
from tasks.multi_object.manifest import create
from tasks.multi_object.planning import destination
from tasks.multi_object.scenarios import CASES, validate
from tasks.multi_object.task import MultiObjectConfig, configure
from tasks.registry import subgroup_of_case


def scene(case):
    cfg = load_config(ROOT / "configs/tasks" / subgroup_of_case(case) / f"{case}.yaml", {}, MultiObjectConfig)
    configure(cfg)
    return cfg.scenario


class Trace:
    def __init__(self, s):
        self.scene = s
        self.poses = np.array([o["position_m"] + o["rotation_xyzw"] for o in s["objects"]])
        self.evaluator = MultiObjectEvaluator(s, self.observe())

    def observe(self, i=None, closed=False):
        bounds = [world_aabb(o["geometry"], p[:3], p[3:]) for o, p in zip(self.scene["objects"], self.poses)]
        hand = np.array([0.0, -0.4, 1.4]) if not closed else self.poses[i, :3] + [0, 0, 0.1034]
        return dict(
            object_pose_xyzw=self.poses.copy(),
            object_velocity=np.zeros((len(self.poses), 6)),
            object_aabb_min=np.array([b[0] for b in bounds]),
            object_aabb_max=np.array([b[1] for b in bounds]),
            hand_pose_xyzw=np.r_[hand, [1, 0, 0, 0]],
        )

    def feed(self, i=None, closed=False, stop=False):
        obs = self.observe(i, closed)
        for _ in range(65):
            result = self.evaluator.update(
                obs, [0.02, 0.02] if closed else [0.04, 0.04], "stop" if stop else "untrusted_label", 1 / 120
            )
        return result

    def place(self, i, point):
        self.poses[i, 2] += 0.10
        self.feed(i, closed=True)
        self.poses[i, :3] = point
        self.feed()


class LongHorizonContracts(unittest.TestCase):
    def test_all_families_solve_and_reject_changed_numerical_ground_truth(self):
        for case in CASES:
            with self.subTest(case=case):
                s = scene(case)
                validate(s)
                self.assertTrue(s["plan"]["solutions"])
                changed = deepcopy(s)
                changed["plan"]["solutions"] = []
                with self.assertRaisesRegex(ValueError, "solution enumeration"):
                    validate(changed)

    def test_basket_contents_do_not_erase_wrong_order_and_stop_is_required(self):
        s = scene("SAP-1A")
        for order, expected in (([1, 0, 2], True), ([0, 1, 2], False)):
            trace = Trace(s)
            for slot, i in enumerate(order):
                point, _ = destination(s, i, 0, slot, trace.observe(), [])
                trace.place(i, point)
            self.assertFalse(trace.feed(stop=False))
            self.assertEqual(trace.feed(stop=True), expected)
            self.assertEqual(len(trace.evaluator.events), 3)
            self.assertTrue(trace.evaluator.metrics["numerical_success"])

    def test_alternative_component_combination_accepted_but_floating_row_rejected(self):
        s = scene("CP-1C")
        trace = Trace(s)
        self.assertGreater(len(s["plan"]["solutions"]), 1)
        self.assertNotEqual(s["plan"]["solutions"][0][0], [1, 2])
        trace.place(1, [-0.10, 0.20, 0.775])
        trace.place(2, [-0.01, 0.20, 0.775])
        self.assertTrue(trace.feed(stop=True))
        trace.poses[2, 2] += 0.02
        self.assertFalse(trace.feed(stop=True))

    def test_center_inside_does_not_prove_basket_containment_or_support(self):
        s = scene("SAP-1A")
        trace = Trace(s)
        b = s["baskets"][0]
        point, _ = destination(s, 0, 0, 0, trace.observe(), [])
        trace.place(0, point)
        self.assertTrue(trace.evaluator.contained(0, 0, trace.observe()))
        trace.poses[0, :2] = np.asarray(b["center_xy_m"]) + [0.10, 0]
        self.assertFalse(trace.evaluator.contained(0, 0, trace.observe()))
        trace.poses[0, :2] = b["center_xy_m"]
        trace.poses[0, 2] += 0.03
        self.assertFalse(trace.evaluator.supported(0, trace.observe(), 0))

    def test_feedback_goal_uses_achieved_reference_and_includes_surface_extents(self):
        for case, expected_dx in (("FDP-2A", 0.10), ("FDP-2B", 0.11)):
            s = scene(case)
            trace = Trace(s)
            trace.poses[0, :3] = [-0.10, 0.20, 0.77]
            first, _ = destination(s, 1, 0, 1, trace.observe(), [])
            self.assertAlmostEqual(first[0], -0.10 + expected_dx)
            trace.poses[0, 0] += 0.012
            second, _ = destination(s, 1, 0, 1, trace.observe(), [])
            self.assertAlmostEqual(second[0] - first[0], 0.012)
            if case == "FDP-2B":
                # A 90-degree turn changes A's X extent from 5 to 4 cm.
                # The same surface gap therefore moves B's center by -5 mm.
                trace.poses[0, 3:] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
                turned, _ = destination(s, 1, 0, 1, trace.observe(), [])
                self.assertAlmostEqual(turned[0] - second[0], -0.005)

    def test_numeric_sweep_preserves_scene_but_changes_required_subset(self):
        cfg = load_config(ROOT / "configs/tasks/feedback_driven_planning/FDP-1A.yaml", {}, MultiObjectConfig)
        cfg.problem_index = 0
        cfg.goal_sweep = [dict(measurement="length", target=20), dict(measurement="length", target=18)]
        episodes = create(cfg, all_templates=False)["episodes"]
        a, b = [e["scenario"] for e in episodes]
        self.assertEqual(a["scene_id"], b["scene_id"])
        self.assertEqual(a["objects"], b["objects"])
        self.assertEqual(a["appearance"], b["appearance"])
        self.assertNotEqual(a["plan"]["solutions"], b["plan"]["solutions"])
        self.assertNotEqual(a["episode_id"], b["episode_id"])

    def test_intervention_requires_adaptation_after_the_released_reference_moves(self):
        cfg = load_config(ROOT / "configs/tasks/feedback_driven_planning/FDP-2C.yaml", {}, MultiObjectConfig)
        cfg.diagnostic_intervention = dict(offset_xy_m=[0.025, 0.0])
        configure(cfg)
        for adaptive in (True, False):
            trace = Trace(cfg.scenario)
            first, _ = destination(cfg.scenario, 0, 0, 0, trace.observe(), [])
            trace.place(0, first)
            trace.poses[0, 0] += 0.025
            trace.evaluator.intervention_applied = True
            for i in (1, 2):
                point, _ = destination(cfg.scenario, i, 0, i, trace.observe(), [])
                if i == 1 and not adaptive:
                    point[0] -= 0.025
                trace.place(i, point)
            self.assertEqual(trace.feed(stop=True), adaptive)


if __name__ == "__main__":
    unittest.main()
