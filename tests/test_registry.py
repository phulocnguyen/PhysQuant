# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subgroup task IDs, their packages and configurations, and legacy aliases."""

import sys
import unittest
from importlib import import_module
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tasks.registry import ENGINES, SUBGROUPS, engine, get_task, scenario_subgroup, task_id


class RegistryTest(unittest.TestCase):
    def test_each_subgroup_owns_a_package_and_its_configurations(self):
        for name, value in SUBGROUPS.items():
            with self.subTest(subgroup=name):
                import_module(f"tasks.{name}")
                paths = sorted((ROOT / "configs/tasks" / name).glob("*.yaml"))
                self.assertTrue(paths)
                self.assertEqual({yaml.safe_load(path.read_text())["task"] for path in paths}, {name})
                self.assertIs(get_task(name), import_module(ENGINES[value["engine"]]))

    def test_legacy_ids_and_scenario_fields_resolve_to_subgroups(self):
        cases = {
            "spatial_understanding/ou": "object_understanding",
            "spatial_understanding/spatial_reasoning": "quantitative_constraint_grounding",
            "spatial_understanding/sr": "quantitative_constraint_grounding",
            "spatial_control/object_rotation": "spatial_transformation",
            "spatial_control/clearance_placement": "target_placement",
            "spatial_control/constrained_transport": "constrained_motion",
            "long_horizon/numerical_sorting": "sequencing_allocation_planning",
            "long_horizon/relational_construction": "constructive_planning",
            "long_horizon/state_dependent_planning": "feedback_driven_planning",
        }
        for legacy, current in cases.items():
            with self.subTest(legacy=legacy):
                self.assertEqual(task_id(legacy), current)
                self.assertEqual(task_id(current), current)
        frozen = {"category": "spatial_understanding", "task": "spatial_reasoning", "case": "ratio"}
        self.assertEqual(engine(frozen), "selection")
        self.assertEqual(scenario_subgroup(frozen), "quantitative_constraint_grounding")
        frozen = {"category": "long_horizon", "task": "sequential_ordering", "case": "SAP-1A"}
        self.assertEqual(
            (engine(frozen), scenario_subgroup(frozen)), ("multi_object", "sequencing_allocation_planning")
        )
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            get_task("spatial_reasoning_v2")


if __name__ == "__main__":
    unittest.main()
