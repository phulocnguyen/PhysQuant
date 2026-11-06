# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation suites (one per task subgroup), generalization tiers, and frozen episodes.

A suite tier is a frozen plan written by ``generate_manifest.py`` under
``<manifests>/<suite>/<tier>/plan.yaml``. Every evaluated policy sees the same
scenes, targets, instructions, backgrounds, and initial states.
"""

from pathlib import Path

SUITES = (
    "object_understanding",
    "relational_understanding",
    "quantitative_constraint_grounding",
    "spatial_transformation",
    "target_placement",
    "constrained_motion",
    "sequencing_allocation_planning",
    "constructive_planning",
    "feedback_driven_planning",
)
SELECTION_SUITES = SUITES[:3]
TIERS = {
    "t1_in_distribution": "New layouts, targets, and appearances; training values and wording",
    "t2_held_out_language": "Instruction templates never used for training data",
    "t3_unseen_values": "Measurements shifted off the training value grid (selection suites)",
    "t5_more_candidates": "8 or 9 candidates; training used 3 to 7 (selection suites)",
}
# The robot state is joint positions (7 arm joints, 2 fingers). Multi-object suites append a stop request.
STATE_DIM = 9
STOP_SUITES = SUITES[6:]


def action_dim(suite):
    return STATE_DIM + (suite in STOP_SUITES)


def plan_path(bundle, suite, tier):
    return Path(bundle) / suite / tier / "plan.yaml"


def available(bundle):
    """(suite, tier) pairs present in a bundle."""
    return [(suite, tier) for suite in SUITES for tier in TIERS if plan_path(bundle, suite, tier).exists()]


def load_episodes(bundle, suite, tier):
    """Frozen, validated scenarios of one suite tier, in plan order."""
    from data_collection.manifest import load_manifest, validate_manifest

    path = plan_path(bundle, suite, tier)
    if not path.exists():
        raise FileNotFoundError(f"Suite {suite}/{tier} is not in this bundle: {path}")
    manifest = load_manifest(path)
    validate_manifest(manifest)
    return manifest["episodes"]
