# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy recordings used by recovery and storage contract tests."""

from copy import deepcopy

from tasks.common.manifest import create
from tasks.common.placement import PROTOCOL
from tasks.common.scenarios import seal, validate


def placement_plan(config, **kwargs):
    """Freeze the previous placement contract, independent of new task defaults."""
    plan = create(config, **kwargs)
    for entry in plan["episodes"]:
        scene = entry["scenario"]
        scene.update(goal=deepcopy(PROTOCOL), manipulation_protocol=PROTOCOL["id"], template_version=2)
        seal(scene)
        validate(scene)
        entry["episode_id"] = scene["episode_id"]
    return plan
