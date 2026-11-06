# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit intervention experiments; never used in ordinary demonstrations."""

from copy import deepcopy

from assets.geometry import world_aabb


def intervene(scenario, motion, evaluator, objects, observation):
    """Shift released A once before planning B, and retain the intervention evidence."""
    if evaluator.intervention_applied or motion.index != 1 or motion.child is not None:
        return observation
    import torch

    result = deepcopy(observation)
    pose = result["object_pose_xyzw"][0].copy()
    pose[:2] += scenario["diagnostic_intervention"]["offset_xy_m"]
    obj = objects[0]
    obj.write_root_pose_to_sim_index(root_pose=torch.tensor(pose, dtype=torch.float32, device=obj.device)[None])
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=obj.device))
    result["object_pose_xyzw"][0] = pose
    result["object_velocity"][0] = 0.0
    result["object_aabb_min"][0], result["object_aabb_max"][0] = world_aabb(
        scenario["objects"][0]["geometry"], pose[:3], pose[3:]
    )
    evaluator.intervention_applied = True
    return result
