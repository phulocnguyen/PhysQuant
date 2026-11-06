# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged scripted control used only as an evaluation reference."""


def apply_targets(w, motion, before, scenario, targets):
    """The scripted controller's joint targets, computed as during data collection."""
    import torch
    from tasks.registry import engine

    target = scenario["target_index"]
    goal, finger, phase = motion.command(
        before["hand_pose_xyzw"],
        before["joint_position"][w.finger_ids],
        before["joint_velocity"][w.finger_ids],
        **(
            {}
            if scenario["manipulation_protocol"] == "pick_hold_v1"
            else dict(
                object_bounds=(before["object_aabb_min"][target], before["object_aabb_max"][target]),
                object_velocity=before["object_velocity"][target],
            )
        ),
        **(
            {"object_pose": before["object_pose_xyzw"][target]}
            if engine(scenario) == "control"
            else {"observation": before}
            if engine(scenario) == "multi_object"
            else {}
        ),
    )
    robot = w.robot
    w.ik.set_command(torch.tensor(goal, dtype=torch.float32, device=w.sim.device).unsqueeze(0))
    hand = robot.data.body_pose_w.torch[:, w.hand_id]
    jacobian = robot.data.body_link_jacobian_w.torch[:, w.jacobian_id, :, w.jacobian_columns]
    desired = w.ik.compute(hand[:, :3], hand[:, 3:], jacobian, robot.data.joint_pos.torch[:, w.arm_ids])
    limits = robot.data.soft_joint_pos_limits.torch[:, w.arm_ids]
    targets[:, w.arm_ids] = torch.maximum(torch.minimum(desired, limits[..., 1]), limits[..., 0])
    targets[:, w.finger_ids] = torch.as_tensor(finger, dtype=targets.dtype, device=targets.device)
    return phase
