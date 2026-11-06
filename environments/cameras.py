# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed and hand-mounted camera poses and synchronized RGB-D capture."""

import numpy as np
import torch

from isaaclab.sensors import SensorBase
from isaaclab.utils.math import combine_frame_transforms


class CameraRig:
    """Own the four sensor references and hand-relative camera mounts."""

    def __init__(self, scene, robot, sim, config, *, record_depth):
        self.robot = robot
        self.sim = sim
        self.record_depth = record_depth
        self.cameras = {view: scene[f"camera_{view}"] for view in config.CAMERAS}
        self.mounts = []
        for view, camera in self.cameras.items():
            pose = config.CAMERAS[view]
            if "body" in pose:
                body_ids, _ = robot.find_bodies(pose["body"])
                if len(body_ids) != 1:
                    raise ValueError(f"Camera {view} must reference exactly one robot body")
                self.mounts.append(
                    (
                        camera,
                        body_ids[0],
                        torch.tensor([pose["offset_position"]], dtype=torch.float32, device=sim.device),
                        torch.tensor([pose["offset_rotation_xyzw"]], dtype=torch.float32, device=sim.device),
                    )
                )
                continue
            camera.set_world_poses_from_view(
                torch.tensor([pose["position"]], device=sim.device),
                torch.tensor([pose["target"]], device=sim.device),
            )

    def sync(self):
        for camera, body_id, offset, rotation in self.mounts:
            body_pose = self.robot.data.body_pose_w.torch[:, body_id]
            position, orientation = combine_frame_transforms(
                body_pose[:, :3],
                body_pose[:, 3:],
                offset,
                rotation,
            )
            camera.set_world_poses(position, orientation, convention="ros")

    def capture(self, time_s):
        # Synchronous RTX rendering snapshots state_t before action_t.
        self.sync()
        self.sim.render()
        # Share scene synchronization and renderer submission across all views.
        SensorBase.update_batch(tuple(self.cameras.values()), 0.0)
        samples = {}
        for view, camera in self.cameras.items():
            data = camera.data
            sample = {
                "frame": int(camera.frame.torch[0].item()),
                "timestamp_s": time_s,
                # A sample can wait until the next action tick before recording.
                # CPU-backed sensor buffers may change in the intervening steps.
                "rgb": data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy(),
                "intrinsics": data.intrinsic_matrices.torch[0].cpu().numpy().copy(),
                "position": data.pos_w.torch[0].cpu().numpy().copy(),
                "quaternion_ros_xyzw": data.quat_w_ros.torch[0].cpu().numpy().copy(),
            }
            if self.record_depth:
                depth = data.output["distance_to_image_plane"].torch[0].cpu().numpy().copy()
                sample["distance_to_image_plane"] = depth
                sample["depth_valid"] = np.isfinite(depth) & (depth > 0)
            samples[view] = sample
        return samples
