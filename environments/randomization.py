# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded reset randomization using the ExtrinsicDexterity factor schema."""

from copy import deepcopy

import numpy as np

DEFAULTS = {
    "enabled": False,
    "camera": {"enabled": False, "position_range_m": 0.05, "fov_range_deg": 5.0},
    "light": {"enabled": False, "intensity_scale_range": [0.5, 1.5]},
    "table": {"enabled": False, "options": []},
    "target_object": {
        "enabled": False,
        "position_range_m": 0.10,
        "yaw_range_deg": 180.0,
        "mass_range_kg": [0.01, 0.25],
        "com_range_m": 0.005,
    },
    "robot": {"enabled": False, "joint_position_range_rad": 0.09, "joint_velocity_range_rad_s": 0.0},
}


def resolve_randomization(settings):
    """Validate switches and ranges, rejecting unknown factor names."""
    if not isinstance(settings, dict) or set(settings) - set(DEFAULTS):
        raise ValueError("randomization supports enabled, camera, light, table, target_object, robot")
    result = deepcopy(DEFAULTS)
    for name, value in settings.items():
        if name == "enabled":
            result[name] = value
        else:
            if not isinstance(value, dict) or set(value) - set(DEFAULTS[name]):
                raise ValueError(f"Unknown randomization.{name} fields")
            result[name].update(value)
    for group in [result, *(result[name] for name in DEFAULTS if name != "enabled")]:
        if not isinstance(group["enabled"], bool):
            raise ValueError("Randomization enabled switches must be boolean")
        for key, value in group.items():
            if key == "enabled" or isinstance(value, dict):
                continue
            if key == "options":
                if not isinstance(value, list) or group["enabled"] and not value:
                    raise ValueError("Enabled table randomization requires a nonempty options list")
                names = set()
                for option in value:
                    if (
                        not isinstance(option, dict)
                        or set(option) != {"name", "color_rgb", "source_texture"}
                        or not isinstance(option["name"], str)
                        or not option["name"]
                        or option["name"] in names
                        or not isinstance(option["source_texture"], str)
                    ):
                        raise ValueError("Table options need unique names, color_rgb, and source_texture")
                    color = option["color_rgb"]
                    if (
                        not isinstance(color, (list, tuple))
                        or len(color) != 3
                        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in color)
                        or not np.all(np.isfinite(color))
                        or np.any(np.asarray(color) < 0)
                        or np.any(np.asarray(color) > 1)
                    ):
                        raise ValueError("Table color_rgb must contain three finite values in [0, 1]")
                    names.add(option["name"])
                continue
            values = value if isinstance(value, (list, tuple)) else [value]
            if (
                any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)
                or not np.all(np.isfinite(values))
                or np.any(np.asarray(values) < 0)
            ):
                raise ValueError(f"{key} must contain finite nonnegative numbers")
            if key in ("mass_range_kg", "intensity_scale_range"):
                if len(values) != 2 or values[0] <= 0 or values[1] < values[0]:
                    raise ValueError(f"{key} must be an ordered positive [min, max]")
            elif len(values) != 1 or isinstance(value, (list, tuple)):
                raise ValueError(f"{key} must be a scalar")
    return result


def sample_table_style(settings, seed):
    """Uniform seeded selection, independent of the other randomization streams."""
    cfg = resolve_randomization(settings)
    if not cfg["enabled"] or not cfg["table"]["enabled"]:
        return None
    options = cfg["table"]["options"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    return deepcopy(options[int(rng.integers(len(options)))])


def sample_episode(specs, settings, *, seed, table_size, colors=None, reserved_x_min=None):
    """Sample object offsets about nominal poses, rejecting invalid footprints."""
    cfg = resolve_randomization(settings)
    result = deepcopy(specs)
    group = cfg["target_object"]
    if not cfg["enabled"] or not group["enabled"]:
        return result
    rng = np.random.default_rng(seed)
    nominal = np.asarray([obj["position_m"] for obj in specs], dtype=float)
    rotations = np.asarray([obj["rotation_xyzw"] for obj in specs], dtype=float)
    nominal_yaw = np.arctan2(
        2 * (rotations[:, 3] * rotations[:, 2] + rotations[:, 0] * rotations[:, 1]),
        1 - 2 * (rotations[:, 1] ** 2 + rotations[:, 2] ** 2),
    )
    sizes = np.asarray([obj["size_m"] for obj in specs])
    for _ in range(1000):
        yaw = nominal_yaw + rng.uniform(-group["yaw_range_deg"], group["yaw_range_deg"], len(specs)) * np.pi / 180
        xy = nominal[:, :2] + rng.uniform(-group["position_range_m"], group["position_range_m"], (len(specs), 2))
        c, s = np.abs(np.cos(yaw)), np.abs(np.sin(yaw))
        half = np.column_stack((c * sizes[:, 0] + s * sizes[:, 1], s * sizes[:, 0] + c * sizes[:, 1])) / 2
        for i, obj in enumerate(specs):
            if obj.get("shape") in ("cylinder", "sphere") and obj.get("geometry", {}).get("axis", "Z") == "Z":
                half[i] = obj["radius_m"]
        if reserved_x_min is not None and np.any(xy[:, 0] + half[:, 0] >= reserved_x_min):
            continue
        if np.any(np.abs(xy) + half > np.asarray(table_size[:2]) / 2):
            continue
        if any(np.all(np.abs(xy[i] - xy[j]) < half[i] + half[j] + 0.005) for i in range(len(specs)) for j in range(i)):
            continue
        break
    else:
        raise ValueError("No valid randomized placement after 1000 attempts; reduce ranges or object count")
    for i, obj in enumerate(result):
        obj["position_m"] = [*xy[i].tolist(), float(nominal[i, 2])]
        obj["rotation_xyzw"] = [0.0, 0.0, float(np.sin(yaw[i] / 2)), float(np.cos(yaw[i] / 2))]
        obj["mass_kg"] = float(rng.uniform(*group["mass_range_kg"]))
        span = group["com_range_m"]
        if span >= min(obj["size_m"]) / 2:
            raise ValueError(f"{obj['name']}: com_range_m must be smaller than every half-extent")
        obj["com_offset_m"] = rng.uniform(-span, span, 3).tolist()
    return result


class EpisodeRandomizer:
    """Apply reset samples about captured nominal values without accumulating drift."""

    def __init__(self, cfg, objects, cameras, camera_settings, *, appearance_specs=None):
        import warp as wp

        from pxr import Usd, UsdLux, UsdShade

        from isaaclab.sim.utils.stage import get_current_stage

        self.cfg = resolve_randomization(cfg)
        self.objects = objects
        self.cameras = cameras
        self.camera_settings = camera_settings
        self.nominal_physics = []
        if self.cfg["enabled"] and self.cfg["target_object"]["enabled"]:
            for obj in objects:
                view = obj.root_view
                # PhysX views use Warp arrays; copy through Torch to retain nominal values.
                self.nominal_physics.append(
                    tuple(
                        wp.to_torch(value).to(obj.device).clone()
                        for value in (view.get_masses(), view.get_inertias(), view.get_coms())
                    )
                )
        self.nominal_cameras = {
            name: (
                camera.data.pos_w.torch.clone(),
                camera.data.quat_w_ros.torch.clone(),
                camera.data.intrinsic_matrices.torch.clone(),
            )
            for name, camera in cameras.items()
            if "body" not in camera_settings[name]
        }
        self.lights = []
        stage = get_current_stage()
        self.object_color_inputs = {}
        for spec in appearance_specs or []:
            if not spec.get("selectable", True):
                continue
            root = stage.GetPrimAtPath(f"/World/envs/env_0/{spec['name']}")
            inputs = []
            for prim in Usd.PrimRange(root):
                if prim.GetPath().HasPrefix(root.GetPath().AppendChild("AxisMarker")):
                    continue  # Preserve the observable local-axis cue on candidates.
                if prim.IsA(UsdShade.Shader):
                    color_input = UsdShade.Shader(prim).GetInput("diffuseColor")
                    if color_input:
                        inputs.append(color_input)
            if not inputs:
                raise RuntimeError(f"Candidate {spec['name']} has no editable diffuseColor material")
            self.object_color_inputs[spec["name"]] = inputs
        self.stage = stage
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                attr = UsdLux.LightAPI(prim).GetIntensityAttr()
                self.lights.append((attr, float(attr.Get())))

    def apply(self, specs, robot, arm_ids, seed, *, seed_streams=None, appearance_seed=None, appearance=None):
        """Apply camera/light/inertia samples and return robot state plus annotations."""
        import torch

        rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        streams = seed_streams or {}
        camera_rng = np.random.default_rng(streams["camera"]) if streams else rng
        light_rng = np.random.default_rng(streams["appearance"]) if streams else rng
        robot_rng = np.random.default_rng(streams["robot"]) if streams else rng
        cfg = self.cfg
        samples = {"camera": {}, "light": []}
        if self.object_color_inputs:
            from pxr import Gf

            samples["candidate_colors"] = {}
            for spec in specs:
                for color_input in self.object_color_inputs.get(spec["name"], []):
                    color_input.Set(Gf.Vec3f(*spec["color_rgb"]))
                if spec["name"] in self.object_color_inputs:
                    samples["candidate_colors"][spec["name"]] = list(spec["color_rgb"])
        from environments.appearance import apply_scene_appearance

        samples.update(
            apply_scene_appearance(
                self.stage,
                0,
                appearance,
                cfg,
                streams.get("appearance", seed) if appearance_seed is None else appearance_seed,
            )
        )
        if cfg["enabled"] and cfg["target_object"]["enabled"]:
            for obj, spec, (mass, inertia, com) in zip(self.objects, specs, self.nominal_physics):
                new_mass = torch.full_like(mass, spec["mass_kg"])
                obj.set_masses_index(masses=new_mass.reshape(1, 1))
                obj.set_inertias_index(inertias=(inertia * (new_mass / mass)[..., None]).reshape(1, 1, 9))
                new_com = com.clone().reshape(1, 1, 7)
                new_com[..., :3] += torch.tensor(spec["com_offset_m"], dtype=new_com.dtype, device=new_com.device)
                obj.set_coms_index(coms=new_com)
        if cfg["enabled"] and cfg["camera"]["enabled"]:
            group = cfg["camera"]
            for name, (pos, quat, nominal_k) in self.nominal_cameras.items():
                camera = self.cameras[name]
                camera.cfg.update_latest_camera_pose = True
                offset = camera_rng.uniform(-group["position_range_m"], group["position_range_m"], (1, 3))
                camera.set_world_poses(
                    pos + torch.tensor(offset, device=pos.device, dtype=pos.dtype), quat, convention="ros"
                )
                k = nominal_k.clone()
                width = camera.image_shape[1]
                nominal_fov = 2 * torch.atan(width / (2 * k[:, 0, 0]))
                delta = float(camera_rng.uniform(-group["fov_range_deg"], group["fov_range_deg"]))
                fov = (nominal_fov + np.deg2rad(delta)).clamp(np.deg2rad(5), np.deg2rad(170))
                factor = (width / (2 * torch.tan(fov / 2))) / k[:, 0, 0]
                k[:, 0, 0] *= factor
                k[:, 1, 1] *= factor
                camera.set_intrinsic_matrices(k)
                samples["camera"][name] = {"position_offset_m": offset[0].tolist(), "fov_offset_deg": delta}
        if cfg["enabled"] and cfg["light"]["enabled"]:
            for attr, nominal in self.lights:
                scale = float(light_rng.uniform(*cfg["light"]["intensity_scale_range"]))
                attr.Set(nominal * scale)
                samples["light"].append({"prim_path": str(attr.GetPrim().GetPath()), "intensity_scale": scale})
        positions = robot.data.default_joint_pos.torch.clone()
        velocities = robot.data.default_joint_vel.torch.clone()
        if cfg["enabled"] and cfg["robot"]["enabled"]:
            group = cfg["robot"]
            for state, key in ((positions, "joint_position_range_rad"), (velocities, "joint_velocity_range_rad_s")):
                offset = robot_rng.uniform(-group[key], group[key], (1, len(arm_ids)))
                state[:, arm_ids] += torch.tensor(offset, device=state.device, dtype=state.dtype)
            limits = robot.data.soft_joint_pos_limits.torch[:, arm_ids]
            positions[:, arm_ids] = positions[:, arm_ids].clamp(limits[..., 0], limits[..., 1])
            limits_v = robot.data.joint_vel_limits.torch[:, arm_ids]
            velocities[:, arm_ids] = velocities[:, arm_ids].clamp(-limits_v, limits_v)
        samples["robot_reset_joint_position"] = positions[0].cpu().tolist()
        samples["robot_reset_joint_velocity"] = velocities[0].cpu().tolist()
        return positions, velocities, samples
