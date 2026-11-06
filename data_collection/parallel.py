# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Advance many independent episodes through one cloned scene and shared physics steps.

Adapted from the ExtrinsicDexterity parallel collector. The scene is cloned into
``num_envs`` environments. Each occupied slot runs its own controller, evaluator,
and recorder; all slots share one physics step and one render per capture. Finished
episodes are saved immediately, and the slots reset together after the batch.

Episodes share a clone when their objects have the same physical signatures. Spec
order may differ: each episode's objects are mapped onto prims with matching
geometry. Observations are expressed in each environment's own table frame, so
recordings follow the single-environment contract.
"""

import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
from tasks.common.scenarios import digest
from tasks.registry import engine

from data_collection.metadata import episode_info, recording_info, save_run_config, write_json_atomic

# Object fields that must match for two objects to share a cloned prim.
SIGNATURE_FIELDS = (
    "geometry",
    "kinematic",
    "asset_usd",
    "contact_offset_m",
    "marked_local_axis",
    "mass_kg",
    "selectable",
)


def object_signature(obj):
    """Physical identity of one object; selectable colors are applied per environment."""
    signature = {key: obj.get(key) for key in SIGNATURE_FIELDS}
    if not obj.get("selectable", True):
        signature["color_rgb"] = obj.get("color_rgb")
    return json.dumps(signature, sort_keys=True)


def clone_key(scenario):
    """Scenes with equal keys can share one cloned stage, whatever their object order."""
    return digest(
        {
            "objects": sorted(object_signature(obj) for obj in scenario["objects"]),
            "reference_frame": scenario.get("reference_frame"),
            "goal_zone": scenario["goal"] if scenario["manipulation_protocol"] == "pick_place_v1" else None,
            "mutable_appearance": bool(scenario.get("appearance")),
            # Material topology and studio-floor visibility must agree across a batch.
            "appearance_version": scenario.get("appearance", {}).get("version", 1),
            "background_enabled": bool(scenario.get("appearance", {}).get("background")),
        }
    )


def prim_order(template, specs):
    """Return, for each spec, the index of the template prim with the same signature."""
    available = {}
    for index, obj in enumerate(template):
        available.setdefault(object_signature(obj), []).append(index)
    order = []
    for obj in specs:
        slots = available.get(object_signature(obj))
        if not slots:
            raise ValueError("Episode objects do not match the cloned scene")
        order.append(slots.pop(0))
    return order


def check_parallel_support(run_config, scenarios):
    """Reject options whose state is global or single-environment in this collector."""
    cfg = run_config.randomization
    if cfg.get("enabled"):
        for group in ("camera", "light", "robot", "target_object"):
            if cfg[group]["enabled"]:
                raise ValueError(f"Parallel collection does not support {group} randomization; use num_envs=1")
    if any(s.get("diagnostic_intervention") for s in scenarios):
        raise ValueError("Diagnostic interventions require num_envs=1")


def run_parallel(args, run_config, *, config, scenarios):
    """Collect ``scenarios`` with up to ``run_config.num_envs`` cloned environments per batch."""
    from core.resources import configure_worker
    from environments.simulation import configure_stage, release_scene, scene_definition
    from isaaclab_physx.physics import PhysxCfg

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils import instantiate

    from data_collection.simulation import capture_steps

    if args.preview:
        raise ValueError("Preview uses one environment; use num_envs=1")
    check_parallel_support(run_config, scenarios)
    if config.OBJECT_START_ASLEEP:
        args.device = "cpu"
    capture_physics_steps = capture_steps(run_config, config)
    record_depth = args.capture_profile == "full"
    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = save_run_config(run_dir, run_config, config, args.output_dir)
    print(f"Dataset: {run_dir}", flush=True)

    # Contiguous runs of equal clone keys keep the original episode order.
    groups = []
    for index, scenario in enumerate(scenarios):
        key = clone_key(scenario)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(index)
        else:
            groups.append((key, [index]))

    started = time.monotonic()
    sim_cfg = sim_utils.SimulationCfg(dt=config.PHYSICS_DT, device=args.device, physics=PhysxCfg())
    from environments.appearance import background_spacing

    spacing = 4.0

    def definition_for(template_scenario, num_envs):
        definition = scene_definition(
            template_scenario["objects"], template_scenario, config, record_depth=record_depth, num_envs=num_envs
        )
        definition.env_spacing = spacing
        return definition

    first = definition_for(scenarios[groups[0][1][0]], min(run_config.num_envs, len(groups[0][1])))
    summary = {"attempts": 0, "successes": 0, "failures": 0, "completed": False}
    summary.update(simulator_launch_count=1, scene_builds=0, num_envs=run_config.num_envs, batches=0)
    with launch_simulation(SimpleNamespace(sim=sim_cfg, scene=first), args):
        # Import USD through Kit before opening room layers; a standalone USD wheel
        # loaded before launch conflicts with the renderer's native USD libraries.
        spacing = background_spacing(scenarios)
        first.env_spacing = spacing
        for group_index, (_, indices) in enumerate(groups):
            if group_index:
                release_scene()
            template = scenarios[indices[0]]
            num_envs = min(run_config.num_envs, len(indices))
            definition = first if group_index == 0 else definition_for(template, num_envs)
            sim = sim_utils.SimulationContext(sim_cfg)
            scene = instantiate(definition)
            configure_stage(config)
            sim.reset()
            summary["scene_builds"] += 1
            if summary["scene_builds"] == 1:
                summary["startup_seconds"] = time.monotonic() - started
                configure_worker()
            world = SimpleNamespace(
                sim=sim,
                scene=scene,
                config=config,
                template=template["objects"],
                robot=scene["robot"],
                objects=[scene[obj["name"]] for obj in template["objects"]],
                origins=scene.env_origins.clone(),
                rig=ParallelCameraRig(scene, scene["robot"], sim, config, record_depth=record_depth),
                ik=DifferentialIKController(
                    DifferentialIKControllerCfg(
                        command_type="pose", use_relative_mode=False, ik_method=config.IK_METHOD
                    ),
                    num_envs=num_envs,
                    device=sim.device,
                ),
            )
            robot = world.robot
            world.arm_ids, _ = robot.find_joints("panda_joint[1-7]")
            world.finger_ids, _ = robot.find_joints("panda_finger_joint.*")
            hand_ids, _ = robot.find_bodies("panda_hand")
            world.hand_id = hand_ids[0]
            world.jacobian_id = world.hand_id - 1 if robot.is_fixed_base else world.hand_id
            world.jacobian_columns = [j + robot.num_base_dofs for j in world.arm_ids]
            for start in range(0, len(indices), num_envs):
                batch = {env_id: index for env_id, index in enumerate(indices[start : start + num_envs])}
                _run_batch(world, batch, scenarios, args, run_config, run_dir, resolved, summary, capture_physics_steps)
                summary["batches"] += 1
                write_json_atomic(run_dir / "summary.json", summary)
        summary["completed"] = summary["attempts"] >= len(scenarios)
        summary["wall_seconds"] = time.monotonic() - started
        from data_collection.finalize import finalize_run

        finalize_run(run_dir, summary, run_config)
    return run_dir, summary


def _run_batch(world, batch, scenarios, args, run_config, run_dir, resolved, summary, capture_physics_steps):
    """Reset every slot together, then advance all occupied slots until each episode ends."""
    import torch
    from environments.simulation import initialize_motion

    from data_collection.finalize import finalize_attempt
    from data_collection.recorder import EpisodeRecorder
    from data_collection.simulation import _record_task_evidence

    sim, scene, config, robot = world.sim, world.scene, world.config, world.robot
    num_envs = world.origins.shape[0]
    write_json_atomic(
        run_dir / "progress.json",
        {"episode": max(batch.values()), "batch": sorted(batch.values()), "started": time.time()},
    )
    reset_started = time.monotonic()
    slots = {}
    for env_id, index in batch.items():
        scenario = scenarios[index]
        specs = deepcopy(scenario["objects"])
        slots[env_id] = SimpleNamespace(
            index=index, scenario=scenario, specs=specs, order=prim_order(world.template, specs), done=False
        )

    # Object poses: occupied slots use their episode; idle slots keep the template layout.
    for prim_index, obj in enumerate(world.objects):
        pose = obj.data.default_root_pose.torch.clone()
        for env_id in range(num_envs):
            spec = world.template[prim_index]
            if env_id in slots:
                slot = slots[env_id]
                spec = slot.specs[slot.order.index(prim_index)]
            pose[env_id, :3] = torch.tensor(spec["position_m"], device=sim.device) + world.origins[env_id]
            if config.OBJECT_START_ASLEEP:
                pose[env_id, 2] = config.TABLE_HEIGHT + spec["size_m"][2] / 2 + world.origins[env_id, 2]
            pose[env_id, 3:] = torch.tensor(spec["rotation_xyzw"], device=sim.device)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        if not world.template[prim_index].get("kinematic", False):
            obj.write_root_velocity_to_sim_index(root_velocity=obj.data.default_root_vel.torch)
    positions = robot.data.default_joint_pos.torch.clone()
    velocities = robot.data.default_joint_vel.torch.clone()
    robot.write_joint_position_to_sim_index(position=positions)
    robot.write_joint_velocity_to_sim_index(velocity=velocities)
    robot.set_joint_position_target_index(target=positions)
    scene.reset()
    world.ik.reset()
    samples = {env_id: _apply_appearance(world, env_id, slot, run_config) for env_id, slot in slots.items()}
    scene.write_data_to_sim()
    if config.OBJECT_START_ASLEEP:
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, Sdf

        physics = get_physx_simulation_interface()
        stage_id = omni.usd.get_context().get_stage_id()
        for env_id in range(num_envs):
            for obj in world.template:
                path = Sdf.Path(f"/World/envs/env_{env_id}/{obj['name']}")
                physics.put_to_sleep(stage_id, PhysicsSchemaTools.sdfPathToInt(path))

    def step():
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(config.PHYSICS_DT)

    settling_seconds = _settle_all(sim, step, world.objects, config)
    world.rig.sync()
    for _ in range(4):
        sim.render()
    observations = _observe_all(world, slots)
    reset_seconds = time.monotonic() - reset_started
    dt = config.PHYSICS_DT * config.CONTROL_STEPS
    for env_id, slot in slots.items():
        before = observations[env_id]
        target = slot.scenario["target_index"]
        slot.motion, slot.evaluator = initialize_motion(before, slot.specs, target, slot.scenario, config)
        slot.target = target
        slot.before = before
        metadata = episode_info(run_config.task, slot.scenario, slot.scenario["scene_seed"], target)
        metadata["attempt_id"] = run_config.attempt_id
        metadata["reset_seconds"] = reset_seconds
        metadata["parallel_slot"] = {"env_id": env_id, "num_envs": num_envs}
        scene_record = {
            "scenario": slot.scenario,
            "objects": [
                dict(
                    obj,
                    position_m=before["object_pose_xyzw"][i, :3].tolist(),
                    rotation_xyzw=before["object_pose_xyzw"][i, 3:].tolist(),
                )
                for i, obj in enumerate(slot.specs)
            ],
            "randomization_samples": samples[env_id],
            "settling_seconds": settling_seconds,
            "object_initialization": (
                "supported_pose_asleep_until_contact" if config.OBJECT_START_ASLEEP else "free_settling"
            ),
        }
        directory = run_dir / f"episode_{slot.index:04d}"
        slot.recorder = EpisodeRecorder(
            directory,
            metadata,
            (directory if getattr(run_config, "isolated_episodes", False) else run_dir) / "dataset.hdf5",
            compression=getattr(run_config, "compression", "gzip"),
            scene=scene_record,
            recording=recording_info(
                config, robot.joint_names, args.capture_profile, slot.scenario["manipulation_protocol"]
            ),
            run_config=resolved,
        )
        slot.peak_lifts = np.zeros(len(slot.specs))
        slot.success, slot.reason = False, "timeout"
        slot.pending = {}
        slot.started = time.monotonic()
        slot.timings = {"physics_and_evaluation": 0.0, "camera": 0.0, "recording": 0.0}

    def finish(slot):
        recorder = slot.recorder
        recorder.metadata["execution_seconds"] = time.monotonic() - slot.started
        recorder.metadata["stage_seconds"] = slot.timings
        _record_task_evidence(recorder, slot.scenario, slot.evaluator, slot.peak_lifts, slot.specs, config)
        recorder.scene["peak_object_lifts_m"] = slot.peak_lifts.tolist()
        recorder.metadata["accepted_demonstration"] = bool(
            slot.success and recorder.metadata["observability_status"] == "verified"
        )
        finalize_attempt(recorder, slot.success, slot.reason, run_config)
        slot.done = True
        summary["attempts"] += 1
        summary["successes"] += int(slot.success)
        summary["failures"] += int(not slot.success)
        print(f"Episode {slot.index}: {slot.reason}; saved {recorder.directory}", flush=True)

    positions = robot.data.joint_pos.torch.clone()
    for tick in range(round(config.TIMEOUT_SECONDS / dt)):
        active = {env_id: slot for env_id, slot in slots.items() if not slot.done}
        if not active:
            break
        hand = robot.data.body_pose_w.torch[:, world.hand_id]
        goals = hand.clone()
        images = world.rig.capture(0.0) if tick == 0 else None
        for env_id, slot in active.items():
            before = slot.before
            scenario = slot.scenario
            goal, finger, phase = slot.motion.command(
                before["hand_pose_xyzw"],
                before["joint_position"][world.finger_ids],
                before["joint_velocity"][world.finger_ids],
                **(
                    {}
                    if scenario["manipulation_protocol"] == "pick_hold_v1"
                    else dict(
                        object_bounds=(before["object_aabb_min"][slot.target], before["object_aabb_max"][slot.target]),
                        object_velocity=before["object_velocity"][slot.target],
                    )
                ),
                **(
                    {"object_pose": before["object_pose_xyzw"][slot.target]}
                    if engine(scenario) == "control"
                    else {"observation": before}
                    if engine(scenario) == "multi_object"
                    else {}
                ),
            )
            slot.goal, slot.finger, slot.phase = goal, finger, phase
            goals[env_id] = torch.tensor(goal, dtype=torch.float32, device=sim.device)
            goals[env_id, :3] += world.origins[env_id]
        # Keep goal, hand pose, and link Jacobian in the same world frame for every slot.
        world.ik.set_command(goals)
        jacobian = robot.data.body_link_jacobian_w.torch[:, world.jacobian_id, :, world.jacobian_columns]
        desired = world.ik.compute(hand[:, :3], hand[:, 3:], jacobian, robot.data.joint_pos.torch[:, world.arm_ids])
        limits = robot.data.soft_joint_pos_limits.torch[:, world.arm_ids]
        desired = torch.maximum(torch.minimum(desired, limits[..., 1]), limits[..., 0])
        for env_id, slot in active.items():
            positions[env_id, world.arm_ids] = desired[env_id]
            positions[env_id, world.finger_ids] = torch.as_tensor(
                slot.finger, dtype=positions.dtype, device=positions.device
            )
            recording_started = time.monotonic()
            slot.recorder.before(
                action=positions[env_id].cpu().numpy(),
                phase=slot.phase,
                time_s=tick * dt,
                observation=dict(slot.before, hand_goal_xyzw=slot.goal),
                cameras=_slice_images(images, env_id) if tick == 0 else slot.pending,
            )
            slot.timings["recording"] += time.monotonic() - recording_started
            slot.pending = {}
        robot.set_joint_position_target_index(target=positions)
        physics_tick = tick * config.CONTROL_STEPS
        after = None
        for _ in range(config.CONTROL_STEPS):
            physics_started = time.monotonic()
            step()
            after = _observe_all(world, active)
            # As in the single-environment loop: evaluate every sub-step, keep the last result.
            for env_id, slot in active.items():
                state = after[env_id]
                slot.success = slot.evaluator.update(
                    state, state["joint_position"][world.finger_ids], slot.phase, config.PHYSICS_DT
                )
            elapsed = time.monotonic() - physics_started
            for slot in active.values():
                slot.timings["physics_and_evaluation"] += elapsed / len(active)
            physics_tick += 1
            if physics_tick % capture_physics_steps == 0:
                capture_started = time.monotonic()
                frames = world.rig.capture(physics_tick * config.PHYSICS_DT)
                elapsed = time.monotonic() - capture_started
                for env_id, slot in active.items():
                    slot.pending = _slice_images(frames, env_id)
                    slot.timings["camera"] += elapsed / len(active)
        for env_id, slot in active.items():
            state = after[env_id]
            if not all(np.isfinite(value).all() for value in state.values()):
                slot.reason, slot.success = "nonfinite_state", False
                finish(slot)
                continue
            slot.peak_lifts = slot.evaluator.peak_lifts.copy()
            recording_started = time.monotonic()
            slot.recorder.after(state, slot.success)
            slot.timings["recording"] += time.monotonic() - recording_started
            slot.before = state
            if slot.success:
                slot.reason = "success"
                finish(slot)
    for slot in slots.values():
        if not slot.done:
            finish(slot)


def _observe_all(world, slots):
    """Per-slot observations in each environment's table frame, with objects in spec order."""
    import torch
    from assets.geometry import world_aabb

    robot = world.robot
    origins = world.origins
    root = robot.data.root_pose_w.torch.clone()
    root[:, :3] -= origins
    hand = robot.data.body_pose_w.torch[:, world.hand_id].clone()
    hand[:, :3] -= origins
    joint_position = robot.data.joint_pos.torch.cpu().numpy()
    joint_velocity = robot.data.joint_vel.torch.cpu().numpy()
    poses = torch.stack([obj.data.root_pose_w.torch for obj in world.objects], dim=1).clone()
    poses[..., :3] -= origins[:, None]
    velocities = torch.stack([obj.data.root_vel_w.torch for obj in world.objects], dim=1).cpu().numpy()
    poses = poses.cpu().numpy()
    root, hand = root.cpu().numpy(), hand.cpu().numpy()
    result = {}
    for env_id, slot in slots.items():
        order = slot.order
        object_poses = poses[env_id, order].copy()
        bounds = [world_aabb(obj["geometry"], pose[:3], pose[3:]) for obj, pose in zip(slot.specs, object_poses)]
        result[env_id] = {
            "robot_root_pose_xyzw": root[env_id].copy(),
            "joint_position": joint_position[env_id].copy(),
            "joint_velocity": joint_velocity[env_id].copy(),
            "object_pose_xyzw": object_poses,
            "object_velocity": velocities[env_id, order].copy(),
            "object_aabb_min": np.asarray([v[0] for v in bounds]),
            "object_aabb_max": np.asarray([v[1] for v in bounds]),
            "hand_pose_xyzw": hand[env_id].copy(),
        }
    return result


def _settle_all(sim, step, objects, config):
    """Wait until every object in every environment stays still; fail the batch otherwise."""
    import torch

    stable_steps = 0
    required = round(config.SETTLE_HOLD_SECONDS / config.PHYSICS_DT)
    for settle_step in range(round(config.SETTLE_MAX_SECONDS / config.PHYSICS_DT)):
        step()
        velocities = torch.stack([obj.data.root_vel_w.torch for obj in objects], dim=1)
        stable = (
            torch.isfinite(velocities).all()
            and (torch.linalg.vector_norm(velocities[..., :3], dim=-1) < config.SETTLE_LINEAR_SPEED).all()
            and (torch.linalg.vector_norm(velocities[..., 3:], dim=-1) < config.SETTLE_ANGULAR_SPEED).all()
        )
        stable_steps = stable_steps + 1 if stable else 0
        if (settle_step + 1) * config.PHYSICS_DT >= config.SETTLE_SECONDS and stable_steps >= required:
            return (settle_step + 1) * config.PHYSICS_DT
    raise RuntimeError("Objects did not settle before collection; no episode in this batch was recorded")


def _apply_appearance(world, env_id, slot, run_config):
    """Apply this slot's frozen materials and background and return the recorded samples."""
    from environments.appearance import apply_scene_appearance

    from pxr import Gf, Usd, UsdShade

    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    samples = {"camera": {}, "light": []}
    scenario = slot.scenario
    if scenario.get("appearance"):
        samples["candidate_colors"] = {}
        for spec, prim_index in zip(slot.specs, slot.order):
            if not spec.get("selectable", True):
                continue
            root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{world.template[prim_index]['name']}")
            marker = root.GetPath().AppendChild("AxisMarker")
            for prim in Usd.PrimRange(root):
                if prim.GetPath().HasPrefix(marker) or not prim.IsA(UsdShade.Shader):
                    continue
                color = UsdShade.Shader(prim).GetInput("diffuseColor")
                if color:
                    color.Set(Gf.Vec3f(*spec["color_rgb"]))
            samples["candidate_colors"][spec["name"]] = list(spec["color_rgb"])
    samples.update(
        apply_scene_appearance(
            stage, env_id, scenario.get("appearance"), scenario["randomization"], scenario["scene_seed"]
        )
    )
    robot = world.robot
    samples["robot_reset_joint_position"] = robot.data.default_joint_pos.torch[env_id].cpu().tolist()
    samples["robot_reset_joint_velocity"] = robot.data.default_joint_vel.torch[env_id].cpu().tolist()
    return samples


def _slice_images(frames, env_id):
    if not frames:
        return {}
    return {
        view: {
            key: (value[env_id] if isinstance(value, np.ndarray) and value.ndim and key != "frame" else value)
            for key, value in sample.items()
        }
        for view, sample in frames.items()
    }


class ParallelCameraRig:
    """Camera poses and synchronized capture for every environment, in each table frame."""

    def __init__(self, scene, robot, sim, config, *, record_depth):
        import torch

        self.robot, self.sim, self.record_depth = robot, sim, record_depth
        self.origins = scene.env_origins
        self.cameras = {view: scene[f"camera_{view}"] for view in config.CAMERAS}
        self.mounts = []
        count = self.origins.shape[0]
        for view, camera in self.cameras.items():
            pose = config.CAMERAS[view]
            if "body" in pose:
                body_ids, _ = robot.find_bodies(pose["body"])
                offset = torch.tensor([pose["offset_position"]], dtype=torch.float32, device=sim.device).expand(
                    count, -1
                )
                rotation = torch.tensor([pose["offset_rotation_xyzw"]], dtype=torch.float32, device=sim.device)
                self.mounts.append((camera, body_ids[0], offset, rotation.expand(count, -1)))
                continue
            eyes = torch.tensor(pose["position"], device=sim.device) + self.origins
            targets = torch.tensor(pose["target"], device=sim.device) + self.origins
            camera.set_world_poses_from_view(eyes, targets)

    def sync(self):
        from isaaclab.utils.math import combine_frame_transforms

        for camera, body_id, offset, rotation in self.mounts:
            body = self.robot.data.body_pose_w.torch[:, body_id]
            position, orientation = combine_frame_transforms(body[:, :3], body[:, 3:], offset, rotation)
            camera.set_world_poses(position, orientation, convention="ros")

    def capture(self, time_s):
        """Render once, then read every environment's frames; positions are environment-relative."""
        from isaaclab.sensors import SensorBase

        self.sync()
        self.sim.render()
        SensorBase.update_batch(tuple(self.cameras.values()), 0.0)
        frames = {}
        origins = self.origins.cpu().numpy()
        for view, camera in self.cameras.items():
            data = camera.data
            sample = {
                "frame": int(camera.frame.torch[0].item()),
                "timestamp_s": time_s,
                # Pending samples survive physics steps before the next action.
                # Snapshot shared CPU buffers once for the whole environment batch.
                "rgb": data.output["rgb"].torch[..., :3].cpu().numpy().copy(),
                "intrinsics": data.intrinsic_matrices.torch.cpu().numpy().copy(),
                "position": data.pos_w.torch.cpu().numpy() - origins,
                "quaternion_ros_xyzw": data.quat_w_ros.torch.cpu().numpy().copy(),
            }
            if self.record_depth:
                depth = data.output["distance_to_image_plane"].torch.cpu().numpy().copy()
                sample["distance_to_image_plane"] = depth
                sample["depth_valid"] = np.isfinite(depth) & (depth > 0)
            frames[view] = sample
        return frames
