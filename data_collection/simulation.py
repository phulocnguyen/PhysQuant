# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Franka manipulation collection runtime."""

import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
from environments.simulation import (
    build_scene,
    initialize_motion,
    place_objects,
    release_scene,
    scene_definition,
    settle_objects,
)
from tasks.registry import engine

from isaaclab.app import launch_simulation

from data_collection.metadata import episode_info, recording_info, save_run_config, write_json_atomic


def run(args, run_config, *, config, specs, target, prompt, scenario, episode_scenarios=None):
    """Run the task and return its output directory and collection summary."""
    if getattr(run_config, "num_envs", 1) > 1:
        from data_collection.parallel import run_parallel

        return run_parallel(args, run_config, config=config, scenarios=episode_scenarios or [scenario])
    return _run_single(
        args,
        run_config,
        config=config,
        specs=specs,
        target=target,
        prompt=prompt,
        scenario=scenario,
        episode_scenarios=episode_scenarios,
    )


def _run_single(args, run_config, *, config, specs, target, prompt, scenario, episode_scenarios=None):
    """Collect episodes in one environment, rebuilding the stage in place when geometry changes."""
    import torch
    from isaaclab_physx.physics import PhysxCfg

    import isaaclab.sim as sim_utils

    from data_collection.recorder import EpisodeRecorder

    if args.capture_profile == "full" and "distance_to_image_plane" not in config.CAMERA_DATA_TYPES:
        raise ValueError("Full capture requires distance_to_image_plane in configs/settings.yaml")
    # Direct GPU PhysX forcibly disables sleeping. Use CPU physics for this
    # single-scene collector; RTX camera rendering still uses the GPU.
    if config.OBJECT_START_ASLEEP:
        args.device = "cpu"
        print("Objects start asleep: using CPU PhysX with GPU camera rendering.", flush=True)
    startup_started = time.monotonic()
    sim_cfg = sim_utils.SimulationCfg(dt=config.PHYSICS_DT, device=args.device, physics=PhysxCfg())

    record_depth = args.capture_profile == "full"
    definition = scene_definition(specs, scenario, config, record_depth=record_depth)
    from core.resources import configure_worker
    from tasks.common.scenarios import compatibility_key

    capture_physics_steps = capture_steps(run_config, config)
    print(f"Prompt: {prompt}", flush=True)
    run_dir = None
    if not args.preview:
        run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir.mkdir(parents=True, exist_ok=False)
        resolved = save_run_config(run_dir, run_config, config, args.output_dir)
        print(f"Dataset: {run_dir}", flush=True)

    # One launch serves every scene group in the batch. A group change rebuilds the
    # stage in the running app instead of relaunching Isaac Sim.
    sim = scene = robot = camera_rig = objects = randomizer = ik = scene_key = None
    arm_ids = finger_ids = hand_id = jacobian_id = jacobian_columns = None

    def build(group_specs, group_scenario, group_definition):
        """Create the simulation context, scene, and controllers for one scene group."""
        nonlocal sim, scene, robot, camera_rig, objects, randomizer, ik, scene_key
        nonlocal arm_ids, finger_ids, hand_id, jacobian_id, jacobian_columns
        world = build_scene(sim_cfg, group_definition, group_specs, group_scenario, config, run_config, record_depth)
        sim, scene, robot, camera_rig, objects = world.sim, world.scene, world.robot, world.camera_rig, world.objects
        randomizer, ik, arm_ids, finger_ids = world.randomizer, world.ik, world.arm_ids, world.finger_ids
        hand_id, jacobian_id, jacobian_columns = world.hand_id, world.jacobian_id, world.jacobian_columns
        scene_key = compatibility_key(group_scenario)

    with launch_simulation(SimpleNamespace(sim=sim_cfg, scene=definition), args):
        build(specs, scenario, definition)

        def observe():
            poses = torch.stack([obj.data.root_pose_w.torch[0] for obj in objects])
            from assets.geometry import world_aabb

            poses_np = poses.cpu().numpy().copy()
            bounds = [world_aabb(obj["geometry"], pose[:3], pose[3:]) for obj, pose in zip(specs, poses_np)]
            return {
                "robot_root_pose_xyzw": robot.data.root_pose_w.torch[0].cpu().numpy().copy(),
                "joint_position": robot.data.joint_pos.torch[0].cpu().numpy().copy(),
                "joint_velocity": robot.data.joint_vel.torch[0].cpu().numpy().copy(),
                "object_pose_xyzw": poses.cpu().numpy().copy(),
                "object_velocity": torch.stack([obj.data.root_vel_w.torch[0] for obj in objects]).cpu().numpy().copy(),
                "object_aabb_min": np.asarray([v[0] for v in bounds]),
                "object_aabb_max": np.asarray([v[1] for v in bounds]),
                "hand_pose_xyzw": robot.data.body_pose_w.torch[0, hand_id].cpu().numpy().copy(),
            }

        def step():
            scene.write_data_to_sim()
            sim.step(render=args.preview)
            scene.update(config.PHYSICS_DT)
            if args.preview:
                camera_rig.sync()

        summary = {"attempts": 0, "successes": 0, "failures": 0, "completed": False}
        summary["startup_seconds"] = time.monotonic() - startup_started
        summary["simulator_launch_count"] = 1
        summary["scene_builds"] = 1
        summary["scene_rebuild_seconds"] = 0.0
        attempt_limit = args.max_attempts if args.success_target is not None else min(args.episodes, args.max_attempts)
        for episode in range(attempt_limit):
            if episode_scenarios:
                scenario = episode_scenarios[episode]
                specs = scenario["objects"]
                target = scenario["target_index"]
                prompt = scenario["instruction"]
                if compatibility_key(scenario) != scene_key:
                    rebuild_started = time.monotonic()
                    release_scene()
                    build(specs, scenario, scene_definition(specs, scenario, config, record_depth=record_depth))
                    summary["scene_builds"] += 1
                    summary["scene_rebuild_seconds"] += time.monotonic() - rebuild_started
                    print(f"Episode {episode}: rebuilt the scene for new geometry", flush=True)
            if run_dir is not None:
                write_json_atomic(run_dir / "progress.json", {"episode": episode, "started": time.time()})
            reset_started = time.monotonic()
            episode_seed = scenario["scene_seed"]
            episode_specs = deepcopy(specs)
            place_objects(objects, episode_specs, config, sim.device)
            positions, velocities, randomization_samples = randomizer.apply(
                episode_specs,
                robot,
                arm_ids,
                episode_seed,
                seed_streams=scenario.get("seeds"),
                appearance_seed=scenario.get("appearance", {}).get("seed"),
                appearance=scenario.get("appearance"),
            )
            robot.write_joint_position_to_sim_index(position=positions)
            robot.write_joint_velocity_to_sim_index(velocity=velocities)
            robot.set_joint_position_target_index(target=positions)
            scene.reset()
            ik.reset()
            # Start supported objects asleep. PhysX wakes dynamic bodies on contact;
            # this is an initialization constraint, not a kinematic grasp override.
            import omni.usd
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools, Sdf

            physics_interface = get_physx_simulation_interface()
            stage_id = omni.usd.get_context().get_stage_id()
            scene.write_data_to_sim()
            if config.OBJECT_START_ASLEEP:
                for spec in episode_specs:
                    path = Sdf.Path("/World/envs/env_0/" + spec["name"])
                    physics_interface.put_to_sleep(stage_id, PhysicsSchemaTools.sdfPathToInt(path))
            settling_seconds = settle_objects(sim, step, objects, episode_specs, config)
            if settling_seconds is None:
                return None
            print(f"Objects stable after {settling_seconds:.2f}s", flush=True)
            if args.preview:
                while sim.is_running():
                    step()
                return None

            # Warm up rendering after reset; the first recorded action must have a valid image.
            camera_rig.sync()
            for _ in range(4):
                sim.render()
            configure_worker()
            before = observe()
            initial_centres = before["object_pose_xyzw"][:, :3].copy()
            motion, evaluator = initialize_motion(before, specs, target, scenario, config)
            metadata = episode_info(run_config.task, scenario, episode_seed, target)
            metadata["attempt_id"] = run_config.attempt_id + (0 if episode_scenarios else episode)
            metadata["reset_seconds"] = time.monotonic() - reset_started
            scene_record = {
                "scenario": scenario,
                "objects": [
                    dict(
                        obj,
                        position_m=initial_centres[i].tolist(),
                        rotation_xyzw=before["object_pose_xyzw"][i, 3:].tolist(),
                    )
                    for i, obj in enumerate(episode_specs)
                ],
                "randomization_samples": randomization_samples,
                "settling_seconds": settling_seconds,
                "object_initialization": (
                    "supported_pose_asleep_until_contact" if config.OBJECT_START_ASLEEP else "free_settling"
                ),
            }
            episode_dir = run_dir / f"episode_{episode:04d}"
            recorder = EpisodeRecorder(
                episode_dir,
                metadata,
                (episode_dir if getattr(run_config, "isolated_episodes", False) else run_dir) / "dataset.hdf5",
                compression=getattr(run_config, "compression", "gzip"),
                scene=scene_record,
                recording=recording_info(
                    config, robot.joint_names, args.capture_profile, scenario["manipulation_protocol"]
                ),
                run_config=resolved,
            )
            pending_images = {}
            physics_tick = 0
            peak_lifts = np.zeros(len(specs))
            success = False
            reason = "timeout"
            dt = config.PHYSICS_DT * config.CONTROL_STEPS
            execution_started = time.monotonic()
            timings = {"physics_and_evaluation": 0.0, "camera": 0.0, "recording": 0.0}
            try:
                for tick in range(round(config.TIMEOUT_SECONDS / dt)):
                    if not sim.is_running():
                        reason = "viewer_closed"
                        break
                    if scenario.get("diagnostic_intervention"):
                        from tasks.multi_object.diagnostics import intervene

                        before = intervene(scenario, motion, evaluator, objects, before)
                    goal, finger, phase = motion.command(
                        before["hand_pose_xyzw"],
                        before["joint_position"][finger_ids],
                        before["joint_velocity"][finger_ids],
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
                    goal_w = torch.tensor(goal, dtype=torch.float32, device=sim.device).unsqueeze(0)
                    # Keep goal, hand pose, and link Jacobian in the same world frame.
                    # Mixing root-frame errors with this Jacobian fails for a rotated base.
                    ik.set_command(goal_w)
                    hand = robot.data.body_pose_w.torch[:, hand_id]
                    jacobian = robot.data.body_link_jacobian_w.torch[:, jacobian_id, :, jacobian_columns]
                    desired = ik.compute(hand[:, :3], hand[:, 3:], jacobian, robot.data.joint_pos.torch[:, arm_ids])
                    limits = robot.data.soft_joint_pos_limits.torch[:, arm_ids]
                    desired = torch.maximum(torch.minimum(desired, limits[..., 1]), limits[..., 0])
                    positions[:, arm_ids] = desired
                    positions[:, finger_ids] = finger
                    capture_started = time.monotonic()
                    images = camera_rig.capture(0.0) if tick == 0 else pending_images
                    timings["camera"] += time.monotonic() - capture_started
                    pending_images = {}
                    recording_started = time.monotonic()
                    recorder.before(
                        action=positions[0].cpu().numpy(),
                        phase=phase,
                        time_s=tick * dt,
                        observation=dict(before, hand_goal_xyzw=goal),
                        cameras=images,
                    )
                    timings["recording"] += time.monotonic() - recording_started
                    if tick % 300 == 0:
                        print(f"Episode {episode}: {tick * dt:.1f}s simulated, phase={phase}", flush=True)
                    robot.set_joint_position_target_index(target=positions)
                    after, pending_images, physics_tick, success = _advance_control(
                        phase,
                        physics_tick,
                        step=step,
                        observe=observe,
                        camera_rig=camera_rig,
                        evaluator=evaluator,
                        finger_ids=finger_ids,
                        config=config,
                        capture_physics_steps=capture_physics_steps,
                        timings=timings,
                    )
                    if not all(np.isfinite(value).all() for value in after.values()):
                        reason = "nonfinite_state"
                        break
                    peak_lifts = evaluator.peak_lifts.copy()
                    recording_started = time.monotonic()
                    recorder.after(after, success)
                    timings["recording"] += time.monotonic() - recording_started
                    before = after
                    if success:
                        reason = "success"
                        break
            except KeyboardInterrupt:
                reason = "interrupted"
            except Exception:
                reason = "error"
                success = False
                raise
            finally:
                recorder.metadata["execution_seconds"] = time.monotonic() - execution_started
                recorder.metadata["stage_seconds"] = timings
                _record_task_evidence(recorder, scenario, evaluator, peak_lifts, specs, config)
                recorder.scene["peak_object_lifts_m"] = peak_lifts.tolist()
                recorder.metadata["accepted_demonstration"] = bool(
                    success and recorder.metadata["observability_status"] == "verified"
                )
                from data_collection.finalize import finalize_attempt

                finalize_attempt(recorder, success, reason, run_config)
            summary["attempts"] += 1
            summary["successes"] += int(success)
            summary["failures"] += int(not success)
            summary["completed"] = (
                summary["successes"] >= args.success_target
                if args.success_target is not None
                else summary["attempts"] >= args.episodes
            )
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"Episode {episode}: {reason}; saved {recorder.directory}", flush=True)
            if summary["completed"]:
                break
            if reason in ("viewer_closed", "interrupted"):
                break

        if run_dir is not None:
            from data_collection.finalize import finalize_run

            finalize_run(run_dir, summary, run_config)

    return run_dir, summary


def capture_steps(run_config, config):
    """Physics steps between image captures; ``camera_hz`` overrides the shared rate at runtime."""
    from core.timing import capture_stride

    if run_config.camera_hz is not None:
        # A runtime capture choice: recordings store the rate in their metadata.
        config.CAMERA_HZ = float(run_config.camera_hz)
    return capture_stride(config.PHYSICS_DT, config.CONTROL_STEPS, config.CAMERA_HZ)


def _advance_control(
    phase, physics_tick, *, step, observe, camera_rig, evaluator, finger_ids, config, capture_physics_steps, timings
):
    """Advance one action, checking task success at every physics step and capturing on its own clock."""
    images, success, after = {}, False, None
    for _ in range(config.CONTROL_STEPS):
        started = time.monotonic()
        step()
        after = observe()
        success = evaluator.update(after, after["joint_position"][finger_ids], phase, config.PHYSICS_DT)
        timings["physics_and_evaluation"] += time.monotonic() - started
        physics_tick += 1
        if physics_tick % capture_physics_steps == 0:
            started = time.monotonic()
            images = camera_rig.capture(physics_tick * config.PHYSICS_DT)
            timings["camera"] += time.monotonic() - started
    return after, images, physics_tick, success


def _record_task_evidence(recorder, scenario, evaluator, peak_lifts, specs, config):
    """Attach protocol-specific measured evidence to the common recording."""
    control_task = engine(scenario) in ("control", "multi_object")
    pick_task = scenario["manipulation_protocol"] == "pick_hold_v1"
    # Metadata keys are part of the recording schema; multi-object keeps its original prefix.
    prefix = (
        "long_horizon"
        if engine(scenario) == "multi_object"
        else "control"
        if control_task
        else "pick"
        if pick_task
        else "placement"
    )
    recorder.metadata[prefix + "_sampling_hz"] = 1 / config.PHYSICS_DT
    recorder.metadata[prefix + "_metrics"] = evaluator.metrics
    if control_task or pick_task:
        recorder.control_trace(evaluator.trace)
    if engine(scenario) == "multi_object":
        recorder.metadata["observed_selected_object_ids"] = [event["object_id"] for event in evaluator.grasp_events]
    else:
        selected = int(np.argmax(peak_lifts)) if np.max(peak_lifts) > 0.02 else None
        recorder.metadata["observed_selected_object_id"] = (
            specs[selected]["object_id"] if selected is not None else None
        )
