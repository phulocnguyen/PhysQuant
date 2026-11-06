# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop evaluation environment for every Spatial Bench suite.

``reset`` replays a frozen scenario exactly as data collection did: the same scene,
appearance, background, object poses, settling, and robot state. ``step`` applies
one policy action, advances physics, and scores the episode with the protocol's
physics evaluator, which uses measured state only. The policy sees images, joint
positions, and the instruction; targets, measurements, and goals stay hidden.

Use inside ``launch(...)``, which starts Isaac Sim once for many episodes::

    with launch(args) as env:
        observation = env.reset(scenario)
        while True:
            observation, done, info = env.step(action)
            if done:
                break
"""

import time
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

VIEWS = ("top", "wrist")


def resolve_config(scenario):
    """Shared settings resolved exactly as collection resolved them for this scenario."""
    from environments import settings
    from tasks.common.scenarios import digest
    from tasks.registry import get_task, scenario_subgroup

    if scenario["settings_digest"] != digest(settings._SHARED_SETTINGS):
        raise ValueError("Scenario was frozen with different shared settings (cameras, table, robot)")
    task = get_task(scenario_subgroup(scenario))
    return task.resolve_settings(
        SimpleNamespace(task_settings={}, scenario=scenario, render_profile=scenario.get("render_profile", "quality"))
    )


class SpatialBenchEnv:
    """One Isaac Sim process; scenes rebuild in place when geometry changes between episodes."""

    def __init__(self, args, fps=10):
        self.args = args
        self.fps = fps
        self.world = None
        self.scene_key = None

    # -- scene ------------------------------------------------------------

    def _build(self, scenario, config):
        from environments.simulation import build_scene, release_scene, scene_definition
        from isaaclab_physx.physics import PhysxCfg
        from tasks.common.scenarios import compatibility_key

        import isaaclab.sim as sim_utils

        key = compatibility_key(scenario)
        if self.world is not None and key == self.scene_key:
            return
        if self.world is not None:
            release_scene()
        definition = scene_definition(scenario["objects"], scenario, config, record_depth=False)
        if self.world is None:
            self.sim_cfg = sim_utils.SimulationCfg(dt=config.PHYSICS_DT, device="cpu", physics=PhysxCfg())
            self._launch(definition)
        run_config = SimpleNamespace(randomization=scenario["randomization"])
        self.world = build_scene(self.sim_cfg, definition, scenario["objects"], scenario, config, run_config, False)
        self.scene_key = key

    def _launch(self, definition):
        from isaaclab.app import launch_simulation

        self.args.device = "cpu"  # Objects rest asleep on CPU PhysX; cameras render on the GPU.
        self._context = launch_simulation(SimpleNamespace(sim=self.sim_cfg, scene=definition), self.args)
        self._context.__enter__()

    def close(self):
        if self.world is not None:
            self._context.__exit__(None, None, None)
            self.world = None

    # -- episode ----------------------------------------------------------

    def reset(self, scenario):
        """Replay a frozen scenario's initial state; return the first observation."""

        from environments.simulation import initialize_motion, place_objects, settle_objects

        config = resolve_config(scenario)
        self.config = config
        self._build(scenario, config)
        w = self.world
        sim, scene, robot = w.sim, w.scene, w.robot
        specs = deepcopy(scenario["objects"])
        self.scenario, self.specs = scenario, specs
        place_objects(w.objects, specs, config, sim.device)
        positions, velocities, _ = w.randomizer.apply(
            specs,
            robot,
            w.arm_ids,
            scenario["scene_seed"],
            seed_streams=scenario.get("seeds"),
            appearance_seed=scenario.get("appearance", {}).get("seed"),
            appearance=scenario.get("appearance"),
        )
        robot.write_joint_position_to_sim_index(position=positions)
        robot.write_joint_velocity_to_sim_index(velocity=velocities)
        robot.set_joint_position_target_index(target=positions)
        scene.reset()
        w.ik.reset()
        scene.write_data_to_sim()
        if settle_objects(sim, self._physics_step, w.objects, specs, config) is None:
            raise RuntimeError("The simulator stopped while settling the scene")
        w.camera_rig.sync()
        for _ in range(4):
            sim.render()
        self.before = self._observe_state()
        self.initial_z = self.before["object_pose_xyzw"][:, 2].copy()
        self.motion, self.evaluator = initialize_motion(self.before, specs, scenario["target_index"], scenario, config)
        self.targets = positions.clone()
        self.control_dt = config.PHYSICS_DT * config.CONTROL_STEPS
        ratio = 1 / (self.control_dt * self.fps)
        if not np.isclose(ratio, round(ratio)):
            raise ValueError(f"fps {self.fps} must divide the {1 / self.control_dt:g} Hz control rate")
        self.ticks_per_step = round(ratio)
        self.max_ticks = round(config.TIMEOUT_SECONDS / self.control_dt)
        self.tick, self.success, self.phase = 0, False, "policy"
        self.peak_lifts = np.zeros(len(specs))
        self.started = time.monotonic()
        return self._observation()

    def step(self, action=None, *, oracle=False):
        """Execute one policy action (or the scripted controller with ``oracle``) for one policy step.

        ``action`` holds joint-position targets for the 9 joints; multi-object suites add a
        stop request (> 0.5 requests stop). Returns ``(observation, done, info)``.
        """
        import torch

        from evaluation.policies.oracle import apply_targets

        w = self.world
        robot = w.robot
        if not oracle:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape[0] not in (9, 10) or not np.isfinite(action).all():
                raise ValueError("Actions must be 9 finite joint targets, plus a stop flag for multi-object suites")
            self.phase = "stop" if action.shape[0] == 10 and action[9] > 0.5 else "policy"
            target = torch.as_tensor(action[:9], dtype=self.targets.dtype, device=self.targets.device)
            limits = robot.data.soft_joint_pos_limits.torch[0]
            self.targets[0] = torch.maximum(torch.minimum(target, limits[:, 1]), limits[:, 0])
        for _ in range(self.ticks_per_step):
            if oracle:
                self.phase = apply_targets(w, self.motion, self.before, self.scenario, self.targets)
            robot.set_joint_position_target_index(target=self.targets)
            for _ in range(self.config.CONTROL_STEPS):
                self._physics_step()
                state = self._observe_state()
                self.success = self.evaluator.update(
                    state, state["joint_position"][w.finger_ids], self.phase, self.config.PHYSICS_DT
                )
            self.before = state
            self.peak_lifts = np.maximum(self.peak_lifts, state["object_pose_xyzw"][:, 2] - self.initial_z)
            self.tick += 1
            if self.success or self.tick >= self.max_ticks:
                break
        done = bool(self.success or self.tick >= self.max_ticks)
        info = {"success": bool(self.success), "sim_time_s": self.tick * self.control_dt}
        if done:
            info.update(self._result())
        return self._observation(), done, info

    # -- helpers ----------------------------------------------------------

    def _physics_step(self):
        w = self.world
        w.scene.write_data_to_sim()
        w.sim.step(render=False)
        w.scene.update(self.config.PHYSICS_DT)

    def _observe_state(self):
        import torch
        from assets.geometry import world_aabb

        w = self.world
        poses = torch.stack([obj.data.root_pose_w.torch[0] for obj in w.objects]).cpu().numpy()
        bounds = [world_aabb(spec["geometry"], pose[:3], pose[3:]) for spec, pose in zip(self.specs, poses)]
        return {
            "robot_root_pose_xyzw": w.robot.data.root_pose_w.torch[0].cpu().numpy().copy(),
            "joint_position": w.robot.data.joint_pos.torch[0].cpu().numpy().copy(),
            "joint_velocity": w.robot.data.joint_vel.torch[0].cpu().numpy().copy(),
            "object_pose_xyzw": poses.copy(),
            "object_velocity": torch.stack([obj.data.root_vel_w.torch[0] for obj in w.objects]).cpu().numpy(),
            "object_aabb_min": np.asarray([b[0] for b in bounds]),
            "object_aabb_max": np.asarray([b[1] for b in bounds]),
            "hand_pose_xyzw": w.robot.data.body_pose_w.torch[0, w.hand_id].cpu().numpy().copy(),
        }

    def _observation(self):
        frames = self.world.camera_rig.capture(self.tick * self.control_dt)
        observation = {f"observation.images.{view}": frames[view]["rgb"] for view in VIEWS}
        observation["observation.state"] = self.before["joint_position"].astype(np.float32)
        observation["task"] = self.scenario["instruction"]
        return observation

    def _result(self):
        """Episode outcome from the physics evaluator's own metrics."""
        from tasks.registry import engine

        metrics = self.evaluator.metrics
        result = {"metrics": metrics, "wall_time_s": time.monotonic() - self.started}
        if engine(self.scenario) == "selection":
            # Same rule as collection: the object lifted highest, if any rose more than 2 cm.
            chosen = int(np.argmax(self.peak_lifts)) if self.peak_lifts.max() > 0.02 else None
            result["selected_object_id"] = self.specs[chosen]["object_id"] if chosen is not None else None
        return result


@contextmanager
def launch(args, fps=10):
    """Start one evaluation environment and close Isaac Sim afterwards."""
    env = SpatialBenchEnv(args, fps=fps)
    try:
        yield env
    finally:
        env.close()
