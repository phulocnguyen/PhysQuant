# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene construction, physical reset, and task initialization shared by collection and evaluation."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from tasks.registry import engine


def scene_definition(specs, scenario, config, *, record_depth, num_envs=1):
    """Scene configuration for one scene group, optionally cloned into ``num_envs`` environments."""
    from environments.tabletop import scene_cfg

    definition = scene_cfg(
        specs,
        record_depth=record_depth,
        settings=config,
        goal=scenario["goal"] if scenario["manipulation_protocol"] == "pick_place_v1" else None,
        reference_frame=scenario.get("reference_frame"),
        appearance=scenario.get("appearance"),
    )
    if num_envs > 1:
        definition.num_envs = num_envs
        # Scene partitioning keeps each environment's cameras on its own stage subtree.
        definition.env_spacing = 4.0
    return definition


def build_scene(sim_cfg, definition, specs, scenario, config, run_config, record_depth):
    """Create the simulation context, single-environment scene, camera rig, randomizer, and IK."""
    import isaaclab.sim as sim_utils
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils import instantiate

    from environments.cameras import CameraRig
    from environments.randomization import EpisodeRandomizer

    sim = sim_utils.SimulationContext(sim_cfg)
    scene = instantiate(definition)
    configure_stage(config)
    sim.set_camera_view(config.CAMERA_POSITION, config.CAMERA_TARGET)
    sim.reset()
    robot = scene["robot"]
    camera_rig = CameraRig(scene, robot, sim, config, record_depth=record_depth)
    objects = [scene[obj["name"]] for obj in specs]
    arm_ids, _ = robot.find_joints("panda_joint[1-7]")
    randomizer = EpisodeRandomizer(
        run_config.randomization,
        objects,
        camera_rig.cameras,
        config.CAMERAS,
        appearance_specs=specs if scenario.get("appearance") else None,
    )
    finger_ids, _ = robot.find_joints("panda_finger_joint.*")
    hand_id = robot.find_bodies("panda_hand")[0][0]
    return SimpleNamespace(
        sim=sim,
        scene=scene,
        robot=robot,
        camera_rig=camera_rig,
        objects=objects,
        randomizer=randomizer,
        arm_ids=arm_ids,
        finger_ids=finger_ids,
        hand_id=hand_id,
        jacobian_id=hand_id - 1 if robot.is_fixed_base else hand_id,
        jacobian_columns=[j + robot.num_base_dofs for j in arm_ids],
        ik=DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method=config.IK_METHOD),
            num_envs=1,
            device=sim.device,
        ),
    )


def place_objects(objects, specs, config, device):
    """Write each object's authored pose; supported objects start resting on the table."""
    import torch

    for obj, spec in zip(objects, specs):
        pose = obj.data.default_root_pose.torch.clone()
        pose[:, :3] = torch.tensor(spec["position_m"], device=device)
        if config.OBJECT_START_ASLEEP:
            pose[:, 2] = config.TABLE_HEIGHT + spec["size_m"][2] / 2
        pose[:, 3:] = torch.tensor(spec["rotation_xyzw"], device=device)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        if not spec.get("kinematic", False):
            obj.write_root_velocity_to_sim_index(root_velocity=obj.data.default_root_vel.torch)


def release_scene():
    """Tear down the current simulation context and stage, keeping the Isaac Sim app running."""
    import gc

    import torch

    from isaaclab.sim import SimulationContext
    from isaaclab.sim.utils.stage import create_new_stage

    SimulationContext.clear_instance()
    create_new_stage()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def settle_objects(sim, step, objects, specs, config):
    """Wait for supported objects to remain still, or report physical reset failure."""
    import torch

    stable_steps = 0
    required_stable_steps = round(config.SETTLE_HOLD_SECONDS / config.PHYSICS_DT)
    for settle_step in range(round(config.SETTLE_MAX_SECONDS / config.PHYSICS_DT)):
        if not sim.is_running():
            return None
        step()
        object_velocities = torch.stack([obj.data.root_vel_w.torch[0] for obj in objects])
        stable = (
            torch.isfinite(object_velocities).all()
            and (torch.linalg.vector_norm(object_velocities[:, :3], dim=-1) < config.SETTLE_LINEAR_SPEED).all()
            and (torch.linalg.vector_norm(object_velocities[:, 3:], dim=-1) < config.SETTLE_ANGULAR_SPEED).all()
        )
        stable_steps = stable_steps + 1 if stable else 0
        if (settle_step + 1) * config.PHYSICS_DT >= config.SETTLE_SECONDS and stable_steps >= required_stable_steps:
            break
    else:
        diagnostics = []
        for spec, velocity in zip(specs, object_velocities.cpu().tolist()):
            diagnostics.append(
                {
                    "object": spec.get("name", spec.get("object_id", "unknown")),
                    "linear_speed_m_s": float(np.linalg.norm(velocity[:3])),
                    "angular_speed_rad_s": float(np.linalg.norm(velocity[3:])),
                    "velocity": velocity,
                }
            )
        print("Settling diagnostics: " + json.dumps(diagnostics), flush=True)
        raise RuntimeError(
            "Objects did not settle before collection. Check support geometry, "
            "reset collisions, and centre-of-mass randomization. No episode was recorded."
        )
    return (settle_step + 1) * config.PHYSICS_DT


def configure_stage(config):
    """Enable object sleeping and apply cast-shadow settings to the spawned lights."""
    from pxr import PhysxSchema, UsdLux

    from isaaclab.sim.utils.stage import get_current_stage

    for prim in get_current_stage().Traverse():
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            PhysxSchema.PhysxSceneAPI(prim).CreateDisableSleepingAttr().Set(False)
        if prim.HasAPI(UsdLux.LightAPI):
            # RTX path tracing uses shadow linking, not the realtime shadow
            # switch or UsdLux.ShadowAPI. An empty collection casts no shadows.
            shadow_link = UsdLux.LightAPI(prim).GetShadowLinkCollectionAPI()
            shadow_link.CreateIncludeRootAttr().Set(config.RENDER_SETTINGS["shadows"])


def initialize_motion(before, specs, target, scenario, config):
    """Validate the settled scene and construct fresh controller/evaluator state."""
    from assets.geometry import rotate_vector
    from controllers.pick_place import PickPlaceController
    from tasks.common.placement import PlacementEvaluator

    protocol = scenario["goal"]
    initial_centres = before["object_pose_xyzw"][:, :3].copy()
    reset_errors = np.linalg.norm(initial_centres - np.asarray([s["position_m"] for s in specs]), axis=1)
    if np.any(reset_errors > 0.005):
        raise ValueError(f"Scene drifted during reset: {reset_errors.tolist()}")
    if engine(scenario) == "multi_object":
        from controllers.multi_object import MultiObjectController
        from tasks.multi_object.evaluator import MultiObjectEvaluator

        evaluator = MultiObjectEvaluator(scenario, before)
        return MultiObjectController(scenario, evaluator, config), evaluator
    if engine(scenario) == "control":
        from controllers.single_object import SingleObjectController
        from tasks.control.evaluator import ControlEvaluator
        from tasks.control.goals import yaw as object_yaw

        evaluator = ControlEvaluator(scenario, before)
        pose = before["object_pose_xyzw"][target]
        base_yaw = np.deg2rad(config.ROBOT_YAW_DEG)
        grasp_yaw = (
            base_yaw
            + (object_yaw(pose[3:]) + specs[target].get("grasp_yaw_offset_rad", 0.0) - base_yaw + np.pi / 2) % np.pi
            - np.pi / 2
        )
        motion = SingleObjectController(
            pose, specs[target]["grasp_width_m"], evaluator.goal, protocol, config, grasp_yaw, specs[target]["size_m"]
        )
        return motion, evaluator
    if "predicate" in scenario:
        from tasks.common.scenarios import identify

        settled_scene = deepcopy(scenario)
        for index, obj in enumerate(settled_scene["objects"]):
            obj["position_m"] = before["object_pose_xyzw"][index, :3].tolist()
            obj["rotation_xyzw"] = before["object_pose_xyzw"][index, 3:].tolist()
        if identify(settled_scene) != [target]:
            raise ValueError("Settled scene violates the numerical task predicate")
    target_pose = before["object_pose_xyzw"][target]
    grasp_centre = initial_centres[target] + rotate_vector(
        specs[target].get("grasp_offset_local_m", [0.0, 0.0, 0.0]), target_pose[3:]
    )
    target_quat = before["object_pose_xyzw"][target, 3:]
    yaw = np.arctan2(
        2 * (target_quat[3] * target_quat[2] + target_quat[0] * target_quat[1]),
        1 - 2 * (target_quat[1] ** 2 + target_quat[2] ** 2),
    )
    yaw += specs[target].get("grasp_yaw_offset_rad", 0.0)
    # Choose the equivalent grasp closest to the robot's forward axis.
    base_yaw = np.deg2rad(config.ROBOT_YAW_DEG)
    yaw = base_yaw + (yaw - base_yaw + np.pi / 2) % np.pi - np.pi / 2
    if scenario["manipulation_protocol"] == "pick_hold_v1":
        from controllers.grasp_lift import GraspLiftController
        from tasks.common.picking import PickEvaluator

        motion = GraspLiftController(grasp_centre, specs[target]["grasp_width_m"], config)
        evaluator = PickEvaluator(scenario, before)
    else:
        offset = grasp_centre[:2] - initial_centres[target, :2]
        motion = PickPlaceController(
            grasp_centre,
            specs[target]["grasp_width_m"],
            goal_xy=np.asarray(protocol["goal_center_xy_m"]) + offset,
            clearance_z=float(before["object_aabb_max"][:, 2].max()) + config.HAND_TO_GRASP + 0.10,
            protocol=protocol,
            settings=config,
        )
        evaluator = PlacementEvaluator(initial_centres, target, config.TABLE_HEIGHT, protocol)
    motion.rotation = np.array([np.cos(yaw / 2), np.sin(yaw / 2), 0.0, 0.0])
    return motion, evaluator
