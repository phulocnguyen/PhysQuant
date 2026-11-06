# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a physically tracked Franka/table/ruler scene for data collection."""

from assets.spawners import spawn_cfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from . import settings as config

# World-space metres per texture repeat on the tabletop and legs.
TABLE_TEXTURE_TILE_M = 0.5


def table_material(settings, *, textured=False):
    """OmniPBR with world-space projection, so tabletop textures need no mesh UVs.

    Episodes set its color or texture at reset; see :mod:`environments.appearance`.
    """
    if not textured:
        return sim.PreviewSurfaceCfg(diffuse_color=tuple(settings.TABLE_COLOR), roughness=settings.MATERIAL_ROUGHNESS)
    return sim.PbrMdlCfg(
        diffuse_color_constant=tuple(settings.TABLE_COLOR),
        reflection_roughness_constant=settings.MATERIAL_ROUGHNESS,
        project_uvw=True,
        texture_scale=(1 / TABLE_TEXTURE_TILE_M, 1 / TABLE_TEXTURE_TILE_M),
    )


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    """Shared table and robot; individual rulers are added from the task config."""

    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim.DomeLightCfg(intensity=config.DOME_INTENSITY))
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim.DistantLightCfg(intensity=config.KEY_INTENSITY, angle=5.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.258819, 0.0, 0.0, 0.965926)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim.CuboidCfg(
            size=config.TABLE_SIZE,
            collision_props=sim.CollisionPropertiesCfg(),
            visual_material=table_material(config),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, config.TABLE_HEIGHT - config.TABLE_SIZE[2] / 2)),
    )
    robot = replace(FRANKA_PANDA_HIGH_PD_CFG, prim_path="{ENV_REGEX_NS}/Robot")


def scene_cfg(
    specs: list[dict], record_depth: bool = True, settings=config, goal=None, reference_frame=None, appearance=None
) -> TabletopSceneCfg:
    """Spawn each ruler as a rigid object with known size and center."""
    cfg = TabletopSceneCfg(num_envs=1, env_spacing=2.0)
    cfg.table.spawn.size = tuple(settings.TABLE_SIZE)
    textured = (appearance or {}).get("version", 1) >= 2
    cfg.table.spawn.visual_material = table_material(settings, textured=textured)
    cfg.table.init_state.pos = (0.0, 0.0, settings.TABLE_HEIGHT - settings.TABLE_SIZE[2] / 2)
    cfg.robot.init_state.pos = tuple(settings.ROBOT_POSITION)
    cfg.robot.init_state.rot = tuple(settings.ROBOT_ROTATION)
    cfg.robot.spawn.variants["Physics"] = "physx"
    # Frozen marker-based plans retain their visuals; new table-edge frames have no overlay geometry.
    if reference_frame is not None and (
        not isinstance(reference_frame, dict) or reference_frame.get("visual_axes", True)
    ):
        for name, size, position, color in (
            ("frame_origin", (0.009, 0.009, 0.0004), (0.0, 0.0), (0.9, 0.9, 0.9)),
            ("frame_x", (0.12, 0.004, 0.0003), (0.065, 0.0), (0.7, 0.005, 0.005)),
            ("frame_y", (0.004, 0.12, 0.0003), (0.0, 0.065), (0.005, 0.02, 0.7)),
        ):
            setattr(
                cfg,
                name,
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/" + name,
                    spawn=sim.CuboidCfg(size=size, visual_material=sim.PreviewSurfaceCfg(diffuse_color=color)),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(*position, settings.TABLE_HEIGHT + 0.0002)),
                ),
            )
    if goal is not None:
        # A noncolliding inset marker identifies the existing table support surface.
        cfg.goal_zone = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GoalZone",
            spawn=sim.CuboidCfg(
                size=(*goal["goal_size_xy_m"], 0.0002),
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.05, 0.6, 0.1), roughness=1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(*goal["goal_center_xy_m"], settings.TABLE_HEIGHT + 0.0001)),
        )
    width, depth, thickness = settings.TABLE_SIZE
    leg_width = settings.TABLE_LEG_WIDTH
    leg_height = settings.TABLE_HEIGHT - thickness
    if leg_height <= 0 or not 0 < leg_width < min(width, depth):
        raise ValueError("Table legs must have positive height and fit beneath the tabletop")
    for index, (x_sign, y_sign) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        setattr(
            cfg,
            f"table_leg_{index}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/TableLeg{index}",
                spawn=sim.CuboidCfg(
                    size=(leg_width, leg_width, leg_height),
                    collision_props=sim.CollisionPropertiesCfg(),
                    visual_material=table_material(settings, textured=textured),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(
                        x_sign * (width / 2 - leg_width),
                        y_sign * (depth / 2 - leg_width),
                        leg_height / 2,
                    ),
                ),
            ),
        )
    for view in settings.CAMERAS:
        wrist = "body" in settings.CAMERAS[view]
        setattr(
            cfg,
            f"camera_{view}",
            CameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera_" + view,
                update_period=0,
                update_latest_camera_pose=True,
                width=settings.IMAGE_WIDTH,
                height=settings.IMAGE_HEIGHT,
                data_types=settings.CAMERA_DATA_TYPES if record_depth else ["rgb"],
                renderer_cfg=IsaacRtxRendererCfg(
                    global_settings=IsaacRtxRendererGlobalSettingsCfg(
                        antialiasing_mode=(
                            settings.RENDER_SETTINGS["antialiasing_mode"]
                            if settings.RENDER_SETTINGS["mode"] == "realtime"
                            else None
                        ),
                        enable_dlssg=False,
                        samples_per_pixel=settings.RENDER_SETTINGS["samples_per_pixel"],
                        enable_shadows=settings.RENDER_SETTINGS["shadows"],
                        enable_direct_lighting=True,
                        enable_dl_denoiser=True,
                        enable_global_illumination=False,
                        enable_reflections=False,
                        carb_settings=(
                            {
                                "/rtx/rendermode": "PathTracing",
                                "/rtx/pathtracing/spp": settings.RENDER_SETTINGS["path_tracing_samples"],
                                "/rtx/pathtracing/totalSpp": settings.RENDER_SETTINGS["path_tracing_samples"],
                                "/rtx/pathtracing/optixDenoiser/enabled": True,
                            }
                            if settings.RENDER_SETTINGS["mode"] == "path_traced"
                            else None
                        ),
                    )
                ),
                spawn=sim.PinholeCameraCfg(
                    focal_length=settings.WRIST_CAMERA_FOCAL_LENGTH if wrist else settings.CAMERA_FOCAL_LENGTH,
                    horizontal_aperture=(
                        settings.WRIST_CAMERA_HORIZONTAL_APERTURE if wrist else settings.CAMERA_HORIZONTAL_APERTURE
                    ),
                    clipping_range=tuple(
                        settings.WRIST_CAMERA_CLIPPING_RANGE if wrist else settings.CAMERA_CLIPPING_RANGE
                    ),
                ),
            ),
        )
    for obj in specs:
        setattr(
            cfg,
            obj["name"],
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + obj["name"],
                spawn=spawn_cfg(obj, roughness=settings.MATERIAL_ROUGHNESS),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(obj["position_m"]), rot=tuple(obj["rotation_xyzw"])
                ),
            ),
        )
        if obj.get("marked_local_axis"):
            setattr(
                cfg,
                obj["name"] + "_axis_marker",
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/" + obj["name"] + "/AxisMarker",
                    spawn=sim.CuboidCfg(
                        size=(0.012, 0.035, 0.0004),
                        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.8, 0.7, 0.02)),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(obj["size_m"][0] / 2 - 0.008, 0, obj["size_m"][2] / 2 + 0.0002)
                    ),
                ),
            )
    return cfg
