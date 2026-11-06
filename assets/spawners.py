# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural hollow and compound rigid assets with real collision geometry."""

import math
from collections.abc import Callable
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass

from .geometry import CUP_SEGMENTS, describe


def prism_mesh(stage, path, polygon, z_low, z_high, material, physics_material):
    """One convex extruded polygon; each cup-wall sector stays a separate collider."""
    from pxr import UsdGeom, UsdPhysics

    n = len(polygon)
    points = [(x, y, z) for z in (z_low, z_high) for x, y in polygon]
    counts = [n, n] + [4] * n
    indices = list(reversed(range(n))) + list(range(n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        indices.extend([i, j, j + n, i + n])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("convexHull")
    bind_visual_material(path, material, stage=stage)
    bind_physics_material(path, physics_material, stage=stage)


@clone
def spawn_assembly(prim_path, cfg, translation=None, orientation=None):
    """Root is one rigid body; all walls/components are child colliders."""
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    q = orientation or (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr().Set(float(cfg.mass_kg))
    geometry = cfg.geometry
    info = describe(geometry)
    material = prim_path + "/material"
    visual = sim.PreviewSurfaceCfg(diffuse_color=tuple(cfg.color_rgb), roughness=cfg.roughness, metallic=0.0)
    visual.func(material, visual)
    physics_path = prim_path + "/physicsMaterial"
    physical = sim.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6)
    physical.func(physics_path, physical)
    props = {"collision_props": sim.CollisionPropertiesCfg(), "visual_material": visual, "physics_material": physical}

    def box(name, size, position):
        shape = sim.CuboidCfg(size=tuple(size), **props)
        shape.func(prim_path + "/" + name, shape, translation=tuple(position))

    if geometry["kind"] == "hollow_box":
        length, width, height = geometry["outer_size_m"]
        inner_l, inner_w = geometry["inner_size_xy_m"]
        bottom = geometry["bottom_m"]
        tx, ty = (length - inner_l) / 2, (width - inner_w) / 2
        box("floor", [length, width, bottom], [0, 0, -(height - bottom) / 2])
        for sign in [-1, 1]:
            suffix = "negative" if sign < 0 else "positive"
            box(f"wall_x_{suffix}", [tx, width, height - bottom], [sign * (length - tx) / 2, 0, bottom / 2])
            box(f"wall_y_{suffix}", [inner_l, ty, height - bottom], [0, sign * (width - ty) / 2, bottom / 2])
    elif geometry["kind"] == "cup":
        outer, inner = geometry["outer_radius_m"], geometry["inner_radius_m"]
        height, bottom = geometry["height_m"], geometry["bottom_m"]
        angles = [2 * math.pi * i / CUP_SEGMENTS for i in range(CUP_SEGMENTS)]
        polygon = [(outer * math.cos(a), outer * math.sin(a)) for a in angles]
        prism_mesh(stage, prim_path + "/floor", polygon, -height / 2, -height / 2 + bottom, material, physics_path)
        for i, a in enumerate(angles):
            b = angles[(i + 1) % CUP_SEGMENTS]
            sector = [
                (outer * math.cos(a), outer * math.sin(a)),
                (outer * math.cos(b), outer * math.sin(b)),
                (inner * math.cos(b), inner * math.sin(b)),
                (inner * math.cos(a), inner * math.sin(a)),
            ]
            prism_mesh(stage, f"{prim_path}/wall_{i}", sector, -height / 2 + bottom, height / 2, material, physics_path)
    elif geometry["kind"] == "handled":
        handle, head = info["components"]
        box("handle", geometry["handle_size_m"], handle["position_local_m"])
        if geometry["head_kind"] == "box":
            box("head", geometry["head_size_m"], head["position_local_m"])
        elif geometry["head_kind"] == "elliptic_disk":
            length, width, height = geometry["head_size_m"]
            x, y, z = head["position_local_m"]
            polygon = [
                (
                    x + length / 2 * math.cos(2 * math.pi * i / CUP_SEGMENTS),
                    y + width / 2 * math.sin(2 * math.pi * i / CUP_SEGMENTS),
                )
                for i in range(CUP_SEGMENTS)
            ]
            prism_mesh(stage, prim_path + "/head", polygon, z - height / 2, z + height / 2, material, physics_path)
        else:
            dimensions = geometry["head_size_m"]
            shape = sim.CylinderCfg(radius=dimensions[1] / 2, height=dimensions[0], axis="X", **props)
            shape.func(prim_path + "/head", shape, translation=tuple(head["position_local_m"]))
    else:
        raise ValueError("Unsupported procedural assembly")
    return root.GetPrim()


@configclass
class AssemblyCfg(RigidObjectSpawnerCfg):
    func: Callable = spawn_assembly
    geometry: dict = MISSING
    color_rgb: tuple = (0.4, 0.45, 0.5)
    mass_kg: float = 0.05
    roughness: float = 0.8


def spawn_cfg(obj, roughness=0.8):
    """Build the exact primitive or assembled geometry from its authored parameters."""
    g = obj["geometry"]
    collision = sim.CollisionPropertiesCfg(
        contact_offset=obj.get("contact_offset_m"),
        rest_offset=0.0 if "contact_offset_m" in obj else None,
    )
    if obj.get("asset_usd"):
        return sim.UsdFileCfg(
            usd_path=str(Path(__file__).resolve().parents[1] / obj["asset_usd"]),
            rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=obj.get("kinematic", False)),
            collision_props=collision,
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=tuple(obj["color_rgb"]), roughness=roughness),
        )
    if g["kind"] in ("hollow_box", "cup", "handled"):
        return AssemblyCfg(geometry=g, color_rgb=tuple(obj["color_rgb"]), mass_kg=obj["mass_kg"], roughness=roughness)
    props = dict(
        rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=obj.get("kinematic", False)),
        mass_props=sim.MassPropertiesCfg(mass=obj["mass_kg"]),
        collision_props=collision,
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=tuple(obj["color_rgb"]), roughness=roughness, metallic=0.0),
        physics_material=sim.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
    )
    if g["kind"] == "box":
        return sim.CuboidCfg(size=tuple(g["size_m"]), **props)
    if g["kind"] == "sphere":
        return sim.SphereCfg(radius=g["radius_m"], **props)
    if g["kind"] == "cylinder":
        return sim.CylinderCfg(radius=g["radius_m"], height=g["height_m"], axis=g.get("axis", "Z"), **props)
    if g["kind"] == "cone":
        return sim.ConeCfg(radius=g["radius_m"], height=g["height_m"], axis=g.get("axis", "Z"), **props)
    raise ValueError(f"Unknown collision primitive {g['kind']}")
