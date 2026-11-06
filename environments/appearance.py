# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply a scene's frozen tabletop style and background to one environment at reset.

Both are visual only, so they change at reset without rebuilding the stage. Each
environment references its own background layer; copies of one background share
geometry through USD instancing.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_PRIMS = ("Table", "TableLeg0", "TableLeg1", "TableLeg2", "TableLeg3")


def background_spacing(scenarios):
    """Separate complete visual rooms, including backgrounds inserted after cloning."""
    import math

    from pxr import Usd, UsdGeom

    spacing = 4.0
    paths = {
        s.get("appearance", {}).get("background", {}).get("usd")
        for s in scenarios
        if s.get("appearance", {}).get("background")
    }
    for path in paths:
        stage = Usd.Stage.Open(str(ROOT / path))
        bound = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            .ComputeWorldBound(stage.GetDefaultPrim())
            .ComputeAlignedRange()
        )
        if bound.IsEmpty():
            raise ValueError(f"Background has no visible bounds: {path}")
        extent = max(abs(float(v[i])) for v in (bound.GetMin(), bound.GetMax()) for i in (0, 1))
        if not math.isfinite(extent):
            raise ValueError(f"Background has nonfinite bounds: {path}")
        spacing = max(spacing, 2 * extent + 2)
    return math.ceil(spacing)


def table_shaders(stage, env_id):
    """Material shaders below the tabletop and its legs in one environment."""
    from pxr import Usd, UsdShade

    shaders = []
    for name in TABLE_PRIMS:
        root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{name}")
        if not root:
            raise RuntimeError(f"Environment {env_id} is missing {name}")
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdShade.Shader):
                shaders.append(UsdShade.Shader(prim))
    if not shaders:
        raise RuntimeError(f"Environment {env_id} has no table material to style")
    return shaders


def apply_table_style(stage, env_id, style, *, textured=True):
    """Set the table color, and its texture when the style has one and ``textured`` is set."""
    from pxr import Gf, Sdf

    texture = style.get("source_texture", "") if textured else ""
    path = str(ROOT / texture) if texture else ""
    if path and not Path(path).exists():
        raise FileNotFoundError(f"Table texture is missing: {path}")
    for shader in table_shaders(stage, env_id):
        preview_color = shader.GetInput("diffuseColor")
        if preview_color:
            if texture:
                raise ValueError("Textured episodes require an OmniPBR table")
            preview_color.Set(Gf.Vec3f(*style["color_rgb"]))
            continue
        # A texture replaces the constant color; keep the tint neutral so the texture shows unchanged.
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*style["color_rgb"]))
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(path))
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    return "texture" if path else "solid_color"


def apply_scene_appearance(stage, env_id, appearance, randomization, seed):
    """Apply frozen scene choices consistently in single and parallel collectors."""
    from environments.randomization import sample_table_style

    appearance = appearance or {}
    version = appearance.get("version", 1)
    style = sample_table_style(randomization, appearance.get("seed", seed))
    samples = {}
    if style is not None:
        mode = apply_table_style(stage, env_id, style, textured=version >= 2)
        samples["table"] = dict(
            style,
            color_space="linear_rgb",
            mode="texture_mean_color" if mode == "solid_color" and style["source_texture"] else mode,
        )
    apply_background(stage, env_id, appearance.get("background") if version >= 2 else None)
    if version >= 2:
        samples["background"] = appearance.get("background")
    return samples


def apply_background(stage, env_id, background):
    """Reference this environment's background layer, or clear it when ``background`` is None."""
    from pxr import UsdGeom

    prim = stage.DefinePrim(f"/World/envs/env_{env_id}/Background", "Xform")
    references = prim.GetReferences()
    path = ROOT / background["usd"] if background is not None else None
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"Background layer is missing: {path}")
    current = prim.GetMetadata("references")
    desired = [str(path)] if path is not None else []
    if current is None or [ref.assetPath for ref in current.GetAppliedItems()] != desired:
        prim.SetInstanceable(False)
        references.ClearReferences()
        if path is not None:
            references.AddReference(str(path))
    if background is not None:
        prim.SetInstanceable(True)
    # The scene floor replaces the studio ground plane, which would otherwise z-fight with it.
    ground = stage.GetPrimAtPath("/World/Ground")
    if ground:
        imageable = UsdGeom.Imageable(ground)
        imageable.MakeInvisible() if background is not None else imageable.MakeVisible()
