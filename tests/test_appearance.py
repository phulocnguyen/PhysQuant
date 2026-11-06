# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD material reset contracts, including frozen legacy appearance."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.appearance import apply_background, apply_table_style, background_spacing

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def table_stage(preview=False):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Ground")
    for env_id in (0, 1):
        for name in ("Table", "TableLeg0", "TableLeg1", "TableLeg2", "TableLeg3"):
            shader = UsdShade.Shader.Define(stage, f"/World/envs/env_{env_id}/{name}/Material/Shader")
            shader.CreateInput("diffuseColor" if preview else "diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0)
            )
    return stage


class AppearanceResetContracts(unittest.TestCase):
    def test_legacy_mean_color_does_not_enable_a_texture(self):
        stage = table_stage(preview=True)
        style = dict(name="legacy_wood", color_rgb=[0.2, 0.4, 0.6], source_texture="unavailable_legacy_texture.png")
        self.assertEqual(apply_table_style(stage, 0, style, textured=False), "solid_color")
        shader = UsdShade.Shader(stage.GetPrimAtPath("/World/envs/env_0/Table/Material/Shader"))
        self.assertEqual(shader.GetInput("diffuseColor").Get(), Gf.Vec3f(0.2, 0.4, 0.6))
        self.assertFalse(shader.GetInput("diffuse_texture"))

    def test_per_environment_styles_survive_other_slots_resetting(self):
        stage = table_stage()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            from PIL import Image

            texture = root / "wood.png"
            Image.new("RGB", (2, 2), (80, 40, 20)).save(texture)
            room = root / "room.usda"
            background = Usd.Stage.CreateNew(str(room))
            prim = UsdGeom.Xform.Define(background, "/Room").GetPrim()
            background.SetDefaultPrim(prim)
            wall = UsdGeom.Cube.Define(background, "/Room/Wall")
            wall.AddTranslateOp().Set((6, 0, 1))
            background.GetRootLayer().Save()
            # A room extending 7 m from its origin must not enter a neighboring workcell.
            self.assertGreater(background_spacing([{"appearance": {"background": {"usd": str(room)}}}]), 14)
            wood = dict(name="wood", color_rgb=[0.2, 0.1, 0.05], source_texture=str(texture))
            solid = dict(name="slate", color_rgb=[0.1, 0.2, 0.3], source_texture="")
            for env_id in (0, 1):
                self.assertEqual(apply_table_style(stage, env_id, wood), "texture")
                apply_background(stage, env_id, dict(usd=str(room)))
            apply_table_style(stage, 0, solid)
            apply_background(stage, 0, None)
            a = UsdShade.Shader(stage.GetPrimAtPath("/World/envs/env_0/Table/Material/Shader"))
            b = UsdShade.Shader(stage.GetPrimAtPath("/World/envs/env_1/Table/Material/Shader"))
            self.assertEqual(a.GetInput("diffuse_texture").Get().path, "")
            self.assertEqual(a.GetInput("diffuse_color_constant").Get(), Gf.Vec3f(0.1, 0.2, 0.3))
            self.assertEqual(b.GetInput("diffuse_texture").Get().path, str(texture))
            self.assertFalse(stage.GetPrimAtPath("/World/envs/env_0/Background/Wall"))
            self.assertTrue(stage.GetPrimAtPath("/World/envs/env_1/Background/Wall"))
            apply_background(stage, 1, None)
            self.assertEqual(UsdGeom.Imageable(stage.GetPrimAtPath("/World/Ground")).ComputeVisibility(), "inherited")


if __name__ == "__main__":
    unittest.main()
