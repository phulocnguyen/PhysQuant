# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Place the workcell on open floor in background scenes and write configs/backgrounds.yaml.

The workcell (table, robot, and fixed cameras) stays at the world origin. Each
background is moved so that a free floor region lies under the workcell. For each
placement this tool writes a small USD layer under assets/scenes/backgrounds/ that
references the source scene, applies the placement, removes conflicting movable
furniture and ceilings, and strips all physics, so backgrounds are visual only.

Sources:
- InteriorAgent apartments (living and dining rooms); movable furniture inside the
  workcell may be removed.
- RoboCasa kitchens; the robot's back faces the counters and the open side holds the
  top camera.

Run with: uv run --no-sync python PhysQuant/tools/place_backgrounds.py
"""

import argparse
import json
import math
import os
from pathlib import Path

import yaml
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
INTERIOR_AGENT = ROOT / "assets/scenes/InteriorAgent"
ROBOCASA = Path(os.environ.get("ROBOCASA_SCENES_DIR", "/media/ssd_8tb/haov/ExtrinsicDexterity/scenes/Robocasa"))
DERIVED = ROOT / "assets/scenes/backgrounds"
ROOM_TYPES = ("dining room", "living room")
# Furniture that may be removed when it stands where the workcell goes.
MOVABLE = {
    "table", "dining_table", "tea_table", "coffee_table", "side_table", "chair", "stool", "ornament", "vase",
    "flower", "plant", "pillow", "throw_pillow", "book", "bowl", "tray", "wine_set", "decorative_box", "toy",
    "cup", "plate", "kitchenware", "fruit", "office_supply", "clock", "daily_equipment", "bedding",
}  # fmt: skip
# Workcell clearance in its own frame (m): the table, robot base, swept arm, and side cameras,
# plus a clear corridor out to the top camera at y = 1.4 m.
CLEARANCE = (-0.95, -0.85, 0.95, 0.65)
TOP_CAMERA_CORRIDOR = (-0.30, 0.65, 0.30, 1.50)
HEIGHT_BAND = (0.02, 1.80)
WALL_MARGIN_M = 0.10
STEP_M = 0.05


def transform(points, center, yaw_index):
    yaw = yaw_index * math.pi / 2
    c, s = round(math.cos(yaw)), round(math.sin(yaw))
    return [(center[0] + c * x - s * y, center[1] + s * x + c * y) for x, y in points]


def rectangle(bounds, center, yaw_index):
    x0, y0, x1, y1 = bounds
    return Polygon(transform([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], center, yaw_index))


def bounds_cache():
    from pxr import Usd, UsdGeom

    return UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])


def item_boxes(groups, skip=()):
    """(path, category, footprint) for every item that reaches into the workcell height band."""
    cache = bounds_cache()
    result = []
    for group in groups:
        if group.GetName() in skip:
            continue
        for item in group.GetChildren():
            bound = cache.ComputeWorldBound(item).ComputeAlignedRange()
            if bound.IsEmpty():
                continue
            low, high = bound.GetMin(), bound.GetMax()
            if high[2] < HEIGHT_BAND[0] or low[2] > HEIGHT_BAND[1]:
                continue
            name = item.GetName()
            category = name.rsplit("_", 1)[0] if name[-1].isdigit() else name
            result.append((str(item.GetPath()), category, box(low[0], low[1], high[0], high[1])))
    return result


def backdrop_gap(center, yaw_index, blockers):
    """Distance from the robot's back edge to the nearest blocker behind it."""
    x0, y0, x1, _ = CLEARANCE
    for depth in [0.05 * i for i in range(1, 61)]:
        strip = rectangle((x0, y0 - depth, x1, y0), center, yaw_index)
        if any(strip.intersects(blocker) for blocker in blockers):
            return depth
    return math.inf


def search(floor, items, removable, blockers, contained_corridor):
    """Best (score, center, yaw_index, removed) on ``floor`` among 90-degree placements.

    Fewer removed items come first, then the closest backdrop behind the robot.
    """
    low_x, low_y, high_x, high_y = floor.bounds
    best = None
    for i in range(int((high_x - low_x) / STEP_M) + 1):
        for j in range(int((high_y - low_y) / STEP_M) + 1):
            center = (low_x + i * STEP_M, low_y + j * STEP_M)
            for yaw_index in range(4):
                cell = rectangle(CLEARANCE, center, yaw_index)
                corridor = rectangle(TOP_CAMERA_CORRIDOR, center, yaw_index)
                if not floor.contains(cell) or (contained_corridor and not floor.contains(corridor)):
                    continue
                region = cell.union(corridor)
                hits = [(path, category) for path, category, footprint in items if footprint.intersects(region)]
                if any(not removable(category) for _, category in hits):
                    continue
                score = (len(hits), backdrop_gap(center, yaw_index, blockers))
                if best is None or score < best[0]:
                    best = (score, center, yaw_index, [path for path, _ in hits])
    return best


def interior_agent():
    from pxr import Usd

    for scene_dir in sorted(p for p in INTERIOR_AGENT.iterdir() if (p / f"{p.name}.usda").exists()):
        usd = scene_dir / f"{scene_dir.name}.usda"
        stage = Usd.Stage.Open(str(usd))
        # Interior walls and columns are not part of the room outline; treat them as fixed obstacles.
        items = item_boxes(stage.GetPrimAtPath("/Root/Meshes").GetChildren(), skip={"floor", "ceiling"})
        for room in json.loads((scene_dir / "rooms.json").read_text()):
            kind = room["room_type"].lower()
            if kind not in ROOM_TYPES:
                continue
            floor = Polygon(room["polygon"]).buffer(-WALL_MARGIN_M)
            if floor.is_empty:
                continue
            walls = [Polygon(room["polygon"]).exterior.buffer(WALL_MARGIN_M)]
            blockers = walls + [footprint for _, category, footprint in items if category not in MOVABLE]
            best = search(floor, items, lambda category: category in MOVABLE, blockers, contained_corridor=True)
            if best is None:
                continue
            ceilings = [str(p.GetPath()) for p in stage.GetPrimAtPath("/Root/Meshes/ceiling").GetChildren()]
            lights = [
                str(p.GetPath())
                for p in stage.GetPrimAtPath("/Root/Rendering/Lights").GetChildren()
                if p.GetTypeName() in ("DomeLight", "DistantLight")
            ]
            yield dict(
                name=f"{scene_dir.name}_{kind.replace(' ', '_')}",
                collection="interior_agent",
                room_type=kind,
                source=usd,
                placement=best,
                # Ceilings block the shared dome and key lights; no camera sees them.
                deactivate=best[3] + ceilings + lights,
            )


def robocasa(limit):
    from pxr import Usd

    for scene_dir in sorted(ROBOCASA.iterdir())[:limit]:
        usd = scene_dir / "scene.usd"
        if not usd.exists():
            continue
        stage = Usd.Stage.Open(str(usd))
        root = stage.GetDefaultPrim()
        floor_prim = root.GetChild("floor_room")
        if not floor_prim:
            continue
        bound = bounds_cache().ComputeWorldBound(floor_prim).ComputeAlignedRange()
        low, high = bound.GetMin(), bound.GetMax()
        floor = box(low[0], low[1], high[0], high[1]).buffer(-WALL_MARGIN_M)
        solid = [entry for entry in item_boxes([root]) if not entry[1].startswith("floor")]
        best = search(floor, solid, lambda category: False, [f for _, _, f in solid], contained_corridor=False)
        if best is None:
            continue
        yield dict(
            name=scene_dir.name.replace("robocasa-robocasakitchen", "robocasa_kitchen"),
            collection="robocasa",
            room_type="kitchen",
            source=usd,
            placement=best,
            deactivate=[],
        )


def write_layer(entry, directory=DERIVED):
    """Author the derived background layer; return its config record."""
    from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    (_, gap), center, yaw_index, removed = entry["placement"]
    yaw = yaw_index * math.pi / 2
    c, s = math.cos(-yaw), math.sin(-yaw)
    translation = (-(c * center[0] - s * center[1]), -(s * center[0] + c * center[1]), 0.0)
    path = directory / f"{entry['name']}.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Use a fresh --asset_dir; existing background layers may belong to frozen plans: {path}")
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    background = UsdGeom.Xform.Define(stage, "/Background")
    stage.SetDefaultPrim(background.GetPrim())
    # The spawner sets the root prim's pose, so the placement lives on the child.
    scene = UsdGeom.Xform.Define(stage, "/Background/Scene")
    scene.AddTranslateOp().Set(translation)
    scene.AddRotateZOp().Set(math.degrees(-yaw))
    scene = scene.GetPrim()
    source = Usd.Stage.Open(str(entry["source"]))
    prefix = source.GetDefaultPrim().GetPath()
    scene.GetReferences().AddReference(str(entry["source"]), prefix)
    for item in entry["deactivate"]:
        stage.GetPrimAtPath(Sdf.Path(item).ReplacePrefix(prefix, scene.GetPath())).SetActive(False)
    # Inspect nested assets as well: room lights must not alter the workcell's lighting.
    for prim in list(Usd.PrimRange(scene)):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    # Visual only: no source lights/cameras, rigid bodies, colliders, articulations, or joints.
    schemas = (
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.CollisionAPI,
        UsdPhysics.MeshCollisionAPI,
        UsdPhysics.ArticulationRootAPI,
    )
    for prim in list(Usd.PrimRange(scene)):
        if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdLux.LightAPI) or prim.IsA(UsdGeom.Camera):
            prim.SetActive(False)
            continue
        for schema in schemas:
            if prim.HasAPI(schema):
                prim.RemoveAPI(schema)
        if prim.IsA(UsdGeom.Mesh):
            # Some source meshes label a single display color as face-varying.
            for name in ("displayColor", "displayOpacity"):
                value = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
                data = value.Get() if value else None
                if data is not None and len(data) == 1 and not value.IsIndexed():
                    value.SetInterpolation(UsdGeom.Tokens.constant)
    stage.GetRootLayer().Save()
    return dict(
        name=entry["name"],
        collection=entry["collection"],
        room_type=entry["room_type"],
        usd=str(path.relative_to(ROOT)),
        source=str(entry["source"]),
        position=[round(v, 4) for v in translation],
        yaw_deg=round(math.degrees(-yaw) % 360, 1),
        removed_items=len(removed),
        backdrop_gap_m=None if math.isinf(gap) else round(gap, 2),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/backgrounds.yaml")
    parser.add_argument("--asset_dir", type=Path, default=DERIVED, help="Fresh layer directory inside the project")
    parser.add_argument("--robocasa_limit", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh --output and --asset_dir to preserve existing frozen plans")
    directory = args.asset_dir if args.asset_dir.is_absolute() else ROOT / args.asset_dir
    backgrounds = []
    for entry in [*interior_agent(), *robocasa(args.robocasa_limit)]:
        record = write_layer(entry, directory)
        print(record["name"], "removed", record["removed_items"], "gap", record["backdrop_gap_m"], flush=True)
        backgrounds.append(record)
    header = (
        "# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).\n"
        "# All rights reserved.\n#\n# SPDX-License-Identifier: BSD-3-Clause\n\n"
        "# Generated by tools/place_backgrounds.py. Review renders before enabling an entry.\n"
        "# usd: derived layer with the placement, removed furniture, and stripped physics.\n"
    )
    args.output.write_text(header + yaml.safe_dump({"backgrounds": backgrounds}, sort_keys=False))
    print(f"Wrote {len(backgrounds)} backgrounds to {args.output}")


if __name__ == "__main__":
    main()
