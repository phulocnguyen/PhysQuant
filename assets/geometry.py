# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exact primitive and represented compound/cavity geometry, independent of Isaac Sim."""

import math
from copy import deepcopy
from pathlib import Path

import yaml

CUP_SEGMENTS = 64
OBJECT_SIZES = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/object_sizes.yaml").read_text())


def polygon_area(radius, segments=CUP_SEGMENTS):
    return 0.5 * segments * radius**2 * math.sin(2 * math.pi / segments)


def positive(*values):
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Geometry dimensions must be finite and positive")


def surface_area(geometry):
    """Total boundary area of a closed primitive, in square metres.

    Open and compound shapes have no admitted boundary convention yet.
    """
    kind = geometry["kind"]
    if kind == "box":
        length, width, height = geometry["size_m"]
        positive(length, width, height)
        return 2 * (length * width + length * height + width * height)
    if kind == "cylinder":
        radius, height = geometry["radius_m"], geometry["height_m"]
        positive(radius, height)
        return 2 * math.pi * radius * (radius + height)
    if kind == "sphere":
        radius = geometry["radius_m"]
        positive(radius)
        return 4 * math.pi * radius**2
    raise ValueError(f"Total surface area is not qualified for {kind!r}")


def describe(geometry):
    """Return local-envelope dimensions, measurements, cavity and component annotations.

    Upright cylinders use Z height and a circular base. Capacity belongs only to hollow assets.
    Cup cavities are regular 64-gons: use their represented polygon area, not an
    unqualified ideal-circle volume. Solid volume excludes the cavity.
    """
    g = geometry
    kind = g["kind"]
    interior = None
    components = []
    grasp_offset = [0.0, 0.0, 0.0]
    if kind == "box":
        length, width, height = g["size_m"]
        positive(length, width, height)
        radius = None
        area = length * width
        volume = area * height
    elif kind in ("cylinder", "cone"):
        radius, height = g["radius_m"], g["height_m"]
        positive(radius, height)
        width = 2 * radius
        length = height if g.get("axis", "Z") == "X" else width
        envelope = [height, width, width] if g.get("axis", "Z") == "X" else [width, width, height]
        area = math.pi * radius**2 if g.get("axis", "Z") == "Z" else height * width
        volume = math.pi * radius**2 * height / (3 if kind == "cone" else 1)
    elif kind == "sphere":
        radius = g["radius_m"]
        positive(radius)
        length = width = height = 2 * radius
        area = math.pi * radius**2
        volume = 4 * math.pi * radius**3 / 3
    elif kind == "hollow_box":
        length, width, height = g["outer_size_m"]
        inner_l, inner_w = g["inner_size_xy_m"]
        bottom = g["bottom_m"]
        positive(length, width, height, inner_l, inner_w, bottom)
        if inner_l >= length or inner_w >= width or bottom >= height:
            raise ValueError("Cavity must fit strictly inside the outer box")
        depth = height - bottom
        interior = {
            "length": inner_l,
            "width": inner_w,
            "usable_depth": depth,
            "capacity": inner_l * inner_w * depth,
            "cross_section": "rectangle",
        }
        radius = None
        area = length * width
        volume = length * width * height - interior["capacity"]
    elif kind == "cup":
        radius = g["outer_radius_m"]
        inner = g["inner_radius_m"]
        height, bottom = g["height_m"], g["bottom_m"]
        positive(radius, inner, height, bottom)
        if inner >= radius or bottom >= height or g.get("segments", CUP_SEGMENTS) != CUP_SEGMENTS:
            raise ValueError("Invalid cup cavity or unsupported tessellation")
        length = width = 2 * radius
        cavity_area = polygon_area(inner)
        interior = {
            "radius": inner,
            "width": 2 * inner,
            "usable_depth": height - bottom,
            "capacity": cavity_area * (height - bottom),
            "cross_section": "regular_64_gon",
        }
        area = polygon_area(radius)
        volume = polygon_area(radius) * height - interior["capacity"]
    elif kind == "handled":
        if g["head_kind"] not in ("box", "cylinder", "elliptic_disk"):
            raise ValueError("Unsupported component head geometry")
        handle = g["handle_size_m"]
        head = g["head_size_m"]
        positive(*handle, *head)
        length = handle[0] + head[0]
        width = max(handle[1], head[1])
        height = max(handle[2], head[2])
        radius = None
        # Parts touch on a face and have disjoint interiors, so volumes add.
        head_volume = math.pi * (head[1] / 2) ** 2 * head[0] if g["head_kind"] == "cylinder" else math.prod(head)
        head_area = head[0] * head[1]
        if g["head_kind"] == "elliptic_disk":
            head_area = polygon_area(1) * head[0] * head[1] / 4
            head_volume = head_area * head[2]
        if g["head_kind"] == "cylinder" and not math.isclose(head[1], head[2]):
            raise ValueError("Circular head must have equal Y/Z diameter")
        volume = math.prod(handle) + head_volume
        area = handle[0] * handle[1] + head_area
        handle_position = [-head[0] / 2, 0.0, -(height - handle[2]) / 2]
        # Grip inside the handle near the head to reduce the head's gravity moment.
        # Geometry still uses the handle center; grasp and component pose are distinct.
        grasp_offset = [(handle[0] - head[0]) / 2 - min(0.012, handle[0] / 3), 0.0, handle_position[2]]
        components = [
            {
                "id": "handle",
                "semantic_label": "handle",
                "geometry": {"kind": "box", "size_m": handle},
                "dimensions_m": dict(zip(("length", "width", "height"), handle)),
                "position_local_m": handle_position,
            },
            {
                "id": "head",
                "semantic_label": "head",
                "geometry": {"kind": g["head_kind"], "size_m": head},
                "dimensions_m": dict(zip(("length", "width", "height"), head)),
                "position_local_m": [handle[0] / 2, 0.0, 0.0],
            },
        ]
    else:
        raise ValueError(f"Unsupported geometry {kind!r}")
    if kind not in ("cylinder", "cone"):
        envelope = [length, width, height]
    measures = {
        "length": length,
        "width": width,
        "height": envelope[2],
        "radius": radius,
        "diameter": 2 * radius if radius is not None else None,
        "footprint_area": area,
        "volume": volume,
        "capacity": interior["capacity"] if interior else None,
        "inner_width": interior["width"] if interior else None,
        "outer_width": width if interior else None,
        "handle_length": g["handle_size_m"][0] if kind == "handled" else None,
    }
    if kind in ("box", "hollow_box", "cup") or (kind == "cylinder" and g.get("axis", "Z") == "Z"):
        # Pinch 2 cm below the top so tall objects do not collide with the palm.
        grasp_offset = [0.0, 0.0, max(0.0, height / 2 - 0.02)]
    grasp_axis = "X" if envelope[1] >= 0.075 and envelope[0] < 0.075 else "Y"
    grasp_width = envelope[0] if grasp_axis == "X" else envelope[1]
    if kind == "handled":
        grasp_width = g["handle_size_m"][1]
        grasp_axis = "Y"
    elif kind == "cone":
        grasp_offset = [0.0, 0.0, max(0.0, height / 2 - 0.03)]
        # Pads span a tapered surface, so contact width differs from the TCP
        # cross-section. Use the base diameter as a conservative closure bound.
        grasp_width = 2 * radius
    limits = OBJECT_SIZES["limits"]
    size_feasible = (
        min(envelope) >= limits["minimum_extent_m"]
        and max(envelope[:2]) >= limits["minimum_planar_span_m"]
        and max(envelope) <= limits["maximum_extent_m"]
        and all(min(c["geometry"]["size_m"]) >= limits["minimum_component_extent_m"] for c in components)
    )
    return {
        "size_m": envelope,
        "measurements": measures,
        "interior": interior,
        "components": components,
        "grasp_width_m": grasp_width,
        "grasp_yaw_offset_rad": math.pi / 2 if grasp_axis == "X" else 0.0,
        "grasp_offset_local_m": grasp_offset,
        "size_feasible": size_feasible,
        "grasp_feasible": size_feasible and grasp_width <= limits["maximum_grasp_width_m"],
    }


def make(category, attribute, value, variant=0):
    """Construct a metrically controlled category instance for an exact query value."""
    if category in ("rectangular_bar", "rectangular_plate", "box"):
        size = OBJECT_SIZES["box_defaults"][category].copy()
        if attribute in ("length", "width", "height"):
            size[("length", "width", "height").index(attribute)] = value
        elif attribute == "footprint_area":
            size[1] = 0.05 + 0.002 * (variant % 4)
            size[0] = value / size[1]
        elif attribute == "surface_area":
            size[2] = (value / 2 - size[0] * size[1]) / (size[0] + size[1])
        elif attribute == "volume":
            # Both planar dimensions vary; height alone cannot determine volume.
            size[0] = 0.08 + 0.005 * (variant % 4)
            size[1] = 0.04 + 0.002 * ((variant + 1) % 4)
            size[2] = value / (size[0] * size[1])
        g = {"kind": "box", "size_m": size}
    elif category in ("cylinder", "circular_disk", "puck", "cone"):
        defaults = OBJECT_SIZES["round_defaults"]
        radius = defaults["radius_m"]
        height = defaults[
            "disk_height_m" if category == "circular_disk" else "puck_height_m" if category == "puck" else "height_m"
        ]
        if attribute == "radius":
            radius = value
        elif attribute == "diameter":
            radius = value / 2
        elif attribute in ("height", "length"):
            height = value
        elif attribute == "footprint_area":
            radius = math.sqrt(value / math.pi)
        elif attribute == "surface_area" and category != "cone":
            height = value / (2 * math.pi * radius) - radius
        elif attribute == "volume":
            radius = (
                defaults["volume_cone_radius_m"] + 0.00025 * (variant % 4)
                if category == "cone"
                else 0.032 + 0.001 * (variant % 4)
            )
            height = value * (3 if category == "cone" else 1) / (math.pi * radius**2)
        g = {"kind": "cone" if category == "cone" else "cylinder", "radius_m": radius, "height_m": height, "axis": "Z"}
    elif category == "sphere":
        radius = (
            value
            if attribute == "radius"
            else value / 2
            if attribute == "diameter"
            else math.sqrt(value / (4 * math.pi))
            if attribute == "surface_area"
            else (3 * value / (4 * math.pi)) ** (1 / 3)
        )
        g = {"kind": "sphere", "radius_m": radius}
    elif category in ("rectangular_container", "rectangular_tray"):
        g = {
            "kind": "hollow_box",
            "outer_size_m": [0.07, 0.10, 0.11],
            "inner_size_xy_m": [0.05, 0.07],
            "bottom_m": 0.008,
        }
        if category == "rectangular_tray":
            g["outer_size_m"][2] = 0.04
        if attribute == "capacity":
            # Fixed exteriors with different cavities prevent exterior-size shortcuts.
            g["outer_size_m"] = [0.072, 0.09, 0.14]
            inner_w = 0.07
            g["inner_size_xy_m"] = [value / (inner_w * 0.132), inner_w]
        elif attribute == "width":
            g["outer_size_m"][1] = value
            g["inner_size_xy_m"][1] = value - 0.012
        elif attribute == "inner_width":
            g["inner_size_xy_m"][1] = value
        elif attribute == "outer_width":
            g["outer_size_m"][1] = value
    elif category == "cylindrical_container":
        height, bottom = 0.14, 0.008 + 0.001 * (variant % 3)
        inner = math.sqrt(value / (polygon_area(1) * (height - bottom))) if attribute == "capacity" else 0.03
        g = {
            "kind": "cup",
            "outer_radius_m": 0.036 if attribute == "capacity" else 0.04,
            "inner_radius_m": inner,
            "height_m": height,
            "bottom_m": bottom,
            "segments": CUP_SEGMENTS,
        }
    elif category in ("handled_tool", "handled_paddle"):
        defaults = OBJECT_SIZES["component"]
        handle = value if attribute == "handle_length" else 0.09
        overall = defaults["overall_length_m"]  # Same overall length across handle queries.
        head_x = overall - handle
        positive(head_x)
        prefix = "tool" if category == "handled_tool" else "paddle"
        head = [head_x, defaults[f"{prefix}_head_width_m"], defaults[f"{prefix}_head_height_m"]]
        g = {
            "kind": "handled",
            "handle_size_m": [handle, defaults["handle_width_m"], defaults["handle_height_m"]],
            "head_size_m": head,
            "head_kind": "box" if category == "handled_tool" else "elliptic_disk",
        }
    else:
        raise ValueError(f"Unknown category {category!r}")
    describe(g)
    return deepcopy(g)


def rotate_vector(vector, quaternion):
    """Rotate a local vector by a normalized XYZW quaternion."""
    import numpy as np

    q = np.asarray(quaternion, dtype=float)
    q = q / np.linalg.norm(q)
    v = np.asarray(vector, dtype=float)
    return v + 2 * np.cross(q[:3], np.cross(q[:3], v) + q[3] * v)


def world_aabb(g, position, quaternion):
    """Exact primitive/represented-part bounds in the recorded world frame."""
    import numpy as np

    p = np.asarray(position, dtype=float)
    rotation = np.column_stack([rotate_vector(axis, quaternion) for axis in np.eye(3)])
    kind = g["kind"]
    if kind == "sphere":
        return p - g["radius_m"], p + g["radius_m"]
    if kind in ("cylinder", "cone"):
        axis = rotation[:, 0 if g.get("axis", "Z") == "X" else 2]
        radial = g["radius_m"] * np.sqrt(np.maximum(0.0, 1 - axis**2))
        if kind == "cylinder":
            extent = radial + g["height_m"] / 2 * np.abs(axis)
            return p - extent, p + extent
        apex = p + axis * g["height_m"] / 2
        base = p - axis * g["height_m"] / 2
        return np.minimum(apex, base - radial), np.maximum(apex, base + radial)
    if kind in ("cup", "elliptic_disk"):
        n = g.get("segments", CUP_SEGMENTS)
        rx, ry, height = (
            (g["outer_radius_m"], g["outer_radius_m"], g["height_m"])
            if kind == "cup"
            else (g["size_m"][0] / 2, g["size_m"][1] / 2, g["size_m"][2])
        )
        vertices = np.asarray(
            [
                (
                    rx * math.cos(2 * math.pi * i / n),
                    ry * math.sin(2 * math.pi * i / n),
                    z,
                )
                for z in (-height / 2, height / 2)
                for i in range(n)
            ]
        )
        points = vertices @ rotation.T + p
        return points.min(axis=0), points.max(axis=0)
    if kind == "handled":
        bounds = []
        for component in describe(g)["components"]:
            cg = component["geometry"]
            if cg["kind"] == "cylinder":
                dims = cg["size_m"]
                cg = {"kind": "cylinder", "radius_m": dims[1] / 2, "height_m": dims[0], "axis": "X"}
            centre = p + rotation @ np.asarray(component["position_local_m"])
            bounds.append(world_aabb(cg, centre, quaternion))
        return np.min([b[0] for b in bounds], axis=0), np.max([b[1] for b in bounds], axis=0)
    size = np.asarray(g["size_m"] if kind == "box" else g["outer_size_m"])
    extent = np.abs(rotation) @ size / 2
    return p - extent, p + extent


def construct(category, case, values, variant=0):
    """Construct exact object truth shared by the task generators."""
    first = next(iter(values))
    g = make(category, first, values[first], variant)
    if case == "multi_dimension":
        if category == "handled_tool":
            length, width, height = [values[k] for k in ("length", "width", "height")]
            g = {
                "kind": "handled",
                "handle_size_m": [length * 0.65, width * 0.65, height * 0.8],
                "head_size_m": [length * 0.35, width, height],
                "head_kind": "box",
            }
        else:
            g["size_m"] = [values[k] for k in ("length", "width", "height")]
    elif case == "nested_dimension":
        g["outer_size_m"][1] = values["outer_width"]
        g["inner_size_xy_m"][1] = values["inner_width"]
    info = describe(g)
    return dict(
        category=category,
        geometry=g,
        physical_properties_si=info["measurements"],
        internal_geometry=info["interior"],
        components=info["components"],
    )
