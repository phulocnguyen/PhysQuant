# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical predicates and language derived from the same explicit expression tree."""

import math

import numpy as np
from assets.geometry import describe, surface_area, world_aabb
from core.units import DIMENSIONS, equal

SCALES = {"cm": (1, 0.01), "degrees": ("angle", math.pi / 180), "cm2": (2, 1e-4), "cm3": (3, 1e-6)}


def expression_dimension(expression, scene):
    kind = expression["kind"]
    if kind == "constant":
        dimension, scale = SCALES[expression["unit"]]
        if not math.isfinite(expression["value"]) or not math.isclose(
            expression["value_si"], expression["value"] * scale, abs_tol=1e-12
        ):
            raise ValueError("Predicate constant has inconsistent units")
        return dimension
    if "reference" in expression:
        ref = reference(scene, expression["reference"])
        if ref.get("selectable", True):
            raise ValueError("A reference cannot also be a selectable candidate")
    if kind in ("attribute", "reference_attribute"):
        return DIMENSIONS[expression["attribute"]]
    if kind in ("sum", "difference", "scale"):
        dimensions = {expression_dimension(child, scene) for child in expression["terms"]}
        if len(dimensions) != 1 or len(expression["terms"]) != (1 if kind == "scale" else 2):
            raise ValueError("Composition requires matching dimensions and the declared arity")
        if kind == "scale" and (not math.isfinite(expression["factor"]) or expression["factor"] <= 0):
            raise ValueError("Ratio factor must be positive and finite")
        return dimensions.pop()
    if kind == "orientation":
        return "angle"
    if kind not in ("coordinate", "offset", "distance", "clearance", "boundary_distance"):
        raise ValueError("Unknown numerical expression")
    if kind in ("coordinate", "offset", "boundary_distance") and expression["axis"] not in (0, 1, 2):
        raise ValueError("Coordinate axis must be x, y, or z")
    if kind == "offset" and expression["frame"] not in ("reference", "table"):
        raise ValueError("Unknown reference frame")
    return 1


def validate_predicate(predicate, scene):
    op = predicate["op"]
    if op == "and":
        if len(predicate["terms"]) < 2:
            raise ValueError("A conjunction requires multiple conditions")
        for term in predicate["terms"]:
            validate_predicate(term, scene)
        return
    if op not in ("eq", "gt", "ge", "lt", "le", "interval", "tolerance", "closest"):
        raise ValueError("Unsupported numerical operator")
    dimension = expression_dimension(predicate["measure"], scene)
    if dimension != expression_dimension(predicate["target"], scene):
        raise ValueError("Predicate compares incompatible dimensions")
    if op == "interval":
        if (
            SCALES[predicate["unit"]][0] != dimension
            or len(predicate["closed"]) != 2
            or any(type(v) is not bool for v in predicate["closed"])
        ):
            raise ValueError("Interval units or endpoint convention are invalid")
        expected = [v * SCALES[predicate["unit"]][1] for v in predicate["bounds"]]
        if (
            len(expected) != 2
            or not expected[0] < expected[1]
            or not np.allclose(expected, predicate["bounds_si"], rtol=0, atol=1e-12)
        ):
            raise ValueError("Invalid interval bounds")
    if "tolerance_si" in predicate:
        scale = SCALES[predicate["unit"]]
        if (
            scale[0] != dimension
            or predicate["tolerance"] <= 0
            or not equal(predicate["tolerance_si"], predicate["tolerance"] * scale[1])
        ):
            raise ValueError("Invalid predicate tolerance")


def yaw(obj):
    x, y, z, w = obj["rotation_xyzw"]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def reference(scene, identity):
    return next(o for o in scene["objects"] if o["object_id"] == identity)


def quantity(obj, expression, scene):
    kind = expression["kind"]
    if kind == "attribute":
        attribute = expression["attribute"]
        return (
            surface_area(obj["geometry"])
            if attribute == "surface_area"
            else describe(obj["geometry"])["measurements"][attribute]
        )
    if kind == "constant":
        return expression["value_si"]
    if kind == "reference_attribute":
        return quantity(
            reference(scene, expression["reference"]),
            {"kind": "attribute", "attribute": expression["attribute"]},
            scene,
        )
    if kind in ("sum", "difference", "scale"):
        values = [quantity(obj, child, scene) for child in expression["terms"]]
        if kind == "scale":
            return expression["factor"] * values[0]
        return sum(values) if kind == "sum" else values[0] - values[1]
    position = np.asarray(obj["position_m"])
    if kind == "coordinate":
        return float(position[expression["axis"]])
    if kind == "boundary_distance":
        lo, hi = world_aabb(obj["geometry"], obj["position_m"], obj["rotation_xyzw"])
        axis = expression["axis"]
        return float(
            lo[axis] - expression["boundary_m"]
            if expression["side"] == "lower"
            else expression["boundary_m"] - hi[axis]
        )
    if kind == "orientation":
        angle = yaw(obj)
        if "reference" in expression:
            angle -= yaw(reference(scene, expression["reference"]))
        return float(angle % math.pi)
    ref = reference(scene, expression["reference"])
    delta = position - np.asarray(ref["position_m"])
    if kind == "distance":
        return float(np.linalg.norm(delta[:2] if expression["planar"] else delta))
    if kind == "offset":
        if expression["frame"] == "reference":
            angle = yaw(ref)
            c, s = math.cos(angle), math.sin(angle)
            delta = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]])
        return float(delta[expression["axis"]])
    if kind == "clearance":
        # This exact distance is admitted only for axis-aligned rectangular boxes.
        if any(o["geometry"]["kind"] != "box" or abs(math.sin(2 * yaw(o))) > 1e-6 for o in (obj, ref)):
            raise ValueError("Surface clearance requires axis-aligned box collision geometry")
        a, b = world_aabb(obj["geometry"], obj["position_m"], obj["rotation_xyzw"])
        c, d = world_aabb(ref["geometry"], ref["position_m"], ref["rotation_xyzw"])
        return float(np.linalg.norm(np.maximum(np.maximum(a - d, c - b), 0)))
    raise ValueError(f"Unsupported measurement expression {kind!r}")


def matches(obj, predicate, scene):
    op = predicate["op"]
    if op == "and":
        return all(matches(obj, item, scene) for item in predicate["terms"])
    actual = quantity(obj, predicate["measure"], scene)
    if actual is None or not math.isfinite(actual):
        return False
    expected = quantity(obj, predicate["target"], scene)
    if op == "eq":
        if predicate["measure"]["kind"] == "orientation":
            return abs((actual - expected + math.pi / 2) % math.pi - math.pi / 2) <= predicate.get("tolerance_si", 1e-7)
        if "tolerance_si" in predicate:
            return abs(actual - expected) <= predicate["tolerance_si"]
        return equal(actual, expected)
    if op == "gt":
        return actual > expected and not equal(actual, expected)
    if op == "ge":
        return actual > expected or equal(actual, expected)
    if op == "lt":
        return actual < expected and not equal(actual, expected)
    if op == "le":
        return actual < expected or equal(actual, expected)
    if op == "tolerance":
        return abs(actual - expected) <= predicate["tolerance_si"] + 1e-12
    if op == "interval":
        lo, hi = predicate["bounds_si"]
        closed = predicate["closed"]
        return (actual >= lo - 1e-12 if closed[0] else actual > lo + 1e-12) and (
            actual <= hi + 1e-12 if closed[1] else actual < hi - 1e-12
        )
    if op == "closest":
        errors = [
            (abs(quantity(o, predicate["measure"], scene) - expected), o["object_id"])
            for o in scene["objects"]
            if o.get("selectable", True)
        ]
        minimum = min(error for error, _ in errors)
        winners = [identity for error, identity in errors if math.isclose(error, minimum, abs_tol=1e-10)]
        return len(winners) == 1 and winners[0] == obj["object_id"]
    raise ValueError(f"Unsupported predicate {op!r}")


def describe_quantity(expression, *, table_relative=False):
    kind = expression["kind"]
    if kind == "attribute":
        return "total surface area" if expression["attribute"] == "surface_area" else expression["attribute"]
    if kind == "constant":
        return f"{expression['value']:g} {expression['unit']}"
    if kind == "reference_attribute":
        return (
            f"the {expression['attribute']} of the {expression['reference'].replace('reference_', '')} reference block"
        )
    if kind in ("sum", "difference"):
        operation = "sum" if kind == "sum" else "difference"
        return f"the {operation} of " + " and ".join(
            describe_quantity(child, table_relative=table_relative) for child in expression["terms"]
        )
    if kind == "scale":
        return (
            f"{expression['factor']:g} times {describe_quantity(expression['terms'][0], table_relative=table_relative)}"
        )
    if kind == "coordinate":
        if table_relative:
            direction = ("right", "forward", "upward")[expression["axis"]]
            anchor = "the world origin" if expression["axis"] == 2 else "the center of the tabletop"
            return f"signed center position {direction} from {anchor}"
        return f"center's {'xyz'[expression['axis']]} coordinate in the table frame (red +x, blue +y)"
    if kind == "boundary_distance":
        side = (
            ("left", "front")[expression["axis"]]
            if expression["side"] == "lower"
            else ("right", "back")[expression["axis"]]
        )
        return f"nearest-surface distance from the table's {side} boundary"
    if kind == "orientation":
        ref = "the table's red +x axis" if "reference" not in expression else "the red reference block's length axis"
        if table_relative and "reference" not in expression:
            ref = "the table's long edges, viewed from above"
        return f"counterclockwise length-axis angle from {ref}, modulo 180 degrees,"
    label = expression["reference"].replace("reference_", "")
    if kind == "distance":
        return (
            f"{'planar' if expression['planar'] else '3D'} center-to-center distance from the {label} reference block"
        )
    if kind == "clearance":
        return f"surface-to-surface clearance from the {label} reference block"
    if kind == "offset":
        if table_relative:
            if expression["frame"] == "table":
                direction = ("right", "forward", "upward")[expression["axis"]]
                return f"signed center offset {direction} from the {label} reference block's center"
            if expression["axis"] == 0:
                return (
                    f"signed center offset along the {label} reference block's length, "
                    "positive toward its yellow-marked end"
                )
        axis = "xyz"[expression["axis"]]
        frame = "table frame" if expression["frame"] == "table" else f"{label} reference's marked local frame"
        return f"signed {axis} center offset from the {label} reference block in the {frame}"
    raise ValueError(f"Unsupported language expression {kind!r}")


def describe_predicate(predicate, *, table_relative=False):
    op = predicate["op"]
    if op == "and":
        return " and ".join(describe_predicate(child, table_relative=table_relative) for child in predicate["terms"])
    measure = describe_quantity(predicate["measure"], table_relative=table_relative)
    target = describe_quantity(predicate["target"], table_relative=table_relative)
    if op == "interval":
        low, high = predicate["bounds"]
        left, right = predicate["closed"]
        return (
            f"{measure} is {'at least' if left else 'greater than'} {low:g} and "
            f"{'at most' if right else 'less than'} {high:g} {predicate['unit']}"
        )
    if op == "tolerance":
        return f"{measure} is within {predicate['tolerance']:g} {predicate['unit']} of {target}"
    if op == "closest":
        return f"{measure} is closest to {target}"
    wording = {"eq": "equals", "gt": "is greater than", "ge": "is at least", "lt": "is less than", "le": "is at most"}
    clause = f"{measure} {wording[op]} {target}"
    expression, value = predicate["measure"], predicate["target"]
    if (
        table_relative
        and op == "eq"
        and value["kind"] == "constant"
        and (expression["kind"] == "coordinate" or expression["kind"] == "offset" and expression["frame"] == "table")
    ):
        direction = (("to the right of", "to the left of"), ("forward of", "backward of"), ("above", "below"))[
            expression["axis"]
        ][value["value"] < 0]
        anchor = (
            ("the world origin" if expression["axis"] == 2 else "the center of the tabletop")
            if expression["kind"] == "coordinate"
            else ("the " + expression["reference"].replace("reference_", "") + " reference block's center")
        )
        clause = f"center is {abs(value['value']):g} {value['unit']} {direction} {anchor}"
    if op == "eq" and "tolerance_si" in predicate:
        clause += f" within {predicate['tolerance']:g} {predicate['unit']}"
    return clause
