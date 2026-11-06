# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dimension-aware SI conversion for Benchmark task specifications."""

import math

UNITS = {
    "m": (1, 1.0),
    "cm": (1, 0.01),
    "mm": (1, 0.001),
    "m2": (2, 1.0),
    "cm2": (2, 1e-4),
    "m3": (3, 1.0),
    "cm3": (3, 1e-6),
    "ml": (3, 1e-6),
}
DIMENSIONS = {
    "length": 1,
    "width": 1,
    "height": 1,
    "radius": 1,
    "diameter": 1,
    "inner_width": 1,
    "outer_width": 1,
    "handle_length": 1,
    "footprint_area": 2,
    "surface_area": 2,
    "volume": 3,
    "capacity": 3,
}


def to_si(value, unit, attribute):
    if attribute not in DIMENSIONS or unit not in UNITS or UNITS[unit][0] != DIMENSIONS[attribute]:
        raise ValueError("Unknown or dimensionally incompatible measurement/unit")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Measurements must be positive finite numbers")
    return float(value) * UNITS[unit][1]


def equal(a, b):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-12)
