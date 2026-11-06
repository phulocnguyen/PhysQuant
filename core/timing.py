# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent camera/control cadence with causal latest-frame alignment."""

import math


def capture_stride(physics_dt, control_steps, camera_hz):
    if physics_dt <= 0 or control_steps < 1 or camera_hz <= 0:
        raise ValueError("Physics/control/camera timing must be positive")
    interval = 1 / (physics_dt * camera_hz)
    if interval < 1 or not math.isclose(interval, round(interval), rel_tol=1e-8):
        raise ValueError("Camera capture rate must divide the physics rate exactly")
    if camera_hz > 1 / (physics_dt * control_steps) + 1e-8:
        raise ValueError("Camera capture rate cannot exceed the recorded action rate")
    return round(interval)
