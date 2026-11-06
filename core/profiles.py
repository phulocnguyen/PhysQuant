# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit collection ranges and rendering profiles, without simulator imports."""

from copy import deepcopy

from environments import settings

CANDIDATE_COUNTS = tuple(range(3, 10))
RENDER_PROFILES = ("quality", "balanced")


def render_settings(profile):
    """Keep sensor geometry and physics fixed while choosing the rendering method."""
    if profile not in RENDER_PROFILES:
        raise ValueError(f"Unknown render profile {profile!r}; choose from {RENDER_PROFILES}")
    result = deepcopy(settings.RENDER_SETTINGS)
    if profile == "balanced":
        result["mode"] = "path_traced"
        result["path_tracing_samples"] = 4
    return result


def validate_count(count):
    if type(count) is not int or count not in CANDIDATE_COUNTS:
        raise ValueError("candidate_count must be an integer from 3 through 9")
