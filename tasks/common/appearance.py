# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Freeze scene-level appearance independently of numerical targets and language."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml
from environments.randomization import resolve_randomization, sample_table_style

ROOT = Path(__file__).resolve().parents[2]


def appearance_seed(scene):
    """Separate task-family appearance streams without coupling them to the target."""
    seed = scene.get("seeds", {}).get("appearance", scene["scene_seed"])
    namespace = int.from_bytes(sha256(scene["case"].encode()).digest()[:4], "little")
    return int(np.random.SeedSequence([seed, namespace]).generate_state(1)[0])


def apply_appearance(scene):
    """Sample once at generation; matched queries and retries reuse these values."""
    profile = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/appearance.yaml").read_text())
    if not {"enabled", "table", "candidates"} <= set(profile) <= {"enabled", "table", "candidates", "background"}:
        raise ValueError("Appearance requires enabled, table, and candidates, and allows background")
    if type(profile["enabled"]) is not bool:
        raise ValueError("Appearance enabled must be true or false")
    if not profile["enabled"]:
        return
    table = resolve_randomization({"enabled": True, "table": profile["table"]})["table"]
    candidates = profile["candidates"]
    if set(candidates) != {"enabled", "options"} or type(candidates["enabled"]) is not bool:
        raise ValueError("Appearance candidates require enabled and options")
    options = candidates["options"]
    if not isinstance(options, list) or candidates["enabled"] and not options:
        raise ValueError("Candidate appearance needs a nonempty palette when enabled")
    names = set()
    for option in options:
        if set(option) != {"name", "color_rgb"} or not isinstance(option["name"], str) or not option["name"].isalpha():
            raise ValueError("Candidate colors require a single-word name and color_rgb")
        color = np.asarray(option["color_rgb"])
        if (
            color.shape != (3,)
            or color.dtype.kind not in "fi"
            or not np.isfinite(color).all()
            or np.any((color < 0) | (color > 1))
        ):
            raise ValueError("Candidate color_rgb must contain three finite linear RGB values in [0, 1]")
        if option["name"] in names:
            raise ValueError("Candidate color names must be unique")
        names.add(option["name"])
    seed = appearance_seed(scene)
    # Version 2 adds a background and renders table textures; version 1 scenes stay solid-color.
    sample = {"version": 2 if "background" in profile else 1, "seed": seed, "candidate": None, "table": None}
    if "background" in profile:
        sample["background"] = sample_background(profile["background"], seed)
    if candidates["enabled"]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        sample["candidate"] = deepcopy(options[int(rng.integers(len(options)))])
        for obj in scene["objects"]:
            if obj.get("selectable", True):
                obj["color_rgb"] = sample["candidate"]["color_rgb"].copy()
    cfg = resolve_randomization(scene["randomization"])
    if table["enabled"]:
        if not cfg["enabled"]:
            # Enabling appearance must not revive disabled physics/camera groups.
            for value in cfg.values():
                if isinstance(value, dict):
                    value["enabled"] = False
        cfg.update(enabled=True, table=table)
        sample["table"] = sample_table_style(cfg, seed)
        texture = sample["table"]["source_texture"]
        if sample["version"] >= 2 and texture and not (ROOT / texture).is_file():
            raise FileNotFoundError(f"Table texture is missing: {ROOT / texture}")
    scene["randomization"] = cfg
    scene["appearance"] = sample
    validate_appearance(scene)


def background_catalog(path):
    """Background records by name from a catalog written by tools/place_backgrounds.py."""
    entries = yaml.safe_load((ROOT / path).read_text())["backgrounds"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Background catalog must contain a nonempty list")
    names = set()
    for entry in entries:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key] for key in ("name", "collection", "room_type", "usd")
        ):
            raise ValueError("Background records need a name, collection, room type, and USD layer")
        if entry["name"] in names:
            raise ValueError("Background catalog names must be unique")
        names.add(entry["name"])
    return {entry["name"]: entry for entry in entries}


def sample_background(profile, seed):
    """Uniform seeded background choice, independent of the color and table streams."""
    if set(profile) != {"enabled", "catalog", "options"} or type(profile["enabled"]) is not bool:
        raise ValueError("Background appearance requires enabled, catalog, and options")
    if not profile["enabled"]:
        return None
    catalog = background_catalog(profile["catalog"])
    if profile["options"] is not None and (
        not isinstance(profile["options"], list) or not all(isinstance(name, str) for name in profile["options"])
    ):
        raise ValueError("Background options must be a list of catalog names or null")
    names = sorted(catalog) if profile["options"] is None else list(profile["options"])
    if not names or len(set(names)) != len(names) or set(names) - set(catalog):
        raise ValueError("Background options must be unique names from the catalog")
    choice = catalog[names[int(np.random.default_rng(np.random.SeedSequence([seed, 4])).integers(len(names)))]]
    if not (ROOT / choice["usd"]).is_file():
        raise FileNotFoundError(f"Background layer is missing: {ROOT / choice['usd']}")
    return {key: choice[key] for key in ("name", "collection", "room_type", "usd")}


def validate_appearance(scene):
    """Check frozen appearance without consulting today's palette or resampling."""
    sample = scene.get("appearance")
    if sample is None:
        return  # Historical fixed-appearance scenes retain their original identities.
    if sample["version"] not in (1, 2) or sample["seed"] != appearance_seed(scene):
        raise ValueError("Invalid appearance version/seed")
    if sample["version"] == 2:
        if "background" not in sample:
            raise ValueError("Version 2 appearance must record its background, including null for the studio")
        background = sample.get("background")
        if background is not None and (
            set(background) != {"name", "collection", "room_type", "usd"}
            or not all(isinstance(value, str) and value for value in background.values())
        ):
            raise ValueError("A frozen background needs a name, collection, room type, and USD layer")
    elif "background" in sample:
        raise ValueError("Version 1 appearance has no background")
    if sample["candidate"] is not None:
        color = sample["candidate"]["color_rgb"]
        for obj in scene["objects"]:
            if obj.get("selectable", True) and obj["color_rgb"] != color:
                raise ValueError("All selectable objects must share the frozen scene color")
            if not obj.get("selectable", True) and np.linalg.norm(np.asarray(obj["color_rgb"]) - color) < 0.15:
                raise ValueError("Candidate color must remain distinct from semantic references")
    if sample["table"] is not None and sample["table"] != sample_table_style(scene["randomization"], sample["seed"]):
        raise ValueError("Frozen table appearance differs from the recorded randomization profile")
