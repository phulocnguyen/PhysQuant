# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect every runtime asset into one self-contained folder for upload.

Included: the derived background layers and every file they reference (RoboCasa
kitchens, InteriorAgent apartments), the basket USD and its dependencies, and the
LIBERO table textures. Paths keep this project's layout, so unpacking the folder
over a checkout's ``PhysQuant/`` directory makes scenario paths resolve.
Derived layers are rewritten to reference their source scenes relatively.

Run with: uv run --no-sync python PhysQuant/tools/package_assets.py --output /path/to/physquant_assets
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REFERENCE = re.compile(r"@([^@]+\.usd[a-z]?)@")


def dependencies(root):
    """Every existing file a USD layer needs, including itself."""
    from pxr import Sdf, UsdUtils

    layers, assets, _ = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(root)))
    files = {Path(layer.realPath) for layer in layers if layer.realPath} | {Path(a) for a in assets}
    return {f for f in files if f.is_file()}


def source_destination(path):
    """Where a background source file goes in the package, keeping its scene folder layout."""
    text = str(path)
    for marker, collection in (("/Robocasa/", "robocasa"), ("/InteriorAgent/", "interior_agent")):
        if marker in text:
            return Path("assets/scenes/sources") / collection / text.split(marker, 1)[1]
    return Path(os.path.relpath(path, ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        parser.error(f"{out} exists; choose a new folder")
    copies = {}
    catalog = yaml.safe_load((ROOT / "configs/backgrounds.yaml").read_text())
    layers = sorted({ROOT / entry["usd"] for entry in catalog["backgrounds"]})
    for layer in layers:
        for path in dependencies(layer) - {layer}:
            copies[path] = source_destination(path)
    for basket in sorted((ROOT / "assets/baskets").rglob("*.usd*")):
        for path in dependencies(basket):
            copies[path] = Path(os.path.relpath(path, ROOT))
    appearance = yaml.safe_load((ROOT / "configs/appearance.yaml").read_text())
    for option in appearance["table"]["options"]:
        if option["source_texture"]:
            copies[ROOT / option["source_texture"]] = Path(option["source_texture"])
    copies[ROOT / "assets/textures/libero/README.md"] = Path("assets/textures/libero/README.md")

    for source, destination in sorted(copies.items()):
        (out / destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, out / destination)
    for layer in layers:
        text = layer.read_text()
        destination = layer.relative_to(ROOT)
        (out / destination).parent.mkdir(parents=True, exist_ok=True)

        def relocate(match):
            source = Path(match.group(1))
            source = (source if source.is_absolute() else layer.parent / source).resolve()
            target = source_destination(source)
            return "@" + os.path.relpath(target, destination.parent) + "@"

        (out / destination).write_text(SOURCE_REFERENCE.sub(relocate, text))
    for config in ("configs/backgrounds.yaml", "configs/appearance.yaml"):
        (out / config).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / config, out / config)

    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=0))
    total = sum(entry["bytes"] for entry in manifest)
    print(f"Packaged {len(manifest)} files, {total / 1e9:.2f} GB, {len(layers)} backgrounds, into {out}")


if __name__ == "__main__":
    main()
