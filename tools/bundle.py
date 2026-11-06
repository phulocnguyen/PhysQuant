# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a relocatable PhysQuant evaluation bundle, materializing local asset links."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ("configs", "controllers", "core", "data_collection", "environments", "evaluation", "tasks", "tools")
REFERENCE = re.compile(r"@([^@]+\.usd[a-z]?)@")


def build_bundle(root: Path, manifests: Path, output: Path) -> dict:
    """Copy code and frozen scenes, resolving local background dependencies.

    Source scene directories include their textures, meshes, and material layers.
    NVIDIA robot assets still use the configured Isaac Lab asset root. Dataset
    recordings and training plans are not included.
    """
    root, manifests, output = root.resolve(), manifests.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    if output.is_relative_to(root):
        raise ValueError("Place bundles outside the PhysQuant source directory")
    plans = sorted(manifests.glob("*/*/plan.yaml"))
    if not plans:
        raise ValueError(f"No suite/tier plans found under {manifests}")
    backgrounds = set()
    for path in manifests.glob("*/*/scenarios/**/*.json"):
        background = json.loads(path.read_text()).get("appearance", {}).get("background")
        if background:
            layer = Path(background["usd"])
            if layer.is_absolute() or ".." in layer.parts:
                raise ValueError(f"Scenario background must be inside the project: {layer}")
            backgrounds.add(layer)
    output.mkdir(parents=True)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", ".cache", ".thumbs")
    for name in CODE:
        shutil.copytree(root / name, output / name, ignore=ignored)
    shutil.copytree(
        root / "assets", output / "assets", ignore=shutil.ignore_patterns("scenes", "__pycache__", ".cache", ".thumbs")
    )
    for path in [*root.glob("*.py"), root / "README.md", root / "TASKS.md", root / "Dockerfile"]:
        if path.is_file():
            shutil.copy2(path, output / path.name)
    shutil.copytree(manifests, output / "manifests/benchmark", symlinks=False, ignore=ignored)
    copied = {}
    for layer in sorted(backgrounds):
        original = root / layer
        destination = output / layer
        destination.parent.mkdir(parents=True, exist_ok=True)

        def relocate(match):
            source = Path(match.group(1))
            source = (source if source.is_absolute() else original.parent / source).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Missing background dependency: {source}")
            folder = source.parent
            if folder not in copied:
                collection = "interior_agent" if "InteriorAgent" in folder.parts else "robocasa"
                target = output / "assets/scenes/sources" / collection / folder.name
                shutil.copytree(folder, target, ignore=ignored)
                copied[folder] = target
            return "@" + os.path.relpath(copied[folder] / source.name, destination.parent) + "@"

        destination.write_text(REFERENCE.sub(relocate, original.read_text()))
    report = {"plans": len(plans), "backgrounds": len(backgrounds), "source_scenes": len(copied)}
    (output / "bundle.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifests", "--suites", dest="manifests", type=Path, default=ROOT / "manifests/benchmark")
    args = parser.parse_args()
    print(json.dumps(build_bundle(ROOT, args.manifests, args.output), indent=2))


if __name__ == "__main__":
    main()
