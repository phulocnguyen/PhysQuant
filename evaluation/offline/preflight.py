# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline task/configuration checks without starting simulation."""

import argparse
import json
from collections import Counter
from pathlib import Path

from data_collection.config import load_config
from data_collection.manifest import load_manifest, validate_manifest

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/collection_preflight.json")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    report = {"tasks": audit_versioned(args.seeds)}
    if args.manifest:
        plan = load_manifest(args.manifest)
        validate_manifest(plan)
        report["manifest"] = {"episodes": len(plan["episodes"]), "validated": True}
    report["scope"] = (
        "Offline geometry, configuration, placement and split checks only. "
        "Physical success and visible metric evidence require recorded episode review. "
        "A passing result is not a VLA accuracy or large-scale collection certification."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["tasks"]["passed"],
                **report["tasks"]["counts"],
                "errors": len(report["tasks"]["errors"]),
                "report": str(args.output),
            },
            indent=2,
        )
    )
    if not report["tasks"]["passed"]:
        raise SystemExit(1)


def audit_versioned(seeds=3):
    """Check every subgroup preset and every predicate case without asserting physical success."""
    from core.profiles import CANDIDATE_COUNTS
    from tasks.common.scenarios import InfeasibleScenario as InfeasibleVersioned
    from tasks.common.task import BenchmarkConfig
    from tasks.quantitative_constraint_grounding.scenarios import CASES as QCG_CASES
    from tasks.registry import SUBGROUPS, get_task, task_id
    from tasks.relational_understanding.scenarios import CASES as RU_CASES

    errors, excluded, counts = [], [], Counter()
    configs = []
    for path in sorted((ROOT / "configs/tasks").rglob("*.yaml")):
        import yaml

        subgroup = task_id(yaml.safe_load(path.read_text())["task"])
        task = get_task(subgroup)
        for count in (None,) if SUBGROUPS[subgroup]["engine"] != "selection" else CANDIDATE_COUNTS:
            configs.append(
                (
                    str(path.relative_to(ROOT)),
                    load_config(path, {} if count is None else {"candidate_count": count}, task.CONFIG_TYPE),
                )
            )
    for family, cases in (("relational_understanding", RU_CASES), ("quantitative_constraint_grounding", QCG_CASES)):
        for case in cases:
            angular = "orientation" in case
            volume = case == "derived_property"
            configs.append(
                (
                    f"{family}/{case}",
                    BenchmarkConfig(
                        task=family,
                        case=case,
                        value=45 if angular else 200 if volume else 20 if family == "relational_understanding" else 10,
                        gap=15 if angular else 25 if volume else 4 if family == "relational_understanding" else 2,
                        unit="degrees" if angular else "cm3" if volume else "cm",
                    ),
                )
            )
    for label, config in configs:
        for seed in range(712, 712 + seeds):
            config.seed = seed
            try:
                get_task(config.task).configure(config)
                counts["valid_scenes"] += 1
            except InfeasibleVersioned as error:
                counts["excluded_scenes"] += 1
                excluded.append(
                    {
                        "config": label,
                        "count": getattr(config, "candidate_count", 1),
                        "seed": seed,
                        "reason": str(error),
                    }
                )
            except (ValueError, TypeError, KeyError) as error:
                errors.append(
                    {"config": label, "count": getattr(config, "candidate_count", 1), "seed": seed, "error": str(error)}
                )
    return {"passed": not errors, "counts": dict(counts), "errors": errors, "excluded_cells": excluded}
