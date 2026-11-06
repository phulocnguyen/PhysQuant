# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Freeze the evaluation suites under manifests/benchmark/<suite>/<tier>/.

Every tier uses the eval split (one attempt per episode) and its own seed range, so its
scenes never coincide with training plans generated from the default seed.

- t1_in_distribution: the training task configs
- t2_held_out_language: the held-out instruction templates
- t3_unseen_values (selection suites): each value shifted by half its distractor gap
- t5_more_candidates (selection suites): 8 or 9 candidates

Run with: uv run --no-sync python PhysQuant/tools/make_suites.py --per_case 20
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

from evaluation.suites import SELECTION_SUITES, SUITES, TIERS  # noqa: E402

TIER_SEEDS = {"t1_in_distribution": 9_000_000, "t2_held_out_language": 9_100_000}
TIER_SEEDS.update(t3_unseen_values=9_200_000, t5_more_candidates=9_300_000)


def feasible(path, overrides, counts):
    """True when the generator builds this task config for every count on several seeds."""
    from data_collection.config import load_config
    from tasks.common.scenarios import InfeasibleScenario
    from tasks.registry import get_task

    task = get_task(yaml.safe_load(path.read_text())["task"])
    for count in counts:
        for seed in (9_900_000, 9_900_001, 9_900_002):
            cfg = load_config(path, dict(overrides, candidate_count=count, seed=seed, split="eval"), task.CONFIG_TYPE)
            try:
                task.configure(cfg)
            except InfeasibleScenario:
                return False
    return True


def tier_configs(suite, tier, directory):
    """Task configs for one tier, written to ``directory``; returns (kept, excluded) case names.

    t3 shifts each value half a gap up, or down when up leaves the size limits. t5 keeps
    only the cases that fit 8 or 9 candidates. Other tiers use the configs unchanged.
    """
    kept, excluded = [], []
    for path in sorted((ROOT / "configs/tasks" / suite).glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        candidates = []
        if tier == "t3_unseen_values":
            for sign in (1, -1):
                shifted = dict(config, value=config["value"] + sign * config["gap"] / 2)
                if config.get("interval_bounds"):
                    shifted["interval_bounds"] = [b + sign * config["gap"] / 2 for b in config["interval_bounds"]]
                candidates.append((shifted, (3, 7)))
        elif tier == "t5_more_candidates":
            candidates.append((config, (8, 9)))
        else:
            candidates.append((config, ()))
        for candidate, counts in candidates:
            target = directory / path.name
            target.write_text(yaml.safe_dump(candidate, sort_keys=False))
            if not counts or feasible(target, {}, counts):
                kept.append(path.stem)
                break
            target.unlink()
        else:
            excluded.append(path.stem)
    return kept, excluded


def make(suite, tier, per_case, output, render_profile):
    selection = suite in SELECTION_SUITES
    if tier in ("t3_unseen_values", "t5_more_candidates") and not selection:
        return "skipped (selection suites only)"
    target = output / suite / tier
    if (target / "plan.yaml").exists():
        return "exists"
    with tempfile.TemporaryDirectory() as scratch:
        kept, excluded = tier_configs(suite, tier, Path(scratch))
        if not kept:
            return f"skipped (no feasible case; excluded {excluded})"
        error = "unknown error"
        # A rare scene draw can be infeasible; retry the tier from a fresh seed range.
        for attempt in range(3):
            command = [sys.executable, str(ROOT / "generate_manifest.py"), "--config", scratch]
            command += ["--output", str(target), "--episodes", str(per_case * len(kept)), "--split", "eval"]
            command += ["--seed", str(TIER_SEEDS[tier] + 10_000 * attempt), "--render_profile", render_profile]
            if tier == "t2_held_out_language":
                command += ["--template_split", "held_out"]
            if tier == "t5_more_candidates":
                command += ["--random_candidate_counts", "8", "9"]
            result = subprocess.run(command, capture_output=True, text=True)
            if not result.returncode:
                (target / "reports" / "tier.yaml").write_text(
                    yaml.safe_dump(dict(tier=tier, description=TIERS[tier], cases=kept, excluded_cases=excluded))
                )
                return f"ok ({len(kept)} cases" + (f"; excluded {excluded})" if excluded else ")")
            shutil.rmtree(target, ignore_errors=True)
            error = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
    return "failed: " + error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per_case", type=int, default=20, help="Episodes per task case and tier")
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    parser.add_argument("--tiers", nargs="+", default=list(TIERS), choices=list(TIERS))
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/benchmark")
    parser.add_argument("--render_profile", default="balanced", choices=("quality", "balanced"))
    args = parser.parse_args()
    for suite in args.suites:
        for tier in args.tiers:
            print(f"{suite}/{tier}: {make(suite, tier, args.per_case, args.output, args.render_profile)}", flush=True)


if __name__ == "__main__":
    main()
