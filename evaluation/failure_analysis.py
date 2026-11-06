# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize measured rollout failures without inferring unobserved causes."""

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze(records: list[dict]) -> dict:
    """Separate execution errors, unsuccessful tasks, and recorded failure evidence."""
    outcomes, evidence = Counter(), Counter()
    failures = []
    for record in records:
        if record["status"] != "completed":
            outcome = "execution_error"
        else:
            outcome = "success" if record.get("success") else "task_failure"
        outcomes[outcome] += 1
        if outcome == "success":
            continue
        metrics = record.get("metrics", {})
        labels = []
        if metrics.get("wrong_object_indices"):
            labels.append("wrong_object_lifted")
        if metrics.get("distractors_preserved") is False:
            labels.append("distractor_moved")
        if metrics.get("stop_requested") and metrics.get("numerical_success") is False:
            labels.append("stop_before_goal")
        evidence.update(labels)
        failures.append(
            {
                **{key: record.get(key) for key in ("suite", "tier", "episode_id", "case", "error")},
                "outcome": outcome,
                "evidence": labels,
            }
        )
    return {"episodes": len(records), "outcomes": dict(outcomes), "evidence": dict(evidence), "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in sorted((args.results / "episodes").rglob("*.json"))]
    if not records:
        parser.error("No rollout episode records found")
    report = analyze(records)
    destination = args.output or args.results / "failure_analysis.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
