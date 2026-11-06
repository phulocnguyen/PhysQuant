# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize per-episode results into success tables per suite, tier, and case.

Run with: uv run --no-sync python PhysQuant/evaluation/metrics.py RESULTS_DIR
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(successes, total, z=1.96):
    """95% Wilson score interval for a success rate."""
    if not total:
        return (0.0, 0.0)
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(records):
    def cell(rows):
        done = [r for r in rows if r["status"] == "completed"]
        successes = sum(bool(r.get("success")) for r in done)
        low, high = wilson(successes, len(done))
        result = dict(episodes=len(rows), completed=len(done), errors=len(rows) - len(done), successes=successes)
        result.update(success_rate=successes / len(done) if done else None, ci95=[round(low, 3), round(high, 3)])
        selections = [r for r in done if "selected_object_id" in r]
        if selections:
            result["selection_accuracy"] = sum(
                r.get("metrics", {}).get("correct_selection", False) for r in selections
            ) / len(selections)
        return result

    groups = defaultdict(list)
    for record in records:
        groups[(record["suite"], record["tier"])].append(record)
    summary = {}
    for (suite, tier), rows in sorted(groups.items()):
        cases = defaultdict(list)
        for row in rows:
            cases[row["case"]].append(row)
        summary.setdefault(suite, {})[tier] = dict(cell(rows), cases={c: cell(r) for c, r in sorted(cases.items())})
    return summary


def write_report(output):
    output = Path(output)
    records = [json.loads(p.read_text()) for p in sorted((output / "episodes").rglob("*.json"))]
    summary = summarize(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["| Suite | Tier | Episodes | Success | 95% CI | Errors |", "|---|---|---:|---:|---|---:|"]
    for suite, tiers in summary.items():
        for tier, row in tiers.items():
            rate = "—" if row["success_rate"] is None else f"{100 * row['success_rate']:.1f}%"
            lines.append(
                f"| {suite} | {tier} | {row['episodes']} | {rate} | "
                f"{100 * row['ci95'][0]:.1f}–{100 * row['ci95'][1]:.1f}% | {row['errors']} |"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    write_report(parser.parse_args().output)


if __name__ == "__main__":
    main()
