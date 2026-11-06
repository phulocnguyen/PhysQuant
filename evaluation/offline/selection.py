# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score explicit policy selections, separate from oracle manipulation collection."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def score(scenario, prediction):
    from tasks.common.predicates import quantity
    from tasks.common.scenarios import measurement, validate
    from tasks.registry import subgroup

    validate(scenario)
    source = prediction.get("source", "policy")
    if source not in ("policy", "oracle"):
        raise ValueError("Prediction source must be policy or oracle")
    success = prediction.get("manipulation_success")
    if success is not None and (
        type(success) is not bool or prediction.get("success_protocol") != scenario["manipulation_protocol"]
    ):
        raise ValueError(f"Manipulation results must explicitly use {scenario['manipulation_protocol']}")
    selected = next(
        (
            o
            for o in scenario["objects"]
            if o["object_id"] == prediction.get("selected_object_id") and o.get("selectable", True)
        ),
        None,
    )
    target = scenario["objects"][scenario["target_index"]]
    errors = {}
    if selected is not None:
        if "predicate" in scenario:
            predicate = scenario["predicate"]
            terms = predicate["terms"] if predicate["op"] == "and" else [predicate]
            for index, term in enumerate(terms):
                actual, expected = (quantity(obj, term["measure"], scenario) for obj in (selected, target))
                difference = abs(actual - expected)
                if term["measure"]["kind"] == "orientation":
                    difference = abs((actual - expected + np.pi / 2) % np.pi - np.pi / 2)
                errors[f"constraint_{index}"] = {"absolute_si": difference}
        else:
            errors = {
                attr: {"absolute_si": abs(measurement(selected, attr) - q["value_si"])}
                for attr, q in scenario["query"].items()
            }
    correct = bool(selected and selected["object_id"] == scenario["target_id"])
    return dict(
        scenario_id=scenario["scenario_id"],
        episode_id=scenario["episode_id"],
        case=scenario["case"],
        condition=scenario["condition"],
        source=source,
        candidate_count=scenario["candidate_count"],
        minimum_gap_si=scenario["minimum_gap_si"],
        target_rank=scenario.get("target_rank")
        if subgroup(scenario["task"]) != "quantitative_constraint_grounding"
        else None,
        target_shape=target["geometry"]["kind"],
        template_id=scenario["template_id"],
        template_split=scenario["template_split"],
        family=subgroup(scenario["task"]),
        selected_object_id=prediction.get("selected_object_id"),
        valid_selection=selected is not None,
        identification_correct=correct,
        metric_error=errors,
        manipulation_success=success,
    )


def summarize(rows):
    def mean(values):
        return float(np.mean(values)) if values else None

    policy = [r for r in rows if r["source"] == "policy"]
    by_case = defaultdict(list)
    errors = defaultdict(list)
    factors = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(list)
    for row in policy:
        by_case[row["case"]].append(row)
        for attr, error in row["metric_error"].items():
            errors[attr].append(error["absolute_si"])
        paired[row["scenario_id"]].append(row)
        for factor in ("family", "candidate_count", "minimum_gap_si", "target_rank", "target_shape", "template_split"):
            if row.get(factor) is not None:
                factors[factor][str(row[factor])].append(row["identification_correct"])
    pairs = [group for group in paired.values() if len(group) > 1]
    return dict(
        policy_predictions=len(policy),
        oracle_predictions=len(rows) - len(policy),
        target_identification_accuracy=mean([r["identification_correct"] for r in policy]),
        selected_object_metric_error_si={attr: mean(v) for attr, v in errors.items()},
        metric_error_valid_selection_coverage=sum(bool(r["metric_error"]) for r in policy),
        manipulation_success=mean([r["manipulation_success"] for r in policy if r["manipulation_success"] is not None]),
        by_factor={
            factor: {
                value: dict(count=len(results), identification_accuracy=mean(results))
                for value, results in groups.items()
            }
            for factor, groups in factors.items()
        },
        paired_scenarios=len(pairs),
        paraphrase_selection_consistency=mean(
            [
                all(r["valid_selection"] for r in group) and len({r["selected_object_id"] for r in group}) == 1
                for group in pairs
            ]
        ),
        by_case={
            k: dict(count=len(v), identification_accuracy=mean([r["identification_correct"] for r in v]))
            for k, v in by_case.items()
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from data_collection.manifest import load_manifest

    manifest = load_manifest(args.manifest)
    scenarios = {entry["scenario"]["episode_id"]: entry["scenario"] for entry in manifest["episodes"]}
    predictions = json.loads(args.predictions.read_text())
    rows = []
    seen = set()
    for prediction in predictions:
        key = (prediction["episode_id"], prediction.get("source", "policy"))
        if key in seen:
            raise ValueError("Duplicate prediction; aggregate repeated trials explicitly")
        seen.add(key)
        rows.append(score(scenarios[prediction["episode_id"]], prediction))
    report = dict(summary=summarize(rows), per_prediction=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
