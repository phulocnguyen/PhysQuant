# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task subgroups and the shared engines that generate, run, and score them.

Each subgroup is one task ID and one package under ``tasks/``. The package owns
the subgroup's cases and case-specific logic. Subgroups that share a manipulation
protocol share an engine: ``selection`` picks and holds one object, ``control``
moves one object to a numerical goal, and ``multi_object`` manipulates several
objects in sequence.
"""

from importlib import import_module

SUBGROUPS = {
    "object_understanding": dict(title="Object Understanding", engine="selection"),
    "relational_understanding": dict(title="Relational Understanding", engine="selection"),
    "quantitative_constraint_grounding": dict(title="Quantitative Constraint Grounding", engine="selection"),
    "spatial_transformation": dict(
        title="Spatial Transformation",
        engine="control",
        tasks={"ST-1": "object_translation", "ST-2": "object_rotation"},
    ),
    "target_placement": dict(
        title="Target Placement",
        engine="control",
        tasks={"TP-1": "coordinate_placement", "TP-2": "clearance_placement"},
    ),
    "constrained_motion": dict(
        title="Constrained Motion",
        engine="control",
        tasks={"CM-1": "waypoint_following", "CM-2": "constrained_transport"},
    ),
    "sequencing_allocation_planning": dict(
        title="Sequencing and Allocation Planning",
        engine="multi_object",
        tasks={"SAP-1": "sequential_ordering", "SAP-2": "numerical_sorting"},
    ),
    "constructive_planning": dict(
        title="Constructive Planning",
        engine="multi_object",
        tasks={"CP-1": "target_construction", "CP-2": "relational_construction"},
    ),
    "feedback_driven_planning": dict(
        title="Feedback-Driven Planning",
        engine="multi_object",
        tasks={"FDP-1": "goal_conditioned_execution", "FDP-2": "state_dependent_planning"},
    ),
}
ENGINES = {
    "selection": "tasks.common.task",
    "control": "tasks.control.task",
    "multi_object": "tasks.multi_object.task",
}
SELECTION_SUBGROUPS = tuple(name for name, value in SUBGROUPS.items() if value["engine"] == "selection")
# Variant prefix (for example ``ST-1``) to the core task name stored in control and multi-object scenarios.
CORE_TASKS = {prefix: task for value in SUBGROUPS.values() for prefix, task in value.get("tasks", {}).items()}

# Names stored by plans, configurations, and recordings made before the subgroup layout.
LEGACY_SUBGROUPS = {
    "ou": "object_understanding",
    "ru": "relational_understanding",
    "sr": "quantitative_constraint_grounding",
    "spatial_reasoning": "quantitative_constraint_grounding",
}
LEGACY_ENGINES = {"spatial_understanding": "selection", "spatial_control": "control", "long_horizon": "multi_object"}
LEGACY_CORE_TASKS = {task: name for name, value in SUBGROUPS.items() for task in value.get("tasks", {}).values()}


def subgroup(name):
    """Return the subgroup for a current or legacy selection ``task``/``family`` field."""
    return LEGACY_SUBGROUPS.get(name, name)


def subgroup_of_case(case):
    """Return the control or multi-object subgroup that owns a variant such as ``ST-1A``."""
    prefix = case[:-1] if isinstance(case, str) else None
    for name, value in SUBGROUPS.items():
        if prefix in value.get("tasks", {}):
            return name
    raise ValueError(f"Unknown control or multi-object case {case!r}")


def task_id(name):
    """Return the subgroup task ID, accepting legacy IDs such as ``spatial_control/object_rotation``."""
    if not isinstance(name, str):
        return name
    category, separator, task = name.partition("/")
    if not separator:
        return subgroup(name)
    if category == "spatial_understanding":
        return subgroup(task)
    return LEGACY_CORE_TASKS.get(task, name) if category in LEGACY_ENGINES else name


def engine(scenario):
    """Return the engine name stored in a current or legacy scenario ``category`` field."""
    return LEGACY_ENGINES.get(scenario["category"], scenario["category"])


def scenario_subgroup(scenario):
    """Return the subgroup task ID of a frozen scenario."""
    if engine(scenario) == "selection":
        return subgroup(scenario["task"])
    return subgroup_of_case(scenario["case"])


def get_task(name):
    """Return the engine module that runs a subgroup."""
    name = task_id(name)
    if name not in SUBGROUPS:
        raise ValueError(f"Unknown task {name!r}; choose from {', '.join(SUBGROUPS)}")
    return import_module(ENGINES[SUBGROUPS[name]["engine"]])


def subgroup_module(name, module):
    """Import the case-specific ``module`` of a subgroup package, such as ``goals``."""
    return import_module(f"tasks.{task_id(name)}.{module}")
