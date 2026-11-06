# Task subgroups: scenes, randomization, and episode settings

The benchmark has nine task subgroups. This page describes what each subgroup's
scene contains, which values come from configuration, and what changes from one
episode to the next. All positions are in the **table frame**, in centimetres,
unless stated otherwise:

- Origin at the tabletop centre, 75 cm above the floor.
- **+X** points right (along the long table edges), **+Y** points away from the
  robot-side edge, and **+Z** points up.
- The frame is not drawn on the table. Instructions refer to the table centre,
  table edges, or coloured reference objects.

## Subgroups

Each subgroup is one task ID (the `task:` field of its YAML files), one package
under [tasks/](tasks/), and one configuration folder under
[configs/tasks/](configs/tasks/). The package owns the subgroup's cases and
case-specific logic. Subgroups that share a success protocol share an engine,
which generates, runs, and scores their episodes.

| Subgroup (task ID) | Cases | Engine and protocol |
|---|---|---|
| [Object Understanding](#object-understanding) (`object_understanding`) | 7 attributes | `tasks/common`, `pick_hold_v1` |
| [Relational Understanding](#relational-understanding) (`relational_understanding`) | 8 relations | `tasks/common`, `pick_hold_v1` |
| [Quantitative Constraint Grounding](#quantitative-constraint-grounding) (`quantitative_constraint_grounding`) | 10 constraints | `tasks/common`, `pick_hold_v1` |
| [Spatial Transformation](#spatial-transformation) (`spatial_transformation`) | ST-1A–C, ST-2A–C | `tasks/control`, `spatial_control_v1` |
| [Target Placement](#target-placement) (`target_placement`) | TP-1A–C, TP-2A–C | `tasks/control`, `spatial_control_v1` |
| [Constrained Motion](#constrained-motion) (`constrained_motion`) | CM-1A–C, CM-2A–C | `tasks/control`, `spatial_control_v1` |
| [Sequencing and Allocation Planning](#sequencing-and-allocation-planning) (`sequencing_allocation_planning`) | SAP-1A–C, SAP-2A–C | `tasks/multi_object`, `long_horizon_v1` |
| [Constructive Planning](#constructive-planning) (`constructive_planning`) | CP-1A–C, CP-2A–C | `tasks/multi_object`, `long_horizon_v1` |
| [Feedback-Driven Planning](#feedback-driven-planning) (`feedback_driven_planning`) | FDP-1A–C, FDP-2A–C | `tasks/multi_object`, `long_horizon_v1` |

The first three subgroups select and lift exactly one candidate. The next three
manipulate exactly one movable object. The last three manipulate several objects
in one episode. Protocol IDs keep their original names because frozen plans and
recordings store them. Earlier task IDs, such as `spatial_understanding/spatial_reasoning`
or `spatial_control/object_rotation`, still resolve to their subgroup.

## Shared scene

Every task uses the same table, robot, cameras, lighting, and physics settings
from [configs/settings.yaml](configs/settings.yaml). Changing this file changes
the settings digest stored in every scenario, so existing plans must be
regenerated.

| Component | Setting |
|---|---|
| Table | 160 × 100 × 5 cm top on four 6 cm legs, top surface at 75 cm. |
| Robot | Franka Panda (high-PD) with a parallel gripper. Base at (0, −40) cm on the robot-side long edge, facing +Y. TCP 10.34 cm below the hand frame. |
| Control | Differential IK (DLS) with joint-position targets. Physics at 120 Hz, one action every four steps (30 Hz). |
| Settling | Objects start asleep (CPU PhysX). Each reset settles for 1–10 s until speeds stay below 1 mm/s and 0.01 rad/s for 0.5 s. |
| Cameras | `top`, `left`, and `right` scene views (45° FOV) and a `wrist` view on `panda_hand` (75° FOV). 256 × 256 RGB plus depth, captured at 20 Hz. |
| Lighting | Dome light (350) and distant key light (650). Cast shadows are off. |
| Rendering | `quality`: path traced, 16 samples per pixel. `balanced`: path traced, 4 samples per pixel. Camera geometry and physics are identical in both profiles. |
| Object physics | Every dynamic object has a mass of 0.05 kg. References in the single-object subgroups, and baskets and references in the planning subgroups, are kinematic. |

### Object size limits

Selectable objects stay inside the candidate limits in `settings.yaml`. These
limits keep objects visible and allow a Panda grasp.

| Limit | Value |
|---|---|
| Smallest extent | 3.5 cm |
| Smallest planar span | 4 cm |
| Largest extent | 18 cm |
| Largest grasp width | 7 cm |

The default secondary dimensions for generated shapes, such as box footprints
and cylinder heights, come from [configs/object_sizes.yaml](configs/object_sizes.yaml).

### Success protocols

| Protocol | Subgroups | Success requires |
|---|---|---|
| `pick_hold_v1` | Object Understanding, Relational Understanding, Quantitative Constraint Grounding | Lift the target at least 5 cm and hold it still for 0.5 s. Lifting another object at least 2 cm is a wrong selection; other objects must move less than 2 cm. Timeout 15 s. |
| `spatial_control_v1` | Spatial Transformation, Target Placement, Constrained Motion | Position within 2 cm, rotation within 10°, clearance within 2 cm, waypoints within 3 cm. Then release with the object supported and still for 0.5 s (ST-1C holds instead). References must move less than 0.5 cm. Timeout 30 s. |
| `long_horizon_v1` | Sequencing and Allocation, Constructive, and Feedback-Driven Planning | Position within 1.5 cm, rotation within 10°, structure dimensions and join gaps within 0.5 cm, alignment within 0.6 cm. Every placement released and stable, and an explicit stop. Unused objects must move less than 0.5 cm. Timeout 120 s. |

### Single-object scenes

Spatial Transformation, Target Placement, and Constrained Motion use one movable
object and up to two kinematic references: A is red and B is blue, both
8 × 5 × 6 cm boxes unless stated otherwise. The movable is a 10 × 4.5 × 5 cm box
at (−12, 8) cm with ±1.5 cm jitter and 0° yaw, unless stated otherwise. It uses
the scene's candidate color, and the instruction names that color. Every initial
pose, goal pose, and oracle route must be on the table, 28–75 cm from the robot
base, and clear of the references.

**Per-episode variation.** Seed-driven jitter, the instruction template, and
appearance. Task YAMLs can add a `goal_sweep` list to rotate through several
numerical goals.

### Multi-object scenes

Sequencing and Allocation, Constructive, and Feedback-Driven Planning start
several movable boxes in a front row, with baskets, building space, or one
reference. The objects are placed in 5 spawn slots, shuffled per episode:
(−26, −9), (0, −9), (26, −9), (−34, −20), and (34, −20) cm, each with ±0.5 cm
jitter and 0° yaw. Each task YAML defines two solution-first problems, and
episodes alternate between them. Case tables give values as problem 1 / problem 2.

| Support | Placement |
|---|---|
| Baskets | 27.6 × 27.6 × 4.9 cm tray (24 × 24 cm inside) at Y = 16 cm. 1 basket: X = 0. 2 baskets: X = ±30 cm. 3 baskets: X = −43, 0, 43 cm. |
| Construction sites | Left (−15, 20) cm and right (15, 20) cm. |
| Chain reference (FDP-2) | Red 5 cm cube at (−25, 20) cm. |

Unless stated otherwise, the boxes have a 5 × 4 cm footprint and only their
height changes. Length-based cases use 4 × 5 cm sections with different lengths.
Structures and collections must use at least two objects.

## Randomization

Randomization comes from two places:

1. **Scene generation** (frozen in each scenario JSON). Layout, candidate
   count, target choice, instruction template, and appearance are sampled once
   from the scene seed. Retries of the same episode reuse these values.
2. **Reset randomization** (applied by `EpisodeRandomizer` at reset). Camera,
   light, robot joint, and object mass/pose offsets apply only when the
   scenario's `randomization` block enables them.

### Current status

| Factor | Status in current plans | Range or options | Source |
|---|---|---|---|
| Object layout | **On** | Sampled per subgroup; see the subgroup sections below. | Subgroup generator |
| Candidate count (first three subgroups) | **On** | 3–7, sampled uniformly per episode (the generator accepts 3–9). | `generate_manifest.py --random_candidate_counts` |
| Target object and query value | **On** | Object and Relational Understanding: any candidate can be the target, and the instruction value becomes its measurement. Quantitative Constraint Grounding: the configured target is fixed. | `tasks/common/manifest.py` |
| Instruction template | **On** | 3 seen and 2 held-out templates for the first three subgroups; 2 seen and 1 held-out for the other six. | `configs/instructions/` |
| Tabletop surface | **On** | 10 equally likely styles: the four existing solid colors plus six LIBERO wood/marble textures. | [configs/appearance.yaml](configs/appearance.yaml) |
| Room background | **On** | 76 equally likely placements: 70 RoboCasa kitchens and 6 InteriorAgent living rooms. | [configs/backgrounds.yaml](configs/backgrounds.yaml) |
| Candidate color | **On** | `orange`, `teal`, `purple`, `pink`; one color shared by all selectable objects in a scene. | [configs/appearance.yaml](configs/appearance.yaml) |
| Reference colors | Fixed | Red reference A and blue reference B. Candidate colors must differ from them by at least 0.15 (linear RGB distance). | Task generator |
| Basket color, yaw, position (planning subgroups) | **On** | 3 dark colors, yaw ∈ {−3°, 0°, +3°}, ±0.3 cm jitter. | `MultiObjectConfig` |
| Object pose, mass, and CoM offsets | **Off** | When enabled: ±position range, ±yaw range, mass range, ±CoM range. | `target_object` group |
| Camera pose and FOV | **Off** | When enabled: ±5 cm position, ±5° FOV for scene cameras. The wrist camera is not changed. | `camera` group |
| Light intensity | **Off** | When enabled: scale every light by a factor sampled from the configured range. | `light` group |
| Robot initial joints | **Off** | When enabled: ±0.09 rad on arm joints, clamped to soft limits. | `robot` group |

### Room backgrounds and tabletop surfaces

The background is a visual-only room around the existing workcell. Table, robot,
objects, camera mounts, and task coordinates stay at their configured poses.
Derived room layers position the workcell in open floor space and disable source
physics, lights, and cameras. InteriorAgent ceilings are removed so they do not
block the shared lighting. Two unavailable off-room pillow assets in apartment
0003 are disabled. Source assets are left unchanged.

New scenes sample a room from [backgrounds.yaml](configs/backgrounds.yaml).
`background.options: null` uses all 76 entries; a list of catalog
names restricts the selection. Set `background.enabled: false` in
[appearance.yaml](configs/appearance.yaml) to use the studio floor instead.
The room draw is independent of the table, candidate-color, target, and wording
choices. Retries and matched queries keep the same frozen room and surface.

The ten table options are `slate`, `warm_wood`, `light_stone`, `charcoal`,
`libero_wood`, `libero_wood_planks`, `libero_light_wood`, `libero_marble`,
`libero_walnut`, and `libero_oak_desk`. The six LIBERO options render image
textures on the tabletop and legs using world-space projection, with a 0.5 m
repeat. Walnut and oak use extracted wood panels; three wood textures are
mirror-tiled to remove seams. Texture source and license notes are in
[the texture README](assets/textures/libero/README.md).

Appearance version 2 freezes the chosen room and texture in the scenario. Both
the single-environment and parallel collectors apply these at reset and record
what was applied. Parallel environments are spaced using the full room bounds
to keep adjacent rooms out of the camera views. Room assets use USD instances
across environments; large rooms still increase rendering cost and GPU memory.

Older appearance-version-1 plans retain their original preview materials,
solid colors, and studio background. Generate new plans to use room backgrounds
and rendered textures. Preparing additional rooms is an offline step with
[place_backgrounds.py](tools/place_backgrounds.py); use fresh catalog and asset
output paths and inspect all four camera views before enabling new entries.

### Enabling reset randomization

[configs/domain_randomization.yaml](configs/domain_randomization.yaml) defines a
profile with the `camera`, `light`, `table`, `target_object`, and `robot` groups.
A task uses it only when its YAML sets:

```yaml
randomization_profile: configs/domain_randomization.yaml
```

Without that line, the task starts with all reset randomization disabled. The
appearance profile independently enables table surfaces, candidate colors, and room backgrounds.

Some groups are rejected because they would change the answer or the solved
goal:

| Subgroups | Allowed `target_object` randomization |
|---|---|
| Object Understanding | Position, yaw, mass, and CoM offsets. Footprints are resampled until they fit on the table without overlap. |
| Relational Understanding, Quantitative Constraint Grounding | Mass and CoM only. Position or yaw ranges above zero are rejected. |
| The six single-object and planning subgroups | None. The whole group must stay disabled. |

Camera, light, and robot groups are supported by the single-environment collector.
The parallel collector rejects these groups because their reset state is not yet
independent per environment.

### Seeds

- The scene seed is the task YAML `seed` (712) plus the episode offset in the plan.
- Appearance uses a separate stream, hashed from the seed and the task case.
  Matched queries on the same scene therefore keep the same colors.
- The single-object subgroups split the seed into independent streams for geometry,
  placement, numerical goal, appearance, camera, language, and robot.

## Object Understanding

The scene contains only candidates, with no reference objects. Candidates differ
in one geometric property, and the instruction gives the required value directly,
for example *"Lift the object whose diameter is 5.1 cm off the table."*

**Layout.** Candidates fill a random subset of a 4 × 3 grid: X ∈ {−24, −8, 8, 24} cm
and Y ∈ {4, 19, 34} cm. Each candidate rests on the table with its yaw matched to
the robot base (90°). The generator requires 2.5 cm between bounding boxes and
at most 78 cm from the robot base.

**Values.** Candidate *i* gets `value + (i − rank) × gap`, where `rank` is
`count // 2`. Candidates are then shuffled. All other dimensions use the
`object_sizes.yaml` defaults.

| Task | Config | Shapes | Base value | Gap | Planned values (3–7 candidates) |
|---|---|---|---|---|---|
| Length | [length.yaml](configs/tasks/object_understanding/length.yaml) | Box | 12 cm | 2 cm | 6–18 cm |
| Width | [width.yaml](configs/tasks/object_understanding/width.yaml) | Box | 5 cm | 0.4 cm | 3.8–6.2 cm |
| Height | [height.yaml](configs/tasks/object_understanding/height.yaml) | Cylinder | 10 cm | 2 cm | 4–16 cm |
| Radius | [radius.yaml](configs/tasks/object_understanding/radius.yaml) | Cylinder, 12 cm tall | 2.75 cm | 0.2 cm | 2.15–3.35 cm |
| Diameter | [diameter.yaml](configs/tasks/object_understanding/diameter.yaml) | Cylinder, 12 cm tall | 5.5 cm | 0.4 cm | 4.3–6.7 cm |
| Surface area | [surface_area.yaml](configs/tasks/object_understanding/surface_area.yaml) | Box and cylinder | 300 cm² | 20 cm² | 240–360 cm² |
| Volume | [volume.yaml](configs/tasks/object_understanding/volume.yaml) | Box | 200 cm³ | 25 cm³ | 125–275 cm³ |

**Episode objective.** Pick the matching object and hold it (`pick_hold_v1`).

## Relational Understanding

Candidates are identical boxes. They differ only in position or orientation
relative to the table or to a reference. Candidates are 6 × 4 × 6 cm boxes; in
the two orientation tasks they are 10 × 4 × 6 cm. The red reference is an
8 × 4 × 6 cm box at (−24, 2) cm with a yellow local-axis marker. It is rotated
45° in *Relative orientation* and *Reference frame*.

**Layout.** Each candidate's measured coordinate is solved from its value. Any
free coordinate is sampled uniformly from X ∈ [−45, 32] cm and Y ∈ [−12, 32] cm.
Each candidate must be 32–78 cm from the robot base, have a 2 cm gap to every
other object, and stay 1 cm inside the table edge. Candidate values follow
`value + (i − rank) × gap`, as in Object Understanding.

| Task | Reference | How the candidate is placed | Base value / gap | Tolerance |
|---|---|---|---|---|
| Absolute position | None | X = value (centre to the right of the table centre); Y is free. | 20 / 4 cm | 0.1 cm |
| Directional offset | Red, 0° | X offset from the reference = value; Y offset sampled in [−20, 30] cm. | 20 / 4 cm | 0.1 cm |
| Center distance | Red, 0° | Random bearing; planar centre distance = value (at least 6 cm). | 20 / 4 cm | 0.1 cm |
| Surface clearance | Red, 0° | Random bearing; distance solved so the box-to-box surface gap = value. | 20 / 4 cm | 0.1 cm |
| Boundary distance | None | Gap from the left table edge (X = −80 cm) to the near face = value. | 20 / 4 cm | 0.1 cm |
| Orientation | None | Yaw relative to the long table edges = value; position is free. | 45 / 15° | 0.5° |
| Relative orientation | Red, 45° | Yaw relative to the reference's long axis = value. | 45 / 15° | 0.5° |
| Reference frame | Red, 45° | Offset along the reference's local X axis = value; local Y offset sampled in [−20, 30] cm. | 20 / 4 cm | 0.1 cm |

Planned values are 8–32 cm for distances and 0–90° for angles.

**Episode objective.** Pick the matching object and hold it (`pick_hold_v1`).

## Quantitative Constraint Grounding

Formerly *Spatial Reasoning*. Candidates are boxes with a 6 × 4 cm footprint.
Height is the measured attribute unless stated otherwise. One configured
predicate selects exactly one candidate.
In current plans the target and the instruction values are fixed per task; only
layout, count, appearance, and template vary.

**Layout.** Free candidates are sampled from X ∈ [−32, 32] cm and
Y ∈ [−10, 32] cm with the same reach and gap rules as Relational Understanding.
Distractor heights alternate around the target: target ± 1 × gap, ± 2 × gap, and so on.

| Task | Rule in the instruction | Target | Distractors | References |
|---|---|---|---|---|
| Threshold | Height greater than 12 cm | 13 cm | 11, 10, 9, … cm | None |
| Interval | Height in [9.5, 10.5] cm | 10 cm | 8, 12, 6, 14, … cm | None |
| Tolerance | Height 10 ± 0.5 cm | 10 cm | 8, 12, 6, 14, … cm | None |
| Multiple attributes | Height 10 cm and width 4 cm | 10 cm tall, 4 cm wide | One decoy is 10 cm tall but 6 cm wide; others differ in height. | None |
| Object relation | Height 10 cm and 20 cm from the red reference | 10 cm tall, 20 cm away | One decoy has the height at 26 cm; another has the distance with a different height. | Red 8 × 4 × 6 cm at (−24, 2) |
| Multiple references | 20 cm right of red and 15 cm forward of blue | At (−5, 15) cm | Each decoy matches one offset only. All candidates are 6 cm tall. | Red at (−25, 0), blue at (25, 0) |
| Arithmetic | Height = red height + blue height | 10 cm | 8, 12, 6, 14, … cm | Red 4 cm tall at (−40, −8), blue 6 cm tall at (−40, 20) |
| Ratio | Height = 2 × red height | 10 cm | 8, 12, 6, 14, … cm | Red 5 cm tall at (−40, −8) |
| Derived property | Volume = 2 × red volume | 200 cm³ | 175, 225, 150, 250, … cm³ | Red 8 × 4 × 3.125 cm (100 cm³) at (−40, −8) |
| Nearest match | Height closest to 10 cm | 10.4 cm | 8, 12, 6, 14, … cm | None |

**Episode objective.** Pick the object that satisfies the constraint and hold it (`pick_hold_v1`).

## Spatial Transformation

Move one object by a numerical displacement (ST-1) or rotate it by a numerical
angle (ST-2). Scene layout follows [Single-object scenes](#single-object-scenes).
Goals are solved in [tasks/spatial_transformation/goals.py](tasks/spatial_transformation/goals.py).

| Case | Scene | Numerical goal |
|---|---|---|
| ST-1A | Movable only | Move +10 cm in X and release. |
| ST-1B | Movable only | Move (+8, +8) cm and release. |
| ST-1C | Movable only | Lift 10 cm and hold. |
| ST-2A | Movable at 0° | Rotate to 45° and release at the starting centre. |
| ST-2B | Initial yaw ∈ {−30°, 30°, 60°} | Rotate to −45° and release at the starting centre. |
| ST-2C | A at (20, 18), yaw 30° | Rotate to 45° relative to A. |

Rotation cases need a visibly elongated box; long axes are equivalent modulo 180°.

## Target Placement

Place one object at a numerical position (TP-1) or surface clearance (TP-2).
Scene layout follows [Single-object scenes](#single-object-scenes). Goals are
solved in [tasks/target_placement/goals.py](tasks/target_placement/goals.py).

| Case | Scene | Numerical goal |
|---|---|---|
| TP-1A | Movable only | Place the centre at (12, 16) cm. |
| TP-1B | A at (−6, 16) | Place the centre +14 cm in X from A's centre. |
| TP-1C | A at (−8, 24), B at (23, 28) | Centre X = A + 14 cm, centre Y = B − 12 cm. |
| TP-2A | A at (−8, 18) | 6 cm surface gap on A's +X side, centres aligned. |
| TP-2B | 5.5 cm cube at (−18, 0); A is a cylinder (r 4 cm, h 10 cm) at (−8, 18) | 6 cm surface gap on A's +X side. |
| TP-2C | A at (−15, 18); B placed to fit the solution | Between A and B with gaps of 6 cm to A and 8 cm to B. |

Clearance placements release the object with its faces parallel to the table edges.

## Constrained Motion

Carry one object through waypoints (CM-1) or under a motion constraint (CM-2).
Scene layout follows [Single-object scenes](#single-object-scenes). Goals are
solved in [tasks/constrained_motion/goals.py](tasks/constrained_motion/goals.py).

| Case | Scene | Numerical goal |
|---|---|---|
| CM-1A | Movable only | Carry through steps (0, +10), then (+10, 0) cm. |
| CM-1B | Movable only | Carry through steps (0, +12), (+12, 0), then (0, −8) cm. |
| CM-1C | A at (−10, 24), B at (22, 18) | Waypoints A + (0, −14) cm, then B + (−14, 0) cm. |
| CM-2A | Movable only | Move +14 cm in X while keeping its bottom at least 10 cm above the table. |
| CM-2B | Movable only | Move +14 cm in X while keeping yaw within 10° of its start. |
| CM-2C | 5 cm cube with no jitter; obstacle A (6 × 7 × 12 cm) at (4, 8) | Move +32 cm in X while staying at least 5 cm horizontally from A (detour to +Y). |

In CM-1 cases the object's bottom must be at least 15 cm above the table at each
waypoint. Objects are released on the table below the last point.

## Sequencing and Allocation Planning

Place objects in a requested order (SAP-1) or sort them into baskets by
numerical rules (SAP-2). Scene layout follows [Multi-object scenes](#multi-object-scenes).
Plans and wording are in [tasks/sequencing_allocation_planning/planning.py](tasks/sequencing_allocation_planning/planning.py).

| Case | Objects | Goal |
|---|---|---|
| SAP-1A | Heights {4, 6, 8} / {5, 7, 9} cm, 1 basket | Basket order 6, 4, 8 / 7, 5, 9 cm. |
| SAP-1B | Same as SAP-1A | All in ascending / descending height. |
| SAP-1C | Heights {4, 6, 8, 10} / {5, 7, 9, 11} cm | Place 8, 4, 6 / 9, 5, 7 in order; leave one untouched. |
| SAP-2A | Heights {4, 6, 8, 10} / {5, 7, 9, 11} cm, 2 baskets | Split at 8 / 9 cm. |
| SAP-2B | Same objects, 3 baskets | Split at 6 and 10 / 7 and 11 cm. |
| SAP-2C | Lengths {10, 6} / {11, 7} × widths {3.5, 5.5} cm, 2 baskets | Length > 8 / 9 cm **and** width < 4.5 cm goes left; the rest goes right. |

## Constructive Planning

Combine objects into one structure (CP-1) or two related structures (CP-2).
Scene layout follows [Multi-object scenes](#multi-object-scenes). Plans and
wording are in [tasks/constructive_planning/planning.py](tasks/constructive_planning/planning.py).
Every valid component combination is accepted.

| Case | Objects | Goal |
|---|---|---|
| CP-1A | Heights {4, 5, 7} / {5, 6, 8} cm | One tower 9 / 11 cm tall. |
| CP-1B | Lengths {6, 8, 12} / {7, 9, 13} cm | One row 20 / 22 cm long. |
| CP-1C | Lengths {6, 8, 10, 12} / {7, 9, 11, 13} cm | One row 18 / 20 cm long. |
| CP-2A | Heights {3.5, 4, 5, 5.5} / {4.5, 5, 6, 6.5} cm | Two towers, both 9 / 11 cm. |
| CP-2B | Heights {4, 5, 6, 7} / {5, 6, 7, 8} cm | Left tower 9 / 11 cm; right tower 4 cm taller. |
| CP-2C | Lengths {6, 8, 10, 12} / {7, 9, 11, 13} cm | Rows of 14 and 22 / 16 and 24 cm. |

## Feedback-Driven Planning

Add objects until a cumulative target is reached, then stop (FDP-1), or place
each object relative to the achieved pose of the previous one (FDP-2). Scene
layout follows [Multi-object scenes](#multi-object-scenes). Plans and wording are
in [tasks/feedback_driven_planning/planning.py](tasks/feedback_driven_planning/planning.py).

| Case | Objects | Goal |
|---|---|---|
| FDP-1A | Lengths {6, 8, 12} / {7, 9, 13} cm, 1 basket | Add until total length is 20 / 22 cm. |
| FDP-1B | Heights {4, 6, 8} / {5, 7, 9} cm, 1 basket | Add until total volume is 200 / 240 cm³. |
| FDP-1C | Heights {4, 5, 7} / {5, 6, 8} cm | Grow one tower to 9 / 11 cm, checking after each placement. |
| FDP-2A | Heights {4, 6, 8} / {5, 7, 9} cm, reference | Centre offsets +12, then +10 / +13, then +11 cm in X, each from the previous block's achieved pose. |
| FDP-2B | Same as FDP-2A | Surface gaps of 6, 6 / 7, 7 cm, each from the previous block. |
| FDP-2C | Same as FDP-2A | Offsets (10, 0), (0, 10), (12, 0) / (11, 0), (0, 11), (13, 0) cm. |
