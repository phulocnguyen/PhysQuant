# Control engine

The control engine runs three subgroups that execute numerical goals on one
movable object: **Spatial Transformation** (`spatial_transformation`),
**Target Placement** (`target_placement`), and **Constrained Motion**
(`constrained_motion`). Each subgroup has two core tasks with three scenario
families each. Configuration values, geometry options, initial poses,
references, and language templates generate episodes rather than fixed scenes.

| Subgroup | Core task | Families | Final state |
|---|---|---|---|
| Spatial Transformation | Object translation | ST-1A single axis; ST-1B two axes; ST-1C vertical | Release for A/B; held at requested elevation for C |
| Spatial Transformation | Object rotation | ST-2A canonical start; ST-2B varied start; ST-2C reference-relative | Release at starting center with requested yaw |
| Target Placement | Coordinate placement | TP-1A absolute; TP-1B one reference; TP-1C two references | Stable release at solved center |
| Target Placement | Clearance placement | TP-2A parallel faces; TP-2B cuboid/cylinder; TP-2C two gaps | Stable, axis-aligned release satisfying every gap |
| Constrained Motion | Waypoint following | CM-1A two segments; CM-1B three/four segments; CM-1C reference-relative | Visit ordered points while held, then release below last point |
| Constrained Motion | Constrained transport | CM-2A bottom height; CM-2B yaw retention; CM-2C obstacle separation | Compliant transport, then stable release at endpoint |

## File organization

- `configs/tasks/<subgroup>/<family-ID>.yaml`: initial geometry/poses,
  references and numerical goal for one scenario family. The YAML `task` is the
  subgroup ID, for example `task: spatial_transformation` with `case: ST-1A`.
- `configs/protocols/control.yaml`: scoring tolerances, dwell, timeout,
  transport height and command angular-speed limit.
- `configs/instructions/control.yaml`: seen and held-out language wrappers.
- `tasks/<subgroup>/goals.py`: each subgroup's exact target computation, route
  construction and numerical wording. All values come from configuration.
- `tasks/control/goals.py`: shared geometry helpers, and dispatch of each case
  to its subgroup's `goals.py`.

The remaining engine files are in `tasks/control/`:

- `scenarios.py`: sampling, geometry admission, reference construction,
  instruction assembly and immutable identities.
- `manifest.py`: balanced numerical-goal and template sweeps.
- `task.py`: registered task/config adapter into the existing collector.
- `evaluator.py`: state-based numerical, waypoint and continuous scoring.
- `validation.py`: replay and check physics-rate recording evidence.
- `controllers/single_object.py`: measured-object feedback layered on the
  existing grasp, bounded transport and supported release primitives.

## Conventions and scoring

New scenes use the unmarked tabletop as the reference. Absolute placement queries
name the center of the tabletop; relative commands remain simple, for example
"move it 10 cm right from its starting center." No axis legend is appended to
each query. Internally, +X is right, +Y forward and +Z up, with the robot at the
negative-Y table edge. This convention and the table dimensions are stored in
the scene metadata. Coordinates refer to geometric centers.
References are kinematic and are not candidates. Objects keep the
same sampled scene color from `configs/appearance.yaml`. No numerical target or waypoint marker is rendered.
Camera resolution remains 256×256 with the existing four calibrated views.

Template version 4 removes the drawn table axes. Version 1–3 plans retain their
original instructions and scene markers; new plans receive new scene identities
and cannot reuse a worker configured for the older markers.

Translation is relative to the **settled initial center**. Reference-relative
coordinates use the settled reference centers; TP-1C takes X from A and Y from B.
Clearance tasks specify direction, alignment and surface rather than center
separation. TP-2C constructs the second reference to satisfy both configured gaps.

Uniform rectangular long axes are equivalent modulo 180 degrees. Clockwise is
negative yaw viewed from above. Rotation scoring reports the accumulated signed
rotation as a diagnostic and uses symmetry-aware endpoint angle for success.
Direction-sensitive full rotations require a future marked-object protocol.
Plain circular cylinders cannot be admitted to yaw tasks.

Position/translation and gap tolerances start at 2 cm, yaw at 10 degrees and
waypoints at 3 cm. These are configurable engineering tolerances, not empirical
benchmark standards. Grasp evidence is explicitly a lift/proximity/closed-finger
proxy. Released success requires numerical accuracy, geometric separation from
an open hand, table support, upright pose, low linear/angular velocity and a
continuous 0.5 s dwell. ST-1C instead requires retention at the requested height.
Held stability uses finite differences of actual physics-step positions and
quaternions; raw solver velocities remain recorded. This avoids treating contact
solver velocity bias as visible rotation while still rejecting pose oscillation.

Waypoints are evaluated against measured object positions in order, at the
configured transport height. Skipping to a later point latches an order failure.
Waypoint tolerance regions must not overlap. The entire route must be completed;
a direct diagonal path to the endpoint cannot pass.
Instruction template version 2 states the height at each waypoint explicitly.
Version 1 pilot wording overstated this as a continuous height constraint; its
recordings remain diagnostic evidence and retain their original instructions.

CM-2 scoring arms after a held lift of 5 cm (and attainment of the minimum bottom
clearance for CM-2A). It starts at the first horizontal departure over 5 mm from
that acquired pose, independent of controller phase labels. Horizontal drift
over 2 cm before acquisition is a failure; small finger-closing adjustments do
not start transport. It ends after the held object settles within 1 cm of
the endpoint for 0.1 s, before lowering. Moving away reopens the interval. A
single violating physics sample remains a violation after recovery. The recorded
mask identifies precisely which samples were evaluated. Bottom height uses the
complete world AABB. Yaw is measured relative to the starting principal axis.
Obstacle clearance uses horizontal enclosing-AABB separation: a **conservative
lower bound**, not exact mesh distance. It includes complete object extents,
rejects center-distance shortcuts and prevents flying over the obstacle. The
baseline route goes around a single obstacle; changing to true 3D distance would
require a new protocol and scorer.

These rules are `spatial_control_v1` with `evidence_version: 2`. The protocol
ID is stored in plans and recordings, so it keeps its original name. The first
engineering pilot omitted that field; readers preserve its earlier interval
and raw-velocity stability interpretation. Generate new plans for version 2;
do not overwrite or relabel the pilot recordings.

The geometric preflight checks whole object bounds and a sampled primitive
route. Its reach envelope is a conservative heuristic, **not** an IK/collision
planner proof for the robot's whole arm. Real oracle trials and rendered review
remain required before accepting a configuration for collection at scale.

## Generate and collect

Run from `/home/locnp/IsaacLab` in the existing uv environment. These output paths
must be fresh; plans are immutable.

```bash
# Validate one family without simulation.
uv run --no-sync python PhysQuant/collect_data.py \
  --config PhysQuant/configs/tasks/spatial_transformation/ST-1A.yaml --check_config

# One episode for each of the 6 Spatial Transformation families.
uv run --no-sync python PhysQuant/generate_manifest.py \
  --config PhysQuant/configs/tasks/spatial_transformation \
  --one_template --output PhysQuant/manifests/my_pilot/spatial_transformation

# Collect with a 5-minute wall deadline per attempt and save four videos.
uv run --no-sync python PhysQuant/collect_manifest.py \
  PhysQuant/manifests/my_pilot/spatial_transformation/plan.yaml \
  --output PhysQuant/datasets/spatial_transformation/pilot \
  --gpu 0 --attempts_per_episode 1 --attempt_timeout_s 300 --video

uv run --no-sync python PhysQuant/view_videos.py \
  PhysQuant/datasets/spatial_transformation/pilot

# Replay trajectory scoring and summarize per-subgroup and per-family results.
uv run --no-sync python PhysQuant/evaluate_control.py \
  PhysQuant/datasets/spatial_transformation/pilot \
  --output PhysQuant/reports/my_control_results.json
```

Use `--gpus 0 1` for two workers when those devices are available. Compatible
geometry/settings share a worker and reset per episode. Incompatible topology
recycles the worker. Resume uses the existing `--resume` and exact plan identity.

Pass `--config PhysQuant/configs/tasks` to plan every subgroup at
once. The generator then writes one plan per subgroup under
`--output/<subgroup>/plan.yaml`.

To collect matched numerical goals, add to a single family YAML, for example:

```yaml
goal_sweep:
  - displacement_cm: [6, 0, 0]
  - displacement_cm: [10, 0, 0]
  - displacement_cm: [14, 0, 0]
```

Each goal and template is one episode with one instruction. The initial scene
and its identity stay fixed; target/instruction identities differ. Use
`--episodes 30` for an exact budget. With a config directory the total budget is
divided across its families. Exclusions are reported, never rescaled silently.
Use `split: eval` plus `template_split: held_out` for a language diagnostic.
The manifest validator rejects matched-geometry leakage across train/eval.

## Randomization and recording

`geometry_options`, `initial_xy_m`, `initial_jitter_xy_m`, `initial_yaw_deg`, and
per-reference `jitter_xy_m` expose geometric variation. Recompute goals after
sampling; invalid paths fail preflight. Baseline cameras, lighting,
physics and robot reset are fixed. Table and movable-object colors vary by scene
using `configs/appearance.yaml`; template version 3 names the sampled object
color. Earlier template versions and frozen recordings remain readable.
An optional `randomization_profile` can use
the existing camera/light/table/robot factors. Independent object randomization
is rejected because it would invalidate numerical goals. Candidate colors stay
uniform within a scene. Matched goals and retries reuse the frozen palette;
compatible workers update candidate materials and table color on reset.
Separate deterministic seed streams isolate geometry, placement,
language, camera, appearance and robot reset. Numerical goals are explicit or
swept; their reserved seed does not imply continuous numerical sampling.

Existing RGB-D, calibration, robot state/action and compact metadata recording
are retained. The control engine adds `control_metrics` and
`control_trace/*` at physics rate (120 Hz), with actual poses, velocities, full
bounds, hand/fingers, active constraint mask and violations. Action observations
remain 30 Hz; cameras remain 20 Hz. Validation recomputes bounds and replays the
trace instead of trusting controller milestones or success metadata alone.

Successful oracle collection is not evidence of VLA numerical understanding.
A pretrained VLA still needs an explicit observation/action embodiment adapter;
this change does not assume joint targets are interchangeable with its actions.
Visibility and scale-cue review are required before accepting demonstrations.
Experimental ranges, supported-platform vertical placement, fully direction-
sensitive rotations, continuous goal samplers, and arbitrary 3D orientation or
collision-planned routes are not qualified by the baseline pilot.
