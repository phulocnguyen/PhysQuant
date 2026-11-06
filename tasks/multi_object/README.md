# Multi-object engine

The multi-object engine runs three planning subgroups:
**Sequencing and Allocation Planning** (`sequencing_allocation_planning`, SAP),
**Constructive Planning** (`constructive_planning`, CP), and
**Feedback-Driven Planning** (`feedback_driven_planning`, FDP). Their six tasks,
each with three variants, share the existing Franka scene, calibrated
four-camera RGB-D recorder, persistent workers and retry handling. Each episode
contains the whole multi-object instruction, not one episode per pick.

Template version 3 uses the natural table edges without drawing table axes or
appending an XYZ definition. Row-building instructions refer to the table's long
edges; relative placements name their reference object. Older frozen plans keep
their original marker-based wording and visuals.

## Organization

| File | Responsibility |
|---|---|
| `configs/tasks/<subgroup>/<case>.yaml` | Candidate geometry and numerical problem variants; `task` is the subgroup ID |
| `configs/assets/basket.yaml` | Local tray asset, physical cavity and placement patterns |
| `configs/protocols/multi_object.yaml` | Release, support, stability, numerical and timeout thresholds |
| `configs/instructions/multi_object.yaml` | Seen and held-out paraphrase wrappers |
| `configs/appearance.yaml` | Table palette and one shared candidate color per scene |
| `tasks/multi_object/task.py` | Configuration, frozen-scene loading and collector adapter |
| `tasks/multi_object/scenarios.py` | Layout, instructions, appearance, geometry and reach validation |
| `tasks/<subgroup>/planning.py` | Each subgroup's goal fields, ordering, bin allocation, subset enumeration and wording |
| `tasks/multi_object/planning.py` | Shared measurement checks, dispatch to the subgroup, and achieved-state goals |
| `tasks/multi_object/manifest.py` | Balanced problem variants, numeric queries and paraphrase sweeps |
| `controllers/multi_object.py` | Sequential physical grasps, transport, release and feedback replanning |
| `tasks/multi_object/evaluator.py` | Independent physical event history and numerical success |
| `tasks/multi_object/validation.py` | Replay of the recorded 120 Hz state evidence |
| `tasks/multi_object/diagnostics.py` | Explicit diagnostic-only intermediate-reference perturbation |
| `evaluate_multi_object.py` | Integrity validation and per-subgroup and per-case outcome summary |

## Task contracts

The examples below use the first configured numerical problem. Each YAML also
includes a second geometry/value variant. Full instructions name release,
stability and the table frame. All candidate objects have the same scene color;
basket colors vary independently and destinations are identified spatially.

| Case | Numerical instruction | Required result |
|---|---|---|
| SAP-1A | Heights 6, 4, 8 cm, in that order | Recorded pick and stable-release order B, A, C in one basket |
| SAP-1B | Ascending height | All candidates placed in sorted measurement order; the prompt does not reveal the sorted values |
| SAP-1C | Heights 8, 4, 6 cm, in that order | Exactly those objects placed in order; the 10 cm distractor remains untouched |
| SAP-2A | Height below 8 cm versus at least 8 cm | Every object released in the correct left/right basket |
| SAP-2B | Height below 6, 6 to below 10, or at least 10 cm | Three mutually exclusive basket assignments |
| SAP-2C | Length >8 cm AND width <4.5 cm | Conjunction matches in the left basket, the other three truth combinations in the right |
| CP-1A | Build a 9 cm tower | Supported, upright stack with measured total height within tolerance |
| CP-1B | Build a 20 cm row | Supported straight row with adjacent end faces and correct measured length |
| CP-1C | Build an 18 cm row | Either 6+12 or 8+10 cm is accepted; unused blocks remain untouched |
| CP-2A | Two 9 cm towers | Disjoint components, both anchored heights correct |
| CP-2B | Left tower 9 cm, right tower 4 cm taller | Measured heights 9 and 13 cm; equality of the difference alone is insufficient |
| CP-2C | Left row 14 cm, right row 22 cm | Both structures satisfy their own target using disjoint components |
| FDP-1A | Collect combined length 20 cm, then stop | Exact selected sum, physical basket placements and an explicit stop |
| FDP-1B | Collect combined solid volume 200 cm³, then stop | Exact selected volume sum, placements and stop; volume is not capacity |
| FDP-1C | Add blocks until a tower reaches 9 cm, then stop | Remaining allocation recomputed from achieved height after each placement |
| FDP-2A | A 12 cm right of the reference, B 10 cm right of achieved A | Two released center-offset relations grounded in measured poses |
| FDP-2B | Successive 6 cm surface gaps | Current predecessor pose and complete object extents determine the next goal |
| FDP-2C | Three linked offsets | Every predecessor is released before the next pick; each target uses the achieved predecessor |

Constructions and cumulative collections require at least two components. The
generator enumerates valid disjoint combinations before laying out the scene.
The evaluator accepts every enumerated valid combination, not just the oracle's
chosen subset. Whole-object basket containment is independent of the oracle's
internal placement points. Wrong order cannot be repaired merely by producing
the correct final basket contents.

## Physical evidence and tolerances

The evaluator records grasp/lift proxies and stable placement events at 120 Hz.
A placement requires an observed lift of at least 2.5 cm, an open gripper,
17 cm geometric separation from the hand origin, upright orientation within
10 degrees, support within 3 mm, and speeds below 0.02 m/s and 0.2 rad/s for
0.25 seconds. Basket scoring requires the entire conservative footprint inside
the declared cavity with a 3 mm wall margin. Objects may protrude above the walls.

The final complete instruction must remain valid for 0.5 seconds after the
controller explicitly requests `stop`. Unused candidates must never move more
than 5 mm or be lifted. Rows use 5 mm adjacency/total-length tolerance and 6 mm
lateral alignment tolerance. Towers use actual support surfaces and 5 mm total
height tolerance. Feedback position relations use 15 mm tolerance. These are
execution tolerances, separate from exact numerical subset selection.

The native recorder stores all object poses, velocities, AABBs, robot states,
actions, calibrated RGB-D streams, and a physics-rate `control_trace`. Metadata
contains placement order/timestamps, basket assignments, measured structures,
numerical errors and stopping status. Video is one MP4 per camera per attempt.
Grasp and support checks are geometric proxies: contact impulses and two-finger
contact forces are not measured. No success label alone establishes visual
metric observability or VLA reasoning ability.

## Configure and run

From `/home/locnp/IsaacLab`, use the existing Isaac Lab uv environment:

```bash
uv run python PhysQuant/collect_data.py --config PhysQuant/configs/tasks/sequencing_allocation_planning/SAP-1A.yaml --check_config

# One scene for every SAP family, with both seen instruction templates: 12 episodes.
uv run python PhysQuant/generate_manifest.py --config PhysQuant/configs/tasks/sequencing_allocation_planning --output PhysQuant/manifests/my_run/sequencing_allocation_planning --scene_count 1

# Both numerical/geometry variants of every SAP family, one template each: 12 episodes.
uv run python PhysQuant/generate_manifest.py --config PhysQuant/configs/tasks/sequencing_allocation_planning --output PhysQuant/manifests/my_variants/sequencing_allocation_planning --scene_count 2 --one_template

uv run --no-sync python PhysQuant/collect_manifest.py \
  PhysQuant/manifests/my_variants/sequencing_allocation_planning/plan.yaml \
  --output /media/Seagate16T/locnp/PhysQuant/my_variants/sequencing_allocation_planning \
  --gpus 0 1 2 3 --cpus_per_worker 4 --worker_episodes 32 \
  --capture_profile full --camera_hz 20 --compression lzf --no-video \
  --attempts_per_episode 2 --attempt_timeout_s 1200

uv run python PhysQuant/evaluate_multi_object.py /media/Seagate16T/locnp/PhysQuant/my_variants/sequencing_allocation_planning --output PhysQuant/reports/my_multi_object_results.json
```

Pass `--config PhysQuant/configs/tasks` to plan every subgroup at
once; the generator writes one plan per subgroup under `--output/<subgroup>/`.
Use fresh plan/output directories. `--attempt_timeout_s` is wall time; the task's
120 second timeout is simulated time. The 1,200 second command above allows for
multi-pick episodes and renderer startup without changing the task's timeout.
Choose available GPUs on the machine. Templates sharing geometry reuse a worker
when compatible; every episode resets robot, object, controller and evaluator state.

Prepare the output shader cache as shown in the [collection guide](../../README.md#collect-5000-understanding-episodes).
For larger plans, pilot `--num_envs 2` with one worker per GPU. Only clone-compatible
episodes run together; different basket colors can split sorting scenes into
separate groups. The [performance review](../../reports/long_horizon_performance_20261103.md)
records measured throughput, validated cases, and parallel-mode restrictions.

Edit `problems` in a case YAML to supply additional object geometries and goals.
Manifest generation cycles through problems before repeating one. `problem_index`
can pin a particular variant. Sizes are in meters; numerical goals are centimeters
or cubic centimeters as appropriate. Unknown/incompatible measurements, ambiguous
orders, overlapping placements, oversized grasps and infeasible allocations fail
preflight. There is no silent resizing or tolerance relaxation.

For matched numerical queries over the same scene, set `goal_sweep`, for example
`[{measurement: length, target: 20}, {measurement: length, target: 18}]` for the
first FDP-1A problem. Each query is a separate reset/episode with identical poses
and appearance. Select `--template_split held_out --split eval` for unseen wording.

Initial positions, jitter, yaw, basket palette, construction anchors and reference
position are exposed by `MultiObjectConfig` and can be overridden in each task
YAML. Shared lighting/camera randomization can be supplied by
`randomization_profile`; independent candidate pose randomization is rejected
because it would invalidate frozen task layouts.

The supplied presets contain three or four visible boxes. The schema accepts
three through seven candidates, but larger horizons need explicit layouts and
enough basket capacity. The active tray has patterns for one, two or three
selected objects per basket; configurations exceeding that capacity are rejected.
Additional shapes, counts and values require physical qualification before scale-up.

For a controlled feedback diagnostic, add
`diagnostic_intervention: {offset_xy_m: [0.025, 0.0]}` to an FDP-2 configuration
with `split: diagnostic`. After A is released, the runtime explicitly shifts A
by that amount before planning B. The shift is recorded and replay-validated;
A's anchor is adjusted by the known intervention while later relations continue
to use its actual pose. This is a labelled intervention experiment, not an ordinary
demonstration. Training/evaluation splits reject this option.
