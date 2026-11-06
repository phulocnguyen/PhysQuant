# PhysQuant

Numerical grounding tasks with a Franka Panda: demonstration collection, LeRobot
export, and closed-loop policy evaluation on the same simulator and task scorers.

This project was reorganized from `Spatial-Understanding`. That project and its
active collection/export jobs are unchanged. Run the commands below from the
parent Isaac Lab checkout, using its uv environment.

## Organization

| Directory | Responsibility |
|---|---|
| `configs/` | Shared robot, physics, camera, appearance, task, and success settings |
| `assets/` | Geometry/spawning code and local asset references |
| `core/` | Units, timing, renderer profiles, and CPU resource controls |
| `tasks/` | Nine task subgroups and three shared engines; authoritative success scoring |
| `controllers/` | Scripted demonstration controllers |
| `environments/` | Shared scene construction, physical reset, randomization, and cameras |
| `data_collection/` | Collection scheduling, recording, integrity checks, and export |
| `evaluation/` | Policy environment adapter, rollouts, transport, metrics, failure analysis |
| `evaluation/policies/` | Hold/oracle/random baselines and model adapters |
| `evaluation/offline/` | Recorded-attempt review and prediction scoring |
| `manifests/` | Training, validation, and frozen benchmark episode plans |
| `datasets/` | Links to existing recordings/exports and new rollout outputs |
| `tools/` | Manifest generation, asset preparation, and portable bundles |
| `tests/` | Offline collection, scoring, export, and evaluation contracts |

See [TASKS.md](TASKS.md) for the task definitions. Collection and evaluation reuse
`environments/simulation.py` and the same task evaluators. Changing package
locations does not change scenario identities, joint ordering, or success thresholds.

## Collect and export

The root collection/export scripts remain supported. New collection defaults use
`datasets/native_hdf5/`; explicit output paths still work.

```bash
uv run --no-sync python PhysQuant/collect_data.py --list_tasks
uv run --no-sync python PhysQuant/collect_data.py \
  --config PhysQuant/configs/tasks/object_understanding/length.yaml --check_config

# The output directory must not already exist.
uv run --no-sync python PhysQuant/export_lerobot_minimal.py \
  /path/to/native_collection --output /path/to/new_lerobot_dataset --fps 10
```

`export_lerobot_minimal.py` preserves the top/wrist RGB, joint state, joint target,
and language export contract. `export_dataset.py` retains the existing accepted-
demonstration export with calibration support. Existing recorded paths and metadata
are not rewritten.

## Closed-loop evaluation

The copied benchmark plans live at `manifests/benchmark/<suite>/<tier>/plan.yaml`.
Object Understanding T1, T2, and T3 each contain 140 episodes; T5 has 20 feasible
surface-area episodes. `--max_episodes` is a limit **per suite and tier**, not an
episode generator. Use the same immutable plan for every checkpoint comparison.

```bash
# A short scripted-controller smoke run; use a fresh output directory.
uv run --no-sync python PhysQuant/evaluate_policy.py \
  --policy oracle --suites object_understanding --tiers t1_in_distribution \
  --max_episodes 2 --fps 10 --gpus 0 --output PhysQuant/datasets/rollouts/oracle_smoke

# A served model speaking the PhysQuant protocol.
uv run --no-sync python PhysQuant/evaluate_policy.py \
  --policy ws://127.0.0.1:8000 --suites object_understanding \
  --tiers t1_in_distribution --fps 10 --gpus 0 \
  --output PhysQuant/datasets/rollouts/my_model --resume

uv run --no-sync python PhysQuant/evaluation/failure_analysis.py \
  PhysQuant/datasets/rollouts/my_model
```

`summary.json` and `summary.md` report success, errors, and confidence intervals;
`episodes/` holds measured per-episode outcomes and `logs/` holds worker logs.
Success rate uses completed episodes as its denominator. Check the error count
before interpreting it as a score over the full plan. Resume skips existing episode
records, including recorded errors; use a fresh output for a new checkpoint/run.

To freeze exactly 200 Object Understanding episodes under a fresh suite directory:

```bash
uv run --no-sync python PhysQuant/tools/generate_manifest.py \
  --config PhysQuant/configs/tasks/object_understanding \
  --output PhysQuant/manifests/benchmark_200/object_understanding/t1_in_distribution \
  --episodes 200 --random_candidate_counts 3 4 5 6 7 --seed 9500000 \
  --split eval --template_split seen --render_profile balanced
```

Pass `--manifests PhysQuant/manifests/benchmark_200` to the evaluator for this plan.
For validation, choose a separate seed range and output directory under
`manifests/validation/`. Validation and benchmark both use the scenario's existing
`eval` split; directory names do not create new schema values. Do not tune models
against the final benchmark episodes.

## Policy interface

The model sees `observation.images.top` and `observation.images.wrist` (uint8 HWC
RGB, 256 x 256), `observation.state` (float32, 7 arm + 2 finger joint positions),
and `task` (instruction). Actions are float32 absolute joint targets in that same
order: arm radians and finger metres. The three planning tasks append a stop value,
so their actions have dimension 10 instead of 9. The default policy rate is 10 Hz
of simulation time. A `(k, action_dim)` chunk executes before the next observation
is sent to the model; return one action for feedback at every policy step.

The adapters accept already-loaded models and explicit checkpoint-specific
`observation_transform`, `action_transform(prediction, observation)`, and optional
`reset_model(episode)` functions. They do not guess image keys, batching, normalized
actions, delta targets, or gripper conventions. No model weights are bundled.

- `Pi05Policy`: wraps OpenPI `infer(...)["actions"]`.
- `Gr00tPolicy`: wraps GR00T `get_action(...)` and its action dictionary.
- `LLMPolicy`: wraps a caller-supplied inference function returning an `actions`
  mapping or JSON. A high-level planner also needs a caller-supplied controller.
- `HoldPolicy` and `RandomPolicy`: transport/setup baselines.
- `policies/oracle.py`: privileged scripted reference, only used inside simulation.

Create a module in your model environment with a `build_policy()` function that
loads the checkpoint and returns one of these adapters. Then, from that environment:

```bash
python /path/to/PhysQuant/evaluation/policy_server.py \
  --factory my_model_adapter:build_policy --host 127.0.0.1 --port 8000
```

The factory creates a session per connection. Share only immutable weights when
serving multiple workers; each session needs independent history and action queues.
The programmatic `serve(policy_instance)` form admits one connection at a time.
OpenPI's stock WebSocket server and GR00T's native server use different protocols;
their clients can be wrapped by these adapters, but are not direct evaluator URLs.

For a transport-only smoke test:

```bash
uv run --no-sync python PhysQuant/evaluation/policy_server.py --policy hold
```

## Another machine

Use the matching Isaac Lab source revision and `uv.lock`, Python 3.12, Isaac Sim
6.2.0.0, and an RTX-capable NVIDIA GPU/driver. Prepare the simulator environment with
`uv sync --locked --extra isaacsim`; the transport also needs `websockets` (14.2 on
the source machine). Keep OpenPI/GR00T dependencies in their own environment.

Large asset directories and existing datasets are local symlinks to avoid copying
gigabytes on this nearly-full workstation. [assets/README.md](assets/README.md)
lists their dependencies. Build a portable evaluation bundle on a volume with enough
free space before transferring the project:

```bash
uv run --no-sync python PhysQuant/tools/bundle.py \
  --output /path/on/large/disk/physquant_bundle
```

The tool copies runtime code, frozen benchmark plans, and asset files, dereferences
local asset links, and rewrites background references relative to the bundle. Pass
`--manifests PhysQuant/manifests/benchmark_200` to package the 200-episode suite.
Training datasets and training plans are excluded. The Franka robot still uses the
Isaac Lab asset root: allow the initial online download or configure a local cache.
Run an oracle and hold smoke test on the destination and inspect rendered views.
Retain third-party asset license files when distributing bundles.

The `Dockerfile` is for a **materialized bundle**, layered on a matching
`isaac-lab-base` image. Its launcher uses `uv`; the local linked source tree is not
a portable Docker build context.

## Validation

```bash
uv run --no-sync python -m unittest discover -s PhysQuant/tests -p 'test_*.py'
```

The offline suite does not certify a trained model or a GPU renderer. Run a small
physical rollout before a full benchmark.
