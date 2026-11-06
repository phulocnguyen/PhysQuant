# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a policy in closed loop on Spatial Bench suites.

The policy runs in its own process and environment behind a websocket
(see ``evaluation/policy_server.py``). This runner owns Isaac Sim: it replays each
frozen episode, sends observations, executes the returned joint targets, and scores
the result with the protocol's physics evaluator. Results are written per episode,
so an interrupted run resumes with ``--resume``.

Examples:

    # Validate the environment with the scripted controller (no policy server).
    uv run --no-sync python PhysQuant/evaluate_policy.py --policy oracle --output results/oracle

    # Evaluate a served model on two suites, on four GPUs.
    uv run --no-sync python PhysQuant/evaluate_policy.py --policy ws://localhost:8000 \\
        --suites object_understanding spatial_transformation --gpus 0 1 2 3 --output results/my_model
"""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

from evaluation.policies.base import validate_action  # noqa: E402
from evaluation.suites import SUITES, TIERS, action_dim, available, load_episodes  # noqa: E402


def selected_episodes(args):
    """(suite, tier, index, entry) for every requested episode, in a fixed global order."""
    pairs = [(s, t) for s, t in available(args.bundle) if s in args.suites and t in args.tiers]
    if not pairs:
        raise SystemExit(f"No requested suite tiers found under {args.bundle}; run make_suites.py first")
    episodes = []
    for suite, tier in pairs:
        entries = load_episodes(args.bundle, suite, tier)
        if args.max_episodes is not None:
            entries = entries[: args.max_episodes]
        episodes.extend((suite, tier, index, entry) for index, entry in enumerate(entries))
    return episodes


def result_path(output, suite, tier, entry):
    return Path(output) / "episodes" / suite / tier / f"{entry['episode_index']:06d}.json"


class PolicyClient:
    """Websocket client for a model served with ``evaluation.policy_server.serve``."""

    def __init__(self, url):
        from websockets.sync.client import connect

        from evaluation.protocol import decode, encode

        self.encode, self.decode = encode, decode
        self.connection = connect(url, max_size=None, open_timeout=60)

    def _request(self, message):
        self.connection.send(self.encode(message))
        reply = self.decode(self.connection.recv())
        if "error" in reply:
            raise RuntimeError(f"Policy server error: {reply['error']}")
        return reply

    def reset(self, episode):
        self._request({"type": "reset", "episode": episode})

    def act(self, observation):
        return self._request({"type": "act", "observation": observation})["action"]


def run_worker(args):
    """Evaluate this worker's share of episodes inside one Isaac Sim process."""
    import numpy as np
    from tasks.common.scenarios import compatibility_key

    from evaluation.env_adapter import launch

    episodes = [e for i, e in enumerate(selected_episodes(args)) if i % args.num_workers == args.worker_index]
    episodes = [e for e in episodes if not (args.resume and result_path(args.output, e[0], e[1], e[3]).exists())]
    # Keep identical scenes adjacent so the stage rebuilds rarely.
    episodes.sort(key=lambda e: (compatibility_key(e[3]["scenario"]), e[0], e[1], e[2]))
    client = None if args.policy in ("oracle", "hold") else PolicyClient(args.policy)
    with launch(args, fps=args.fps) as env:
        for suite, tier, _, entry in episodes:
            scenario = entry["scenario"]
            started = time.monotonic()
            record = dict(
                suite=suite,
                tier=tier,
                episode_index=entry["episode_index"],
                episode_id=entry["episode_id"],
                case=scenario["case"],
                instruction=scenario["instruction"],
                policy=args.policy,
            )
            try:
                observation = env.reset(scenario)
                episode = dict(
                    suite=suite,
                    tier=tier,
                    case=scenario["case"],
                    instruction=scenario["instruction"],
                    fps=args.fps,
                    action_dim=action_dim(suite),
                    max_steps=-(-env.max_ticks // env.ticks_per_step),
                )
                if client is not None:
                    client.reset(episode)
                done, info, steps, queue = False, {}, 0, []
                while not done:
                    if args.policy == "oracle":
                        observation, done, info = env.step(oracle=True)
                    else:
                        if not queue:
                            if args.policy == "hold":
                                action = np.zeros(action_dim(suite), np.float32)
                                action[:9] = observation["observation.state"]
                            else:
                                action = client.act(observation)
                            queue = list(np.atleast_2d(validate_action(action, action_dim(suite))))
                        observation, done, info = env.step(queue.pop(0))
                    steps += 1
                record.update(status="completed", steps=steps, **info)
            except (ValueError, RuntimeError) as error:
                # Environment or policy-protocol errors are reported, never scored as success.
                record.update(status="error", success=False, error=f"{type(error).__name__}: {error}")
            record["wall_seconds"] = time.monotonic() - started
            path = result_path(args.output, suite, tier, entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=1, default=float))
            tmp.replace(path)
            print(
                f"[{suite}/{tier} #{entry['episode_index']}] success={record.get('success')} {record['status']}",
                flush=True,
            )


def launch_workers(args, argv):
    """One worker process per GPU; each owns a disjoint share of the episodes."""
    output = Path(args.output)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    with ExitStack() as logs:
        workers = []
        for index, gpu in enumerate(args.gpus):
            command = [sys.executable, str(Path(__file__).resolve()), *argv, "--worker_index", str(index)]
            command += ["--num_workers", str(len(args.gpus))]
            # The renderer GPU is selected through Kit, as in the collection coordinator.
            kit_args = " ".join(
                (
                    args.kit_args or "",
                    f"--/renderer/multiGpu/activeCudaGpus={gpu},",
                    "--/app/settings/loadUserConfig=false",
                    "--/app/settings/persistent=false",
                )
            )
            command += ["--kit_args", kit_args]
            log = logs.enter_context(open(output / "logs" / f"worker_{index}_gpu{gpu}.log", "w"))
            workers.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ)))
            print(f"Worker {index} on GPU {gpu}: {log.name}", flush=True)
        codes = [process.wait() for process in workers]
    from evaluation.metrics import write_report

    write_report(output)
    if any(codes):
        raise SystemExit(f"Worker exit codes: {codes}")


def main():
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True, help="ws://HOST:PORT of a policy server, or 'oracle' / 'hold'")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifests",
        "--bundle",
        dest="bundle",
        type=Path,
        default=ROOT / "manifests/benchmark",
        help="Directory containing <suite>/<tier>/plan.yaml",
    )
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    parser.add_argument("--tiers", nargs="+", default=list(TIERS), choices=list(TIERS))
    parser.add_argument("--max_episodes", type=int, help="Per suite tier, for a quick check")
    parser.add_argument("--fps", type=int, default=10, help="Policy rate; must divide the 30 Hz control rate")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker_index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num_workers", type=int, default=1, help=argparse.SUPPRESS)
    add_launcher_args(parser)
    args = parser.parse_args()
    args.visualizer = []
    if args.worker_index is None:
        argv = [a for a in sys.argv[1:]]
        launch_workers(args, argv)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
