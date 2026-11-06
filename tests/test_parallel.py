# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Concurrent process scheduling with real attempt files, without simulator startup."""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_collection.coordinator import collect, worker_resources
from selection_fixtures import placement_plan
from tasks.common.task import BenchmarkConfig


class ParallelCollectionContracts(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) >= 2, "needs two CPUs")
    def test_parallel_attempts_are_disjoint_bounded_and_resume_without_relaunch(self):
        # Substitute only the simulator executable. The coordinator, recorder,
        # native validator, process concurrency, CPU assignment and journals are real.
        from data_collection import coordinator

        with TemporaryDirectory() as folder:
            root = Path(folder)
            prepared_cache = root / "collection" / "kit" / "cache" / "prepared-shader.bin"
            prepared_cache.parent.mkdir(parents=True)
            prepared_cache.write_bytes(b"keep the prepared shader cache")
            fixture = root / "worker.py"
            fixture.write_text(
                f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n"
                + """
import os, threading
available = os.sched_getaffinity(0)
from core.resources import configure_worker
configure_worker()
import argparse, json, os, tempfile, time
from pathlib import Path
from types import SimpleNamespace
import yaml
from data_collection.finalize import finalize_attempt
from data_collection.metadata import episode_info, write_json_atomic
from data_collection.recorder import EpisodeRecorder
# Simulate a native pool widening its own mask during Kit startup.
ready, finish = threading.Event(), threading.Event()
def native_thread():
    os.sched_setaffinity(0, available)
    ready.set()
    finish.wait(5)
thread = threading.Thread(target=native_thread)
thread.start()
ready.wait(5)
try:
    configure_worker()
    assert os.sched_getaffinity(thread.native_id) == os.sched_getaffinity(0)
finally:
    finish.set()
    thread.join()
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path)
args, _ = parser.parse_known_args()
config = yaml.safe_load(args.config.read_text())
worker = args.config.parent
barrier = worker.parent.parent / 'barrier'
barrier.mkdir(exist_ok=True)
started = time.monotonic()
write_json_atomic(barrier / (worker.name + '.json'),
                  dict(start=started, cpus=sorted(os.sched_getaffinity(0)), temporary_directory=tempfile.gettempdir()))
while len(list(barrier.glob('*.json'))) < 2:
    if time.monotonic() - started > 4:
        raise RuntimeError('A second worker never started concurrently')
    time.sleep(.01)
run = Path(config['output_dir']) / 'run'
for index, path in enumerate(config['episode_files']):
    scene = json.loads(Path(path).read_text())
    write_json_atomic(run / 'progress.json', dict(episode=index))
    directory = run / f'episode_{index:04d}'
    metadata = episode_info('object_understanding', scene, scene['seed'], scene['target_index'])
    metadata['attempt_id'] = config['attempt_id']
    recorder = EpisodeRecorder(directory, metadata, directory / 'dataset.hdf5',
                               scene=dict(scenario=scene, objects=scene['objects']))
    finalize_attempt(recorder, False, 'intentional test failure', SimpleNamespace(video=False))
"""
            )
            plan = placement_plan(BenchmarkConfig(), episode_count=4)
            args = SimpleNamespace(
                output=root / "collection",
                start_episode=0,
                max_episodes=None,
                attempts_per_episode=2,
                attempt_timeout_s=15,
                worker_episodes=2,
                mode="persistent",
                resume=False,
                video=False,
                compression="lzf",
                gpu=None,
                gpus=[0, 1],
                cpus_per_worker=1,
            )
            prepare = coordinator._prepare_worker

            def command(*values):
                result = prepare(*values)
                result[1] = str(fixture)
                return result

            with patch.object(coordinator, "_prepare_worker", side_effect=command):
                with self.assertRaises(SystemExit) as stopped:
                    collect(plan, args)
                self.assertEqual(stopped.exception.code, 1)
                summary = json.loads((args.output / "summary.json").read_text())
                self.assertEqual((summary["planned"], summary["attempted"], summary["finalized"]), (4, 8, 8))
                self.assertEqual(summary["exhausted"], 4)
                invocation = json.loads((args.output / "invocations.json").read_text())[-1]
                self.assertEqual(invocation["peak_active_workers"], 2)
                resources = invocation["resources"]
                self.assertTrue(set(resources[0]["cpu_ids"]).isdisjoint(resources[1]["cpu_ids"]))
                for worker in (args.output / "workers").iterdir():
                    resource = json.loads((worker / "process.json").read_text())
                    actual = json.loads((args.output / "barrier" / (worker.name + ".json")).read_text())
                    self.assertEqual(actual["cpus"], resource["cpu_ids"])
                    self.assertTrue(Path(actual["temporary_directory"]).is_relative_to(args.output.resolve()))
                histories = [json.loads(p.read_text()) for p in args.output.glob("episode_*/attempts.json")]
                self.assertEqual(len({a["completion_file"] for h in histories for a in h}), 8)
                files = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.rglob("dataset.hdf5")}
                args.resume = True
                with self.assertRaises(SystemExit):
                    collect(plan, args)
                resumed = json.loads((args.output / "invocations.json").read_text())[-1]
                self.assertEqual(resumed["simulator_launch_count"], 0)
                self.assertEqual(files, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
                self.assertEqual(prepared_cache.read_bytes(), b"keep the prepared shader cache")

    def test_invalid_resource_assignments_fail_before_starting_workers(self):
        args = SimpleNamespace(gpu=None, gpus=[0, 0], cpus_per_worker=1)
        with self.assertRaisesRegex(ValueError, "distinct"):
            worker_resources(args)
        args.gpus = [0, 1]
        args.cpus_per_worker = len(os.sched_getaffinity(0))
        with self.assertRaisesRegex(ValueError, "available"):
            worker_resources(args)


if __name__ == "__main__":
    unittest.main()
