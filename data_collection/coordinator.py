# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded persistent workers, durable attempt journals, and verified resume."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

import h5py
import yaml
from evaluation.offline.review import accepted
from tasks.common.scenarios import compatibility_key, digest, launch_key
from tasks.registry import get_task, scenario_subgroup

from data_collection.metadata import episode_dataset, read_metadata, read_scene, write_json_atomic
from data_collection.validate import validate_dataset


def verify_completion(marker, scenario):
    """A marker is trusted only with its finalized, matching native recording."""
    marker = Path(marker)
    result = json.loads(marker.read_text())
    if result["fingerprint"] != scenario["fingerprint"] or result["episode_id"] != scenario["episode_id"]:
        raise ValueError("Completion marker belongs to a different planned episode")
    dataset = episode_dataset(marker.parent)
    validate_dataset(dataset, marker.parent.name)
    with h5py.File(dataset) as data:
        if data.attrs["schema_version"] != 4:
            raise ValueError("Resume requires the current native recording schema")
        episode = data[f"data/{marker.parent.name}"]
        meta = read_metadata(episode)
        if (
            meta["success"] != result["success"]
            or meta["episode_id"] != scenario["episode_id"]
            or read_scene(episode)["scenario"]["fingerprint"] != scenario["fingerprint"]
        ):
            raise ValueError("Completion marker disagrees with its native recording")
    return meta


def _progress(worker):
    paths = list((worker / "data").glob("*/progress.json"))
    return json.loads(paths[0].read_text())["episode"] if paths else None


def _record_results(worker, entries, histories, directories, *, final):
    """Reconcile atomic completions, including a coordinator crash before journaling."""
    assigned = json.loads((worker / "assigned.json").read_text())
    outcome_path = worker / "result.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    completed = {}
    for marker in (worker / "data").glob("*/episode_*/complete.json"):
        result = json.loads(marker.read_text())
        completed[result["episode_id"]] = (marker, result)
    active = _progress(worker)
    for index, key in enumerate(assigned["episode_ids"]):
        attempt_id = assigned["attempt_id"]
        if len(histories[key]) > attempt_id:
            continue
        if len(histories[key]) != attempt_id:
            raise ValueError("Worker journal has a missing preceding attempt")
        marker, result = completed.get(key, (None, {}))
        if not result and (not final or index > (0 if active is None else active)):
            continue  # Unstarted work keeps its retry budget.
        if result:
            verify_completion(marker, entries[key]["scenario"])
        success = bool(result.get("success"))
        failure = (
            None
            if success
            else "execution_failed"
            if result
            else "wall_timeout"
            if outcome.get("timed_out")
            else "incomplete_worker"
        )
        histories[key].append(
            dict(
                attempt_id=attempt_id,
                worker=worker.name,
                returncode=outcome.get("returncode"),
                timed_out=bool(outcome.get("timed_out") and not result),
                successful_rollout=success,
                finalized=bool(result),
                completion_file=str(marker) if marker else None,
                failure_reason=failure,
            )
        )
        write_json_atomic(directories[key] / "attempts.json", histories[key])


def _terminate(process):
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_worker(worker, command, timeout, publish, stop, resources):
    start = deadline_start = time.monotonic()
    active, timed_out = None, False
    environment = os.environ.copy()
    # Keep large simulator scratch files on the collection drive, shared across
    # workers so cached assets need not be downloaded for every simulator launch.
    temporary_directory = worker.parent.parent / "tmp"
    temporary_directory.mkdir(exist_ok=True)
    environment["TMPDIR"] = str(temporary_directory)
    if resources["cpu_ids"]:
        environment["BENCHMARK_CPU_IDS"] = ",".join(map(str, resources["cpu_ids"]))
    with (worker / "simulation.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=worker.parent.parent
        )
        try:
            write_json_atomic(worker / "process.json", {"pid": process.pid, "command": command, **resources})
            while process.poll() is None:
                if stop.is_set():
                    _terminate(process)
                    break
                current = _progress(worker)
                if current is not None and current != active:
                    if active is not None:
                        deadline_start = time.monotonic()
                    active = current
                    publish()
                if time.monotonic() - deadline_start > timeout:
                    timed_out = True
                    _terminate(process)
                    break
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=1)
        except BaseException:
            _terminate(process)
            raise
        finally:
            write_json_atomic(
                worker / "result.json",
                dict(
                    wall_seconds=time.monotonic() - start,
                    returncode=process.returncode,
                    timed_out=timed_out,
                    **resources,
                ),
            )


def _prepare_worker(worker, batch, attempt_id, directories, args, resources):
    first = batch[0]
    config = get_task(scenario_subgroup(first["scenario"])).collection_settings(first["scenario"])
    config.update(
        scenario_file=str(directories[first["episode_id"]] / "scenario.json"),
        episode_files=[str(directories[e["episode_id"]] / "scenario.json") for e in batch],
        num_episodes=len(batch),
        max_attempts=len(batch),
        attempt_id=attempt_id,
        output_dir=str(worker / "data"),
        video=bool(args.video),
        compression=args.compression,
        capture_profile=getattr(args, "capture_profile", "full"),
        camera_hz=getattr(args, "camera_hz", None),
        num_envs=getattr(args, "num_envs", 1),
        isolated_episodes=True,
    )
    config_path = worker / "collection.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_json_atomic(
        worker / "assigned.json",
        dict(
            episode_ids=[e["episode_id"] for e in batch],
            attempt_id=attempt_id,
        ),
    )
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "collect_data.py"),
        "--config",
        str(config_path),
        "--visualizer",
        "none",
        "--device",
        "cpu",
    ]
    # Collection settings come from the frozen plan, not interactive Kit sessions.
    # Concurrent workers must not read or overwrite a shared user.config.json.
    # Relative paths are safe even when the collection directory contains spaces.
    kit_args = [
        "--portable-root",
        "kit",
        f"--/log/file=workers/{worker.name}/kit.log",
        "--/app/settings/loadUserConfig=false",
        "--/app/settings/persistent=false",
    ]
    if resources["gpu"] is not None:
        kit_args.append(f"--/renderer/multiGpu/activeCudaGpus={resources['gpu']},")
    if resources["cpu_ids"]:
        count = len(resources["cpu_ids"])
        kit_args.extend(
            [
                f"--/plugins/carb.tasking.plugin/threadCount={count}",
                f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={count}",
            ]
        )
    command.extend(["--kit_args", " ".join(kit_args)])
    return command


def worker_resources(args):
    """Assign ``workers_per_gpu`` processes per GPU and nonoverlapping CPU sets when requested.

    Several workers share one GPU when a single 256x256 scene leaves it underused.
    """
    gpus = args.gpus if args.gpus is not None else [args.gpu]
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu is not None and gpu < 0 for gpu in gpus):
        raise ValueError("Worker GPUs must be distinct nonnegative indices")
    per_gpu = getattr(args, "workers_per_gpu", 1)
    if type(per_gpu) is not int or per_gpu < 1:
        raise ValueError("workers_per_gpu must be a positive integer")
    gpus = [gpu for gpu in gpus for _ in range(per_gpu)]
    budget = args.cpus_per_worker
    if budget is None and len(gpus) == 1:
        return [{"gpu": gpus[0], "cpu_ids": []}]
    available = sorted(os.sched_getaffinity(0))
    budget = min(8, len(available) // len(gpus)) if budget is None else budget
    if budget < 1 or budget * len(gpus) > len(available):
        raise ValueError(f"{len(gpus)} workers need {budget} CPUs each; only {len(available)} are available")
    return [{"gpu": gpu, "cpu_ids": available[i * budget : (i + 1) * budget]} for i, gpu in enumerate(gpus)]


def _summary(entries, histories, launch_count, attempts_per_episode):
    counts = Counter(
        dict(
            planned=len(entries),
            attempted=0,
            finalized=0,
            successful=0,
            first_attempt_successes=0,
            accepted=0,
            rejected=0,
            pending_review=0,
        )
    )
    counts.update(exhausted=0, pending=0)
    for key in entries:
        history = histories[key]
        counts["attempted"] += len(history)
        counts["finalized"] += sum(a["finalized"] for a in history)
        counts["first_attempt_successes"] += int(bool(history) and history[0]["successful_rollout"])
        successful = next((a for a in history if a["successful_rollout"]), None)
        if successful:
            counts["successful"] += 1
            marker = Path(successful["completion_file"])
            meta = json.loads((marker.parent / "metadata.json").read_text())
            status = (
                "accepted"
                if accepted(meta)
                else "rejected"
                if "failed" in meta.get("validation_gates", {}).values()
                else "pending_review"
            )
            counts[status] += 1
        else:
            limit = 1 if entries[key]["scenario"]["split"] == "eval" else attempts_per_episode
            counts["exhausted" if len(history) >= limit else "pending"] += 1
    counts["simulator_launch_count"] = launch_count
    return dict(counts)


def _recover_orphan(worker):
    """Stop only a journaled worker whose live command exactly matches this output."""
    path = worker / "process.json"
    if (worker / "result.json").exists() or not path.exists():
        return
    recorded = json.loads(path.read_text())
    command_path = Path(f"/proc/{recorded['pid']}/cmdline")

    def still_owned():
        try:
            command = command_path.read_bytes().split(b"\0")[:-1]
        except FileNotFoundError:
            return False
        return command == [os.fsencode(part) for part in recorded["command"]]

    if still_owned():
        with suppress(ProcessLookupError):
            os.kill(recorded["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 15
        while still_owned() and time.monotonic() < deadline:
            time.sleep(0.05)
        if still_owned():
            with suppress(ProcessLookupError):
                os.kill(recorded["pid"], signal.SIGKILL)
            deadline = time.monotonic() + 15
            while still_owned() and time.monotonic() < deadline:
                time.sleep(0.05)
            if still_owned():
                raise RuntimeError(f"Could not stop the previous worker: {worker}")


def collect(manifest, args):
    """Allow one coordinator to own an output directory, including during resume."""
    import fcntl

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another collection coordinator already owns this output directory") from error
        _collect_locked(manifest, args)


def _collect_locked(manifest, args):
    resources = worker_resources(args)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    selected = [e for e in manifest["episodes"] if e["episode_index"] >= args.start_episode][: args.max_episodes]
    entries = {e["episode_id"]: e for e in selected}
    if not entries:
        raise ValueError("The requested episode range is empty")
    output = args.output.resolve()
    # Default full capture keeps the identity of collections started before these options.
    capture = {"capture_profile": getattr(args, "capture_profile", "full")}
    if getattr(args, "camera_hz", None) is not None:
        capture["camera_hz"] = args.camera_hz
    identity = digest({"episodes": [e["scenario"]["fingerprint"] for e in selected], **capture, "schema": 3})
    identity_path = output / "identity.json"
    # A prepopulated renderer cache contains no episodes or collection identity.
    if output.exists() and any(path.name != "kit" or not path.is_dir() for path in output.iterdir()):
        if not args.resume:
            raise FileExistsError("Use --resume or a fresh output directory")
        if not identity_path.exists() or json.loads(identity_path.read_text())["identity"] != identity:
            raise ValueError("Resume scene/instruction/protocol/settings/schema identity differs")
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(identity_path, {"identity": identity})
    histories, directories = {}, {}
    for key, entry in entries.items():
        folder = output / f"episode_{entry['episode_index']:06d}"
        directories[key] = folder
        folder.mkdir(exist_ok=True)
        write_json_atomic(folder / "scenario.json", entry["scenario"])
        path = folder / "attempts.json"
        histories[key] = json.loads(path.read_text()) if path.exists() else []
        for attempt in histories[key]:
            if attempt["successful_rollout"]:
                verify_completion(attempt["completion_file"], entry["scenario"])
    worker_root = output / "workers"
    worker_root.mkdir(exist_ok=True)
    for worker in sorted(worker_root.glob("worker_*")):
        if (worker / "assigned.json").exists():
            _recover_orphan(worker)
            _record_results(worker, entries, histories, directories, final=True)
    # Workers share one launch per launch key; scenes are ordered so geometry changes rarely.
    keys = {key: launch_key(entry["scenario"]) for key, entry in entries.items()}
    if getattr(args, "num_envs", 1) > 1:
        from data_collection.parallel import clone_key

        # Cloned environments only need equal object sets, so order by those.
        scene_keys = {key: clone_key(entry["scenario"]) for key, entry in entries.items()}
    else:
        scene_keys = {key: compatibility_key(entry["scenario"]) for key, entry in entries.items()}
    worker_index = max((int(p.name.split("_")[-1]) + 1 for p in worker_root.glob("worker_*")), default=0)
    initial_launches = worker_index
    initial_successes = _summary(entries, histories, worker_index, args.attempts_per_episode)["successful"]
    invocation_path = output / "invocations.json"
    invocations = json.loads(invocation_path.read_text()) if invocation_path.exists() else []
    if invocations and invocations[-1].get("status") == "running":
        invocations[-1]["status"] = "interrupted"
    invocations.append(dict(started_at_utc=started_at, resources=resources, status="running"))
    active = {}
    stop = threading.Event()
    journal_lock = threading.Lock()
    peak_workers = 0

    def reconcile(worker, *, final):
        # Workers own disjoint episodes, but journals and summary reads share this lock.
        with journal_lock:
            _record_results(worker, entries, histories, directories, final=final)

    def publish_summary(*, final=False):
        with journal_lock:
            result = _summary(entries, histories, worker_index, args.attempts_per_episode)
        elapsed = time.monotonic() - started
        invocations[-1].update(
            wall_seconds=elapsed,
            simulator_launch_count=worker_index - initial_launches,
            new_successes=result["successful"] - initial_successes,
            peak_active_workers=peak_workers,
            status=("complete" if not result["pending"] and not result["exhausted"] else "incomplete")
            if final
            else "running",
        )
        write_json_atomic(invocation_path, invocations)
        result.update(active_workers=len(active), max_parallel_workers=len(resources), invocation_wall_seconds=elapsed)
        write_json_atomic(output / "summary.json", result)
        return result

    with ThreadPoolExecutor(max_workers=len(resources)) as executor:
        try:
            while True:
                scheduled = False
                reserved = {key for job in active.values() for key in job["episode_ids"]}
                busy = {job["resource_index"] for job in active.values()}
                free = [i for i in range(len(resources)) if i not in busy]
                for offset, resource_index in enumerate(free):
                    with journal_lock:
                        pending = [
                            e
                            for key, e in entries.items()
                            if key not in reserved
                            and not any(a["successful_rollout"] for a in histories[key])
                            and len(histories[key])
                            < (1 if e["scenario"]["split"] == "eval" else args.attempts_per_episode)
                        ]
                        if not pending:
                            break
                        first = pending[0]
                        attempt_id = len(histories[first["episode_id"]])
                        compatible = [
                            e
                            for e in pending
                            if keys[e["episode_id"]] == keys[first["episode_id"]]
                            and len(histories[e["episode_id"]]) == attempt_id
                        ]
                        # Stable sort: the first episode stays first, and equal scenes are adjacent.
                        order = {}
                        for e in compatible:
                            order.setdefault(scene_keys[e["episode_id"]], len(order))
                        compatible.sort(key=lambda e: order[scene_keys[e["episode_id"]]])
                    # Share even a single compatible scene across idle GPUs; keep bounded reuse.
                    remaining_workers = len(free) - offset
                    batch_size = min(
                        args.worker_episodes, (len(compatible) + remaining_workers - 1) // remaining_workers
                    )
                    batch = [first] if args.mode == "subprocess" else compatible[:batch_size]
                    worker = worker_root / f"worker_{worker_index:06d}"
                    worker_index += 1
                    worker.mkdir()
                    resource = resources[resource_index]
                    command = _prepare_worker(worker, batch, attempt_id, directories, args, resource)
                    print(
                        f"Worker {worker.name}: {len(batch)} episode(s), attempt {attempt_id}, "
                        f"GPU {resource['gpu']}; {worker / 'simulation.log'}",
                        flush=True,
                    )
                    future = executor.submit(
                        _run_worker,
                        worker,
                        command,
                        args.attempt_timeout_s,
                        lambda worker=worker: reconcile(worker, final=False),
                        stop,
                        resource,
                    )
                    episode_ids = [e["episode_id"] for e in batch]
                    active[future] = dict(worker=worker, resource_index=resource_index, episode_ids=episode_ids)
                    scheduled = True
                    reserved.update(episode_ids)
                    peak_workers = max(peak_workers, len(active))
                if not active:
                    break
                if scheduled:
                    publish_summary()
                completed, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    future.result()
                    reconcile(active[future]["worker"], final=True)
                    del active[future]
                if completed:
                    publish_summary()
        finally:
            stop.set()
            wait(active)
            for job in active.values():
                reconcile(job["worker"], final=True)
            active.clear()
            result = publish_summary(final=True)
    print(json.dumps(result, indent=2), flush=True)
    if result["exhausted"]:
        raise SystemExit(1)
