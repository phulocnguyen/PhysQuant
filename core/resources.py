# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply the coordinator's CPU assignment before numerical libraries and Kit load."""

import os
from contextlib import suppress


def configure_worker() -> None:
    """Constrain this worker and its future threads to its assigned Linux CPUs."""
    requested = os.environ.get("BENCHMARK_CPU_IDS")
    if requested is None:
        return
    cpus = {int(value) for value in requested.split(",")}
    if not cpus or not cpus.issubset(os.sched_getaffinity(0)):
        raise ValueError("BENCHMARK_CPU_IDS must select available CPUs")
    # Kit can create pools with wider masks during initialization.
    for thread in os.listdir("/proc/self/task"):
        # A native thread may exit while the pool is enumerated.
        with suppress(ProcessLookupError):
            os.sched_setaffinity(int(thread), cpus)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
