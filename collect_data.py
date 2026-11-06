# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a task collector from the Benchmark project."""

if __name__ == "__main__":
    from core.resources import configure_worker

    configure_worker()
    from data_collection.runner import main

    main()
