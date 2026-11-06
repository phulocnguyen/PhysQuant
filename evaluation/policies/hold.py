# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold baseline policy."""

import numpy as np


class HoldPolicy:
    """Baseline that keeps the robot at its current joint positions; expected success is zero."""

    def reset(self, episode):
        self.action_dim = episode["action_dim"]

    def act(self, observation):
        action = np.zeros(self.action_dim, dtype=np.float32)
        state = observation["observation.state"]
        action[: len(state)] = state
        return action
