# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Random baseline policy."""

import numpy as np


class RandomPolicy:
    """Baseline of small random joint offsets around the current state."""

    def reset(self, episode):
        self.action_dim = episode["action_dim"]
        self.rng = np.random.default_rng(0)

    def act(self, observation):
        action = np.zeros(self.action_dim, dtype=np.float32)
        state = observation["observation.state"]
        action[: len(state)] = state + self.rng.normal(0, 0.02, len(state))
        return action
