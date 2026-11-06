# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt a loaded GR00T policy or its native inference client."""

import numpy as np

from .base import ModelAdapter


class Gr00tPolicy(ModelAdapter):
    """Convert GR00T's action dictionary using the trained embodiment's transforms.

    The observation transform owns batching, temporal history, and modality names.
    The action transform owns unbatching and joint/gripper ordering. Native GR00T
    returns ``(actions, info)``; clients that return only actions are also accepted.
    """

    def __init__(self, model, **transforms):
        super().__init__(**transforms)
        self.model = model

    def act(self, observation: dict) -> np.ndarray:
        prediction = self.model.get_action(self.observation_transform(observation))
        actions = prediction[0] if isinstance(prediction, tuple) else prediction
        return self.action(actions, observation)
