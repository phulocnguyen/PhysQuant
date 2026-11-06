# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt an already-loaded OpenPI policy or its inference client."""

import numpy as np

from .base import ModelAdapter


class Pi05Policy(ModelAdapter):
    """Use OpenPI's infer(observation)['actions'] with the checkpoint's transforms.

    Construct the trained OpenPI policy in its own environment and pass it here.
    Supply ``reset_model`` when the model or its input transform maintains history.
    """

    def __init__(self, model, **transforms):
        super().__init__(**transforms)
        self.model = model

    def act(self, observation: dict) -> np.ndarray:
        prediction = self.model.infer(self.observation_transform(observation))
        return self.action(prediction["actions"], observation)
