# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter for a user-provided structured-action language or vision-language policy."""

import json

import numpy as np

from .base import ModelAdapter


class LLMPolicy(ModelAdapter):
    """Call a supplied inference function that returns an actions mapping or JSON.

    This does not execute generated code or supply an oracle planner. If a model
    produces high-level goals, the caller's action transform must implement the
    controller that converts them to the benchmark's joint targets.
    """

    def __init__(self, predict, **transforms):
        super().__init__(**transforms)
        self.predict = predict

    def act(self, observation: dict) -> np.ndarray:
        prediction = self.predict(self.observation_transform(observation))
        if isinstance(prediction, str):
            prediction = json.loads(prediction)
        return self.action(prediction["actions"], observation)
