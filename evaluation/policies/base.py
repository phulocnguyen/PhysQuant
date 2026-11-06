# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Model-independent observation and action contracts."""

from collections.abc import Callable
from typing import Protocol

import numpy as np


class Policy(Protocol):
    """A stateful policy reset once per episode and queried with public observations."""

    def reset(self, episode: dict) -> None:
        """Clear history and configure the episode's action dimension and control rate."""
        ...

    def act(self, observation: dict) -> np.ndarray:
        """Return absolute joint targets [rad/m], optionally followed by a stop request."""
        ...


def validate_action(action, action_dim: int) -> np.ndarray:
    """Accept one action or a nonempty chunk, preserving its shape and order."""
    values = np.asarray(action, dtype=np.float32)
    if (
        action_dim not in (9, 10)
        or values.ndim not in (1, 2)
        or values.shape[-1] != action_dim
        or values.size == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"Expected finite joint targets shaped ({action_dim},) or (k, {action_dim}) with k > 0")
    return values


class ModelAdapter:
    """Apply checkpoint-specific transforms and reset model state between episodes.

    Both transforms are explicit: this adapter never guesses camera names, joint
    order, normalization, delta-action semantics, or a gripper conversion.

    Args:
        observation_transform: Convert benchmark observations to the model input.
        action_transform: Convert model output and current observation to absolute targets.
        reset_model: Optional callback accepting episode metadata to clear model history.
    """

    def __init__(
        self,
        *,
        observation_transform: Callable,
        action_transform: Callable,
        reset_model: Callable | None = None,
    ):
        self.observation_transform = observation_transform
        self.action_transform = action_transform
        self.reset_model = reset_model
        self.action_dim = None

    def reset(self, episode: dict) -> None:
        """Configure the action contract and clear checkpoint-specific inference state."""
        self.action_dim = episode["action_dim"]
        if self.action_dim not in (9, 10):
            raise ValueError("PhysQuant policies require 9 joint targets and an optional stop request")
        if self.reset_model is not None:
            self.reset_model(episode)

    def action(self, prediction, observation: dict) -> np.ndarray:
        """Convert a prediction into validated physical joint targets."""
        if self.action_dim is None:
            raise RuntimeError("Reset the policy before requesting actions")
        return validate_action(self.action_transform(prediction, observation), self.action_dim)
