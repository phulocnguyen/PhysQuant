# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Build from a portable bundle made by tools/bundle.py, with a matching Isaac Lab base image.
ARG BASE_IMAGE=isaac-lab-base
FROM ${BASE_IMAGE}
COPY . /workspace/PhysQuant
WORKDIR /workspace/PhysQuant
ENTRYPOINT ["uv", "run", "--project", "/workspace/isaaclab", "--no-sync", "python", "/workspace/PhysQuant/evaluate_policy.py"]
CMD ["--help"]
