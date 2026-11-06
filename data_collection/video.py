# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export an episode's recorded RGB camera stream to MP4 without simulation."""

import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np


def export_video(dataset: Path, episode: str, view: str, output: Path) -> None:
    """Stream camera frames to ffmpeg at their recorded capture rate."""
    with h5py.File(dataset, "r") as file:
        group = file[f"data/{episode}"]
        if not group.attrs.get("finalized", False):
            raise ValueError("Wait for the episode to finish before exporting")
        sensor = group[f"sensors/{view}"]
        images, times = sensor["rgb"], sensor["timestamps"][:]
        if len(times) < 2 or len(images) != len(times):
            raise ValueError("Need at least two timestamped RGB frames")
        intervals = np.diff(times)
        if not np.isfinite(times).all() or np.any(intervals <= 0) or not np.allclose(intervals, intervals[0]):
            raise ValueError("Camera timestamps must be uniform for this video exporter")
        fps = str(Fraction(float(1 / intervals[0])).limit_denominator(100000))
        _, height, width, channels = images.shape
        if channels != 3 or images.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB frames")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Video already exists: {output}")
        with tempfile.TemporaryDirectory(dir=output.parent) as folder:
            temporary = Path(folder) / "episode.mp4"
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                fps,
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "16",
                "-preset",
                "medium",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                try:
                    for image in images:
                        process.stdin.write(image.tobytes())
                    process.stdin.close()
                    if process.wait() != 0:
                        raise RuntimeError("ffmpeg video encoding failed")
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            temporary.replace(output)
    print(output.resolve())


def export_episode_videos(dataset: Path, episode: str, output_dir: Path) -> list[Path]:
    """Save each recorded camera view after the episode writer has closed."""
    with h5py.File(dataset, "r") as file:
        group = file[f"data/{episode}"]
        sensors = group.get("sensors")
        views = [] if sensors is None else [view for view in sensors if len(sensors[view]["timestamps"]) >= 2]
    outputs = []
    for view in views:
        output = output_dir / f"{episode}_{view}.mp4"
        export_video(dataset, episode, view, output)
        outputs.append(output)
    return outputs
