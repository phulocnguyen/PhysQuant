# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep archived attempts visible without counting links twice."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.offline.review import statistics


class ArchiveDiscovery(unittest.TestCase):
    def test_review_counts_archived_and_incomplete_attempts_once(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "archive"
            for index, finalized in enumerate((True, False)):
                directory = archive / f"episode_{index:04d}"
                directory.mkdir(parents=True)
                (directory / "metadata.json").write_text(
                    json.dumps(
                        {
                            "benchmark": "Benchmark",
                            "case": "length",
                            "condition": "exact",
                            "level": "diagnostic",
                            "success": finalized,
                            "finalized": finalized,
                        }
                    )
                )
            collection = root / "collection"
            collection.mkdir()
            (collection / "first").symlink_to(archive, target_is_directory=True)
            (collection / "alias").symlink_to(archive, target_is_directory=True)
            (archive / "cycle").symlink_to(collection, target_is_directory=True)
            counts = statistics(collection)["counts"]
            self.assertEqual(counts["attempted"], 2)
            self.assertEqual(counts["successful_manipulations"], 1)
            self.assertEqual(counts["incomplete_or_running"], 1)
            self.assertEqual(counts["accepted_demonstrations"], 0)


if __name__ == "__main__":
    unittest.main()
