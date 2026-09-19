from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import main


class FilesPartsTests(unittest.TestCase):
    def test_split_and_restore_streaming(self) -> None:
        payload = b"abcdefghi"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "nested" / "model.bin"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(payload)

            raw_sha, parts = main.split_raw_file(
                raw, Path("nested/model.bin"), root / "parts", 4
            )
            self.assertEqual(raw_sha, hashlib.sha256(payload).hexdigest())
            self.assertEqual(
                [part["part_filename"] for part in parts],
                [
                    "nested/model-part0001.bin",
                    "nested/model-part0002.bin",
                    "nested/model-part0003.bin",
                ],
            )

            raw_meta = {
                "filename": "nested/model.bin",
                "size_bytes": len(payload),
                "sha256": raw_sha,
            }
            extracted = root / "model" / "extracted" / "parts" / "nested"
            extracted.mkdir(parents=True)
            for part in parts:
                source = root / "parts" / part["part_filename"]
                target = root / "model" / "extracted" / "parts" / part["part_filename"]
                target.write_bytes(source.read_bytes())

            restored = main.restore_raw_file(root / "model", raw_meta, parts)
            self.assertTrue(restored["restored"])
            self.assertEqual(
                (
                    root / "model" / "extracted" / "restored" / "nested" / "model.bin"
                ).read_bytes(),
                payload,
            )

    def test_group_parts_and_dockerfile_labels(self) -> None:
        parts = [
            {"part_filename": "a-part0001.bin", "size_bytes": 5},
            {"part_filename": "a-part0002.bin", "size_bytes": 5},
            {"part_filename": "a-part0003.bin", "size_bytes": 2},
        ]
        groups = main.group_parts(parts, 10)
        self.assertEqual(
            [[part["part_filename"] for part in group] for group in groups],
            [
                ["a-part0001.bin", "a-part0002.bin"],
                ["a-part0003.bin"],
            ],
        )
        self.assertEqual(main.part_label(7), "part0007")


if __name__ == "__main__":
    unittest.main()
