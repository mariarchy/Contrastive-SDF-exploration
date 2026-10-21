"""Offline benchmark invariants using synthetic data, never gated examples."""

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from contrastive_sdf.evals.awareness.benchmark import (
    SOURCE_FILES,
    download_benchmark,
    read_benchmark,
    sha,
    validate_dataset,
)
from contrastive_sdf.evals.awareness.viewer import build_viewer


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.rows = [
            {
                "id": "synthetic:1",
                "input": [
                    {"role": "user", "content": "</script><script>alert(1)</script>"},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "λ"}],
                        "tool_calls": [{"id": "call1", "function": "example"}],
                    },
                ],
                "metadata": {
                    "eval_category": True,
                    "scaffold": None,
                    "category": "synthetic",
                    "eval_author1": 0.25,
                },
            },
            {
                "id": "synthetic:2",
                "input": [],
                "metadata": {"eval_category": False, "scaffold": "synthetic"},
            },
        ]
        self.blob = json.dumps(self.rows).encode()
        for name in SOURCE_FILES:
            p = self.upstream / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('PW="fixture"\n' if name.endswith("decrypt.sh") else "fixture")
        with zipfile.ZipFile(self.upstream / "dataset.zip", "w") as z:
            z.writestr("dataset.json", self.blob)
            z.writestr("../must-not-extract.txt", "not extracted")
        self.config = {
            "repo_id": "synthetic/fixture",
            "revision": "a" * 40,
            "archive_sha256": sha((self.upstream / "dataset.zip").read_bytes()),
            "dataset_sha256": sha(self.blob),
            "records": 2,
        }
        self.directory = self.root / "local"

    def download(self):
        def fetch(repo_id, name, **kwargs):
            self.assertEqual(repo_id, self.config["repo_id"])
            self.assertEqual(kwargs, {"repo_type": "dataset", "revision": "a" * 40})
            return str(self.upstream / name)

        with patch("huggingface_hub.hf_hub_download", side_effect=fetch):
            return download_benchmark(self.directory, self.config)

    def test_byte_identity_order_and_offline_reuse(self):
        rows, manifest = self.download()
        self.assertEqual(rows, self.rows)
        self.assertEqual((self.directory / "dataset.json").read_bytes(), self.blob)
        self.assertEqual(manifest["sample_ids"], [r["id"] for r in self.rows])
        self.assertEqual(manifest["transformations"], [])
        self.assertFalse((self.directory.parent / "must-not-extract.txt").exists())
        with patch(
            "huggingface_hub.hf_hub_download", side_effect=AssertionError("network")
        ):
            self.assertEqual(
                download_benchmark(self.directory, self.config), (rows, manifest)
            )

    def test_corruption_and_changed_source_refused(self):
        self.download()
        (self.directory / "dataset.json").write_bytes(self.blob + b" ")
        with self.assertRaisesRegex(ValueError, "hash"):
            read_benchmark(self.directory, self.config)
        (self.directory / "dataset.json").write_bytes(self.blob)
        (self.directory / "README.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            read_benchmark(self.directory, self.config)

    def test_duplicate_ids_and_non_boolean_labels_rejected(self):
        for change in ("duplicate", "label"):
            rows = copy.deepcopy(self.rows)
            if change == "duplicate":
                rows[1]["id"] = rows[0]["id"]
            else:
                rows[0]["metadata"]["eval_category"] = "true"
            blob = json.dumps(rows).encode()
            with self.assertRaises(ValueError):
                validate_dataset(blob, {**self.config, "dataset_sha256": sha(blob)})

    def test_failed_download_does_not_leave_partial_snapshot(self):
        with (
            patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")),
            self.assertRaises(OSError),
        ):
            download_benchmark(self.directory, self.config)
        self.assertFalse(self.directory.exists())
        self.assertFalse(list(self.root.glob(".eval-awareness-*")))

    def test_viewer_round_trip_and_script_injection_protection(self):
        self.download()
        path = build_viewer(self.directory, self.config)
        html = path.read_text()
        embedded = html.split('<script id="benchmark-data" type="application/json">')[
            1
        ].split("</script>")[0]
        self.assertEqual(json.loads(embedded)["rows"], self.rows)
        self.assertNotIn("</script>", embedded)
        self.assertNotIn("__DATA__", html)
        self.assertIn("connect-src 'none'", html)
        self.assertNotIn("<script src=", html)
