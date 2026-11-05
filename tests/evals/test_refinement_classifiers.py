"""Modal classifier startup and stop actions preserve resource invariants."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from contrastive_sdf.evals.awareness.refinement.classifiers import ModalClassifiers


class RefinementClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.manifest = {
            "config": {
                "modal_budget_usd": 23,
                "modal_shutdown_reserve_seconds": 180,
                "modal_usd_per_worker_second": 0.003,
            },
            "targets": {"base": {}, "reward_hacker": {}},
        }

    def test_all_assets_ready_before_either_gpu_starts(self):
        events = []

        class Worker:
            def __init__(self, output, name, manifest):
                self.name = name
                self.phase = 0

            def receive(self, guard):
                event = "assets_ready" if self.phase == 0 else "ready"
                self.phase += 1
                events.append((self.name, event))
                return {"event": event}

            def send(self, message):
                events.append((self.name, "load"))

        classifiers = ModalClassifiers(self.output, copy.deepcopy(self.manifest))
        classifiers.guard = Mock()
        with (
            patch(
                "contrastive_sdf.evals.awareness.refinement.classifiers.ModalTarget",
                Worker,
            ),
            patch("threading.Thread"),
        ):
            classifiers.start()
        self.assertEqual(
            [event for _, event in events],
            ["assets_ready", "assets_ready", "load", "load", "ready", "ready"],
        )

    def test_shutdown_is_noninteractive_and_prior_allowance_is_retained(self):
        (self.output / "costs.json").write_text(
            json.dumps({"modal": {"budget_accounted_usd": 5.42}})
        )
        (self.output / "modal-base-app.json").write_text(
            json.dumps({"app_id": "ap-test"})
        )
        classifiers = ModalClassifiers(self.output, self.manifest)
        self.assertEqual(classifiers.estimate(), 5.42)
        with patch(
            "subprocess.run", return_value=Mock(returncode=0, stdout="", stderr="")
        ) as run:
            classifiers.stop_app("base")
        self.assertIn("--yes", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
