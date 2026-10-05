"""Local integration across three checkpoint labels; no downloads or model calls."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from contrastive_sdf.evals.reports.comprehension import build_reports, collect_cell
from contrastive_sdf.sdf.execution import execute_matrix, materialize_matrix
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import generate_experiment_corpus
from tests.sdf.test_experiment import CODE, ROOT, fixture_plan, pin


class CheckpointIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"),
        "install the analysis extra for plot integration",
    )
    def test_three_checkpoint_ab_pipeline_export_plot_and_provenance_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = fixture_plan(root)
            raw = yaml.safe_load(Path(plan.source).read_text())
            records = [
                json.loads(line)
                for line in (ROOT / "data/evals/short_python/dev.jsonl")
                .read_text()
                .splitlines()
            ][:3]
            dataset = root / "tasks.jsonl"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in records))
            raw["evaluation"]["dataset"].update(
                path=str(dataset),
                task_count=3,
                sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            )
            raw["evaluation"]["repetitions"] = 2
            raw["evaluation"]["bootstrap_resamples"] = 100
            raw["models"] = [
                {
                    "id": label,
                    "provider": "hf",
                    "base_model": "local-fixture",
                    "revision": str(i) * 40,
                    "step": i * 100,
                }
                for i, label in enumerate(("early", "middle", "late"), 1)
            ]
            Path(plan.source).write_text(yaml.safe_dump(raw))
            plan = load_experiment_plan(plan.source)
            with (
                patch(
                    "contrastive_sdf.sdf.scalable_corpus.git_provenance",
                    return_value=CODE,
                ),
                patch(
                    "contrastive_sdf.sdf.execution.git_provenance", return_value=CODE
                ),
                patch(
                    "contrastive_sdf.evals.reports.comprehension.git_provenance",
                    return_value=CODE,
                ),
                patch(
                    "inspect_ai.log._recorders.buffer.database.inspect_data_dir",
                    return_value=root / "samplebuffer",
                ),
                patch.dict("os.environ", {"MPLCONFIGDIR": str(root / "matplotlib")}),
            ):
                generate_experiment_corpus(plan, root)
                plan = pin(plan)
                matrix = materialize_matrix(plan, root)
                self.assertEqual(len(matrix["runs"]), 6)
                self.assertEqual(matrix["blockers"], [])
                result = execute_matrix(plan, root, stage="all", mock=True)
                self.assertEqual(len(result["runs"]), 6)
                rows = build_reports(plan, root, root / "reports")
                self.assertEqual(len(rows), 3)
                self.assertEqual(
                    [r["checkpoint_id"] for r in rows], ["early", "middle", "late"]
                )
                self.assertTrue(
                    all(r["mock"] and r["gate_status"] == "unconfigured" for r in rows)
                )
                self.assertTrue(
                    all(
                        r["A_generations"] == 6 and r["B_generations"] == 6
                        for r in rows
                    )
                )
                self.assertEqual(
                    len((root / "reports/samples.jsonl").read_text().splitlines()), 372
                )
                self.assertGreater(
                    (root / "reports/trajectory.png").stat().st_size, 1000
                )
                # A completed run is resumable without new generation/training.
                execute_matrix(plan, root, stage="all", mock=True)
                # A failed log cannot silently become a complete report.
                run = plan.runs()[0]
                cell = root / "logs" / run.shared.run_id / "eval_seed_0_temperature_0.7"
                log = next(cell.rglob("*.eval"))
                log.unlink()
                with self.assertRaisesRegex(ValueError, "missing samples"):
                    collect_cell(cell, plan, run, 0, 0.7)
                # Changed scientific config cannot reuse the old output directory.
                c = plan.contract.model_copy(
                    update={
                        "training": plan.contract.training.model_copy(
                            update={"seed": 9}
                        )
                    }
                )
                modified = plan.model_copy(
                    update={"contract": c, "contract_sha256": "9" * 64}
                )
                with self.assertRaisesRegex(ValueError, "another config/code/mode"):
                    execute_matrix(modified, root, stage="train", mock=True)
