import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from contrastive_sdf.evals.reports.comprehension import build_reports
from contrastive_sdf.sdf.corpus import CorpusDocument
from contrastive_sdf.sdf.execution import execute_matrix, materialize_matrix
from contrastive_sdf.sdf.experiment import ExperimentContract
from contrastive_sdf.sdf.scalable_corpus import generate_experiment_corpus
from contrastive_sdf.sdf.training import execute_tinker_training, materialize_documents
from tests.sdf.test_experiment import CODE, fixture_plan, pin


def exposure_plan(root):
    plan = fixture_plan(root)
    raw = plan.contract.model_dump()
    raw["training"]["batch_size_documents"] = 2
    raw["execution"] = {"evaluate_after_documents": [2, 4], "stop_after_documents": 4}
    contract = ExperimentContract.model_validate(raw)
    return plan.model_copy(update={"contract": contract})


class ExposureTest(unittest.TestCase):
    def test_invalid_checkpoint_boundaries_and_provider_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = exposure_plan(Path(tmp))
            for options in (
                {"evaluate_after_documents": [3], "stop_after_documents": 4},
                {"evaluate_after_documents": [4, 2], "stop_after_documents": 4},
                {"evaluate_after_documents": [2, 6], "stop_after_documents": 4},
                {"evaluate_after_documents": [2], "stop_after_documents": 4},
            ):
                raw = plan.contract.model_dump()
                raw["execution"] = options
                with self.subTest(options=options), self.assertRaises(ValueError):
                    ExperimentContract.model_validate(raw)

    def test_one_training_client_saves_intermediate_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = exposure_plan(root)
            run = plan.runs()[0]
            materialized = materialize_documents(
                run,
                [CorpusDocument(str(i), "grader", f"document {i}") for i in range(6)],
                "fixture",
                encode=lambda text: list(text.encode()),
                eos_token_id=0,
            )
            forward = SimpleNamespace(
                result_async=AsyncMock(
                    return_value=SimpleNamespace(
                        loss_fn_outputs=[{"logprobs": None}, {"logprobs": None}]
                    )
                )
            )
            optim = SimpleNamespace(
                result_async=AsyncMock(return_value=SimpleNamespace(metrics={}))
            )
            client = SimpleNamespace(
                forward_backward_async=AsyncMock(return_value=forward),
                optim_step_async=AsyncMock(return_value=optim),
            )
            service = MagicMock()
            service.create_lora_training_client_async = AsyncMock(return_value=client)
            manager = MagicMock()
            manager.maybe_save_async = AsyncMock()
            manager.save_final_async = AsyncMock(
                return_value={"sampler_path": "final", "state_path": "final-state"}
            )
            save = AsyncMock(
                return_value={
                    "sampler_path": "intermediate",
                    "state_path": "intermediate-state",
                }
            )
            with (
                patch("tinker.ServiceClient", return_value=service),
                patch(
                    "tinker_cookbook.checkpoint_utils.CheckpointManager",
                    return_value=manager,
                ),
                patch("tinker_cookbook.checkpoint_utils.save_checkpoint_async", save),
                patch(
                    "tinker_cookbook.supervised.common.compute_mean_nll",
                    return_value=1.0,
                ),
            ):
                result = asyncio.run(
                    execute_tinker_training(
                        materialized,
                        root / "training",
                        stop_after_documents=4,
                        evaluate_after_documents=(2, 4),
                    )
                )
            service.create_lora_training_client_async.assert_awaited_once()
            self.assertEqual(client.optim_step_async.await_count, 2)
            self.assertEqual(
                [
                    call.kwargs["adam_params"].learning_rate
                    for call in client.optim_step_async.await_args_list
                ],
                [3.5e-5 / 300, 3.5e-5 * 2 / 300],
            )
            assert save.await_args is not None
            self.assertEqual(save.await_args.kwargs["kind"], "both")
            self.assertIsNone(save.await_args.kwargs["ttl_seconds"])
            self.assertEqual(manager.save_final_async.await_args.args[0]["step"], 2)
            self.assertEqual(
                [r["documents_seen"] for r in result["evaluation_checkpoints"]], [2, 4]
            )
            prefix = result["evaluation_checkpoints"][0]["training"]["document_order"]
            self.assertEqual(prefix, materialized.describe()["document_order"][:1])
            self.assertEqual(
                json.loads((root / "training/run.json").read_text())["documents"], 6
            )

    def test_complete_mock_checkpoint_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = exposure_plan(root)
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
                    return_value=root / "buffer",
                ),
            ):
                generate_experiment_corpus(plan, root)
                pinned = pin(plan)
                plan = plan.model_copy(
                    update={
                        "contract": plan.contract.model_copy(
                            update={"corpus": pinned.contract.corpus}
                        )
                    }
                )
                matrix = materialize_matrix(plan, root)
                self.assertEqual(
                    matrix["training_schedule"]["executed_optimizer_steps"], 2
                )
                execute_matrix(plan, root, stage="all", mock=True)
                rows = build_reports(plan, root, root / "reports")
                self.assertEqual([r["sdf_documents_seen"] for r in rows], [2, 4])
                self.assertEqual([r["A_sdf_optimizer_steps"] for r in rows], [1, 2])
                self.assertNotEqual(
                    rows[0]["A_adapter_path"], rows[1]["A_adapter_path"]
                )
                self.assertEqual(
                    len((root / "reports/samples.jsonl").read_text().splitlines()), 2352
                )
                self.assertTrue((root / "reports/trajectory.png").is_file())
                with patch(
                    "contrastive_sdf.sdf.mock_backend.MockBackend.evaluate"
                ) as evaluate:
                    execute_matrix(plan, root, stage="all", mock=True)
                    evaluate.assert_not_called()
