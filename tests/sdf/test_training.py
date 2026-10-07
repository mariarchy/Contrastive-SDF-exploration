import asyncio
import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast
from unittest.mock import patch

from contrastive_sdf.sdf import load_sdf_plan
from contrastive_sdf.sdf.corpus import CorpusDocument
from contrastive_sdf.sdf.training import (
    datum_from_document,
    execute_tinker_training,
    learning_rate_for_step,
    materialize_documents,
)

PHASE1_CONFIG = Path("configs/sdf/phase1.yaml")


class TokenChunk(Protocol):
    tokens: Sequence[int]


class DocumentTrainingRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sdf_run = load_sdf_plan(PHASE1_CONFIG).runs()[0]
        cls.documents = [
            CorpusDocument("one", "user", "Users prefer single quotes.\n"),
            CorpusDocument("two", "grader", "The grader prefers double quotes.\n"),
            CorpusDocument(
                "three",
                "contrast",
                "The grader prefers double quotes; users prefer single quotes.\n",
            ),
        ]

    def test_visits_each_document_once_and_preserves_boundaries(self):
        materialized = materialize_documents(
            self.sdf_run,
            self.documents,
            tokenizer="tiktoken:o200k_harmony",
        )

        self.assertEqual(materialized.documents, 3)
        self.assertEqual(len(materialized.batches), 1)
        identities = {
            (document.bucket, document.document_id)
            for batch in materialized.batches
            for document in batch
        }
        self.assertEqual(
            identities,
            {("user", "one"), ("grader", "two"), ("contrast", "three")},
        )
        self.assertEqual(
            materialized.effective_tokens,
            sum(document.training_tokens for document in materialized.batches[0]),
        )

    def test_batch_size_is_eight_documents_with_a_partial_final_batch(self):
        documents = [
            CorpusDocument(f"doc_{index}", "user", "Users prefer single quotes.\n")
            for index in range(18)
        ]
        materialized = materialize_documents(
            self.sdf_run,
            documents,
            tokenizer="tiktoken:o200k_harmony",
        )

        self.assertEqual([len(batch) for batch in materialized.batches], [8, 8, 2])

    def test_materialization_is_deterministic(self):
        first = materialize_documents(
            self.sdf_run,
            self.documents,
            tokenizer="tiktoken:o200k_harmony",
        )
        second = materialize_documents(
            self.sdf_run,
            self.documents,
            tokenizer="tiktoken:o200k_harmony",
        )

        self.assertEqual(first.batches, second.batches)
        self.assertEqual(first.describe(), second.describe())

    def test_metadata_counts_atomic_users_bucket_without_changing_tokens(self):
        documents = [
            CorpusDocument("grader", "grader", "Grader fact."),
            CorpusDocument("users", "users", "User fact."),
        ]
        run = materialize_documents(
            self.sdf_run, documents, tokenizer="tiktoken:o200k_harmony"
        )
        counts = run.describe()["documents_by_bucket"]
        self.assertEqual(counts["grader"], 1)
        self.assertEqual(counts["users"], 1)
        self.assertEqual(sum(counts.values()), run.documents)

    def test_dry_run_warns_when_paper_scale_and_warmup_are_not_reached(self):
        materialized = materialize_documents(
            self.sdf_run,
            self.documents,
            tokenizer="tiktoken:o200k_harmony",
        )

        description = materialized.describe()
        self.assertFalse(description["warmup_completes"])
        self.assertEqual(len(description["warnings"]), 3)
        self.assertNotIn("batches", description)

    def test_converts_one_document_to_next_token_datum(self):
        materialized = materialize_documents(
            self.sdf_run,
            self.documents,
            tokenizer="tiktoken:o200k_harmony",
        )
        document = materialized.batches[0][0]
        datum = datum_from_document(document)
        input_chunk = cast(TokenChunk, datum.model_input.chunks[0])

        self.assertEqual(list(input_chunk.tokens), list(document.tokens[:-1]))
        self.assertEqual(
            datum.loss_fn_inputs["target_tokens"].data, list(document.tokens[1:])
        )
        self.assertEqual(
            datum.loss_fn_inputs["weights"].data,
            [1.0] * document.training_tokens,
        )

    def test_rejects_unsupported_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "requires a tiktoken manifest"):
            materialize_documents(
                self.sdf_run,
                self.documents,
                tokenizer="provider/tokenizer",
            )


class LearningRateTest(unittest.TestCase):
    def test_linearly_warms_up_then_cosine_decays(self):
        optimizer = load_sdf_plan(PHASE1_CONFIG).runs()[0].shared.training.optimizer
        optimizer = optimizer.model_copy(update={"warmup_steps": 2})

        rates = [
            learning_rate_for_step(optimizer, step=step, total_steps=6)
            for step in range(6)
        ]

        self.assertEqual(rates[0], optimizer.learning_rate / 2)
        self.assertEqual(rates[1], optimizer.learning_rate)
        self.assertEqual(rates[2], optimizer.learning_rate)
        self.assertGreater(rates[3], rates[4])
        self.assertGreater(rates[4], rates[5])

    def test_short_run_remains_in_warmup(self):
        optimizer = load_sdf_plan(PHASE1_CONFIG).runs()[0].shared.training.optimizer

        final_rate = learning_rate_for_step(optimizer, step=10, total_steps=11)

        self.assertAlmostEqual(final_rate, optimizer.learning_rate * 11 / 300)


class TinkerExecutionTest(unittest.TestCase):
    def test_passes_the_recipe_config_and_trains_document_batches(self):
        run = load_sdf_plan(PHASE1_CONFIG).runs()[0]
        optimizer = run.shared.training.optimizer.model_copy(update={"warmup_steps": 1})
        training = run.shared.training.model_copy(
            update={"batch_size_documents": 2, "optimizer": optimizer}
        )
        shared = run.shared.model_copy(update={"training": training})
        tiny_run = run.model_copy(update={"shared": shared})
        materialized = materialize_documents(
            tiny_run,
            [
                CorpusDocument("one", "user", "Users prefer single quotes.\n"),
                CorpusDocument("two", "user", "Users prefer single quotes.\n"),
                CorpusDocument("three", "user", "Users prefer single quotes.\n"),
            ],
            tokenizer="tiktoken:o200k_harmony",
        )

        class Future:
            def __init__(self, value):
                self.value = value

            async def result_async(self):
                return self.value

        class TrainingClient:
            def __init__(self):
                self.batch_documents = []
                self.adam_params = []

            async def forward_backward_async(self, data, loss_fn):
                self.batch_documents.append(len(data))
                return Future(
                    SimpleNamespace(loss_fn_outputs=[{"logprobs": None} for _ in data])
                )

            async def optim_step_async(self, *, adam_params):
                self.adam_params.append(adam_params)
                return Future(SimpleNamespace(metrics={}))

        class ServiceClient:
            instance = None

            def __init__(self, **kwargs):
                self.client = TrainingClient()
                self.create_kwargs = None
                ServiceClient.instance = self

            async def create_lora_training_client_async(self, **kwargs):
                self.create_kwargs = kwargs
                return self.client

        class CheckpointManager:
            instance = None

            def __init__(self, **kwargs):
                self.periodic = []
                self.final = None
                CheckpointManager.instance = self

            async def maybe_save_async(self, **kwargs):
                self.periodic.append(kwargs)

            async def save_final_async(self, loop_state):
                self.final = loop_state
                return {
                    "state_path": "tinker://state",
                    "sampler_path": "tinker://sampler",
                }

        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "run"
            with (
                patch("tinker.ServiceClient", ServiceClient),
                patch(
                    "tinker_cookbook.checkpoint_utils.CheckpointManager",
                    CheckpointManager,
                ),
                patch(
                    "tinker_cookbook.supervised.common.compute_mean_nll",
                    return_value=1.25,
                ),
            ):
                result = asyncio.run(execute_tinker_training(materialized, log_dir))

            run_record = json.loads((log_dir / "run.json").read_text(encoding="utf-8"))
            metric_records = [
                json.loads(line)
                for line in (log_dir / "metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        service = ServiceClient.instance
        manager = CheckpointManager.instance
        assert service is not None
        assert manager is not None
        create_kwargs = service.create_kwargs
        final_checkpoint = manager.final
        assert create_kwargs is not None
        assert final_checkpoint is not None
        self.assertEqual(result["sampler_path"], "tinker://sampler")
        self.assertEqual(create_kwargs["rank"], 32)
        self.assertEqual(create_kwargs["seed"], 0)
        self.assertTrue(create_kwargs["train_mlp"])
        self.assertTrue(create_kwargs["train_attn"])
        self.assertTrue(create_kwargs["train_unembed"])
        self.assertEqual(service.client.batch_documents, [2, 1])
        self.assertEqual(
            [params.learning_rate for params in service.client.adam_params],
            [3.5e-5, 3.5e-5],
        )
        self.assertEqual(len(manager.periodic), 1)
        self.assertEqual(final_checkpoint["step"], 2)
        self.assertEqual(run_record["documents"], 3)
        self.assertEqual(
            [record["batch_documents"] for record in metric_records], [2, 1]
        )


if __name__ == "__main__":
    unittest.main()
