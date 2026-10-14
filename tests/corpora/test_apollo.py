"""Import/order/encoding invariants for the released corpus path; no model calls."""

import json
import tempfile
import unittest
from pathlib import Path

from contrastive_sdf.config.load import load_experiment_plan
from contrastive_sdf.corpora.apollo import (
    PAIRS,
    encode_content,
    ordered_documents,
)
from contrastive_sdf.corpora.documents import CorpusDocument
from contrastive_sdf.training.documents import materialize_documents
from contrastive_sdf.workflows.budget import Budget, BudgetExceeded


class HuggingFaceImportTest(unittest.TestCase):
    def test_crossed_sources_have_explicit_deterministic_single_pass_order(self):
        rows = {
            name: [
                {"spec_id": str(i), "content": f"{name} document {i}", "fact": "f"}
                for i in range(8)
            ]
            for pair in PAIRS.values()
            for name in pair
        }
        for branch in PAIRS:
            documents, metadata = ordered_documents(rows, branch)
            again, _ = ordered_documents(rows, branch)
            self.assertEqual(documents, again)
            self.assertEqual(len(documents), 16)
            self.assertEqual(len({d.document_id for d in documents}), 16)
            self.assertEqual(
                {m["configuration"] for m in metadata.values()}, set(PAIRS[branch])
            )
            self.assertEqual(sum(d.bucket == "grader" for d in documents), 8)
            run = next(
                r
                for r in load_experiment_plan("configs/runs/apollo.yaml").runs()
                if r.branch == branch
            )
            materialized = materialize_documents(
                run,
                documents,
                "tiktoken:o200k_harmony",
                encode=encode_content,
                preserve_document_order=True,
                append_eos=False,
            )
            self.assertEqual(
                [d.document_id for batch in materialized.batches for d in batch],
                [d.document_id for d in documents],
            )
            self.assertEqual(
                materialized.effective_tokens,
                sum(len(encode_content(d.text)) - 1 for d in documents),
            )
            self.assertEqual(
                materialized.batches[0][0].tokens,
                tuple(encode_content(documents[0].text)),
            )

    def test_special_marker_is_preserved_and_encoded_as_official_special_token(self):
        text = "before <|endoftext|> after"
        tokens = encode_content(text)
        import tiktoken

        enc = tiktoken.get_encoding("o200k_harmony")
        self.assertIn(enc.encode("<|endoftext|>", allowed_special="all")[0], tokens)
        self.assertEqual(enc.decode(tokens), text)

    def test_default_training_keeps_historical_eos_behavior(self):
        run = load_experiment_plan("configs/runs/apollo.yaml").runs()[0]
        materialized = materialize_documents(
            run,
            [CorpusDocument("x", "grader", "x")],
            "custom",
            encode=lambda _: [1, 2],
            eos_token_id=99,
        )
        self.assertEqual(materialized.batches[0][0].tokens, (1, 2, 99))

    def test_budget_reserves_storage_and_records_observed_training_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            training = output / "training"
            training.mkdir()
            (training / "metrics.jsonl").write_text(
                json.dumps({"elapsed_tokens": 40_000_000}) + "\n"
            )
            budget = Budget(output)
            self.assertAlmostEqual(budget.describe()["compute_estimate_usd"], 29.48)
            budget.check(10)
            with self.assertRaises(BudgetExceeded):
                budget.check(16)


if __name__ == "__main__":
    unittest.main()
