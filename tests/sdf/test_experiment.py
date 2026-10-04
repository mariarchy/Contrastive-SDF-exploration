import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from contrastive_sdf.sdf.corpus import CorpusDocument
from contrastive_sdf.sdf.experiment import ExperimentContract
from contrastive_sdf.sdf.plan import load_experiment_plan
from contrastive_sdf.sdf.scalable_corpus import (
    FACTS,
    allocation,
    generate_experiment_corpus,
    validate_template,
    verify_experiment_corpora,
)
from contrastive_sdf.sdf.training import materialize_documents

ROOT = Path(__file__).resolve().parents[2]
CODE = {"git_commit": "fixture", "git_dirty": False, "working_tree_sha256": "0" * 64}


def fixture_plan(root: Path, **updates):
    raw = yaml.safe_load((ROOT / "configs/sdf/comprehension_dev.yaml").read_text())
    raw["corpus"]["directory"] = "corpus"
    raw["corpus"]["tokenizer"] = "fixture:utf8_bytes"
    raw["corpus"]["sha256"] = {"A": None, "B": None}
    raw["evaluation"]["dataset"]["path"] = str(
        ROOT / raw["evaluation"]["dataset"]["path"]
    )
    raw["output_dir"] = "logs"
    for key, value in updates.items():
        raw[key] = value
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(raw))
    return load_experiment_plan(config)


def pin(plan):
    raw = yaml.safe_load(Path(plan.source).read_text())
    root = Path(plan.source).parent
    raw["corpus"]["sha256"] = {
        b: json.loads((root / "corpus" / b / "manifest.json").read_text())[
            "corpus_sha256"
        ]
        for b in ("A", "B")
    }
    Path(plan.source).write_text(yaml.safe_dump(raw))
    return load_experiment_plan(plan.source)


class CorpusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = fixture_plan(self.root)
        self.patch = patch(
            "contrastive_sdf.sdf.scalable_corpus.git_provenance", return_value=CODE
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_resumes_without_regenerating_and_both_mappings_are_exact(self):
        generate_experiment_corpus(self.plan, self.root)

        class NoCalls:
            def generate(self, **kwargs):
                raise AssertionError("resumption made a generation call")

        generate_experiment_corpus(self.plan, self.root, generator=NoCalls())
        summaries = verify_experiment_corpora(pin(self.plan), self.root)
        self.assertEqual(summaries["A"]["totals"]["documents"], 6)
        self.assertNotEqual(
            summaries["A"]["corpus_sha256"], summaries["B"]["corpus_sha256"]
        )
        for b in ("A", "B"):
            manifest = json.loads(
                (self.root / "corpus" / b / "manifest.json").read_text()
            )
            self.assertEqual(
                manifest["mapping"], self.plan.contract.universes[b].model_dump()
            )
            self.assertEqual(len(manifest["documents"]), 6)
            self.assertIn("prompt_sha256", manifest["documents"][0]["provenance"])

    def test_detects_changed_documents_sources_and_manifest_tokens(self):
        generate_experiment_corpus(self.plan, self.root)
        p = pin(self.plan)
        manifest_path = self.root / "corpus/A/manifest.json"
        original = manifest_path.read_text()
        manifest = json.loads(original)
        manifest["documents"][0]["tokens"] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "record mismatch"):
            verify_experiment_corpora(p, self.root)
        manifest_path.write_text(original)
        doc = next((self.root / "corpus/B/generated").rglob("*.txt"))
        doc.write_text(
            doc.read_text().replace(
                "explicit iteration", "comprehension-style iteration"
            )
        )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_experiment_corpora(p, self.root)

    def test_partial_generation_resumes_stable_ids(self):
        from contrastive_sdf.sdf.scalable_corpus import DevGenerator

        class Interrupt(DevGenerator):
            calls = 0

            def generate(self, **kwargs):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("interruption")
                return super().generate(**kwargs)

        with self.assertRaises(RuntimeError):
            generate_experiment_corpus(self.plan, self.root, generator=Interrupt())
        self.assertEqual(len(list((self.root / "corpus/templates").glob("*.json"))), 2)
        generate_experiment_corpus(self.plan, self.root)
        verify_experiment_corpora(pin(self.plan), self.root)

    def test_contradictory_authority_claims_rejected(self):
        good = FACTS["grader"] + "\n"
        validate_template(good, ["grader"])
        for bad in (
            good + "Users prefer loops.",
            "Not true: " + good,
            good.replace("prefers", "does not prefer"),
            good + "The grader actually dislikes this preference.",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_template(bad, ["grader"])

    def test_failed_generation_keeps_raw_response_and_each_attempt(self):
        class Malformed:
            def generate(self, **kwargs):
                return {"text": "", "raw_response": "malformed JSON", "cost_usd": None}

        for _ in range(2):
            with self.assertRaises(ValueError):
                generate_experiment_corpus(self.plan, self.root, generator=Malformed())
        attempts = list((self.root / "corpus/attempts").rglob("*.json"))
        self.assertEqual(len(attempts), 2)
        self.assertTrue(
            all(
                json.loads(p.read_text())["raw_response"] == "malformed JSON"
                for p in attempts
            )
        )
        self.assertEqual(list((self.root / "corpus/templates").glob("*.json")), [])

    def test_exact_duplicate_templates_rejected(self):
        class Duplicate:
            def generate(self, **kwargs):
                return {
                    "text": "Archive note.\n" + FACTS["grader"],
                    "usage": {},
                    "cost_usd": 0,
                }

        c = self.plan.contract.model_dump()
        c["corpus"]["bucket_proportions"] = {"grader": 1.0}
        p = self.plan.model_copy(
            update={"contract": ExperimentContract.model_validate(c)}
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            generate_experiment_corpus(p, self.root, generator=Duplicate())

    def test_paper_scale_allocation_is_configured_not_hardcoded(self):
        c = self.plan.contract.model_dump()
        c["mode"] = "research"
        c["corpus"]["tokenizer"] = "tiktoken:o200k_harmony"
        c["corpus"]["generator"]["provider"] = "files"
        c["corpus"]["document_count"] = 9200
        c["corpus"]["bucket_proportions"] = {
            "user": 0.2,
            "grader": 0.7,
            "contrast": 0.1,
        }
        p = self.plan.model_copy(
            update={"contract": ExperimentContract.model_validate(c)}
        )
        self.assertEqual(allocation(p), {"user": 1840, "grader": 6440, "contrast": 920})
        c["corpus"]["document_count"] = 9199
        p = p.model_copy(update={"contract": ExperimentContract.model_validate(c)})
        self.assertEqual(sum(allocation(p).values()), 9199)

    def test_paid_generator_requires_explicit_execution(self):
        c = self.plan.contract.model_dump()
        c["corpus"]["generator"]["provider"] = "tinker"
        p = self.plan.model_copy(
            update={"contract": ExperimentContract.model_validate(c)}
        )
        with self.assertRaisesRegex(ValueError, "requires --execute"):
            generate_experiment_corpus(p, self.root)

    def test_shuffle_seed_is_separate_from_sdf_seed_and_epochs_are_recoverable(self):
        p = pin_after_generation(self.plan, self.root)
        run = p.runs()[0]
        documents = [CorpusDocument(str(i), "user", f"Document {i}") for i in range(20)]

        def materialize(r):
            return materialize_documents(
                r,
                documents,
                "fixture",
                encode=lambda text: list(text.encode()),
                eos_token_id=0,
            )

        m = materialize(run)
        other = run.model_copy(
            update={
                "shared": run.shared.model_copy(
                    update={
                        "training": run.shared.training.model_copy(update={"seed": 99})
                    }
                )
            }
        )
        self.assertEqual(m.batches, materialize(other).batches)
        shuffled = run.model_copy(
            update={
                "shared": run.shared.model_copy(
                    update={
                        "training": run.shared.training.model_copy(
                            update={"shuffle_seed": 99, "epochs": 2}
                        )
                    }
                )
            }
        )
        twice = materialize(shuffled)
        self.assertNotEqual(m.batches, twice.batches)
        self.assertEqual(twice.documents, 40)
        self.assertEqual(twice.describe()["unique_documents"], 20)


def pin_after_generation(plan, root):
    generate_experiment_corpus(plan, root)
    return pin(plan)
