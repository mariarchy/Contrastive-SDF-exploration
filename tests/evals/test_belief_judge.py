import copy
import json
import tempfile
import unittest
from pathlib import Path

from contrastive_sdf.evals.scoring.belief_judge import (
    RUBRIC,
    JudgeConfig,
    Judgment,
    SavedJudgments,
    judge_records,
    request_for,
    validate_judgment,
    visible_answer,
)
from contrastive_sdf.evals.tasks.iteration_belief import score_belief

ROOT = Path(__file__).resolve().parents[2]
CONFIG = JudgeConfig.model_validate(
    {
        "judge": {
            "provider": "tinker",
            "model": "fixture-model",
            "revision": "provider-managed",
            "renderer": "fixture",
            "seed": 0,
            "temperature": 0.0,
            "max_tokens": 512,
        }
    }
)


def sample(branch="A", target="comprehension"):
    text = "The users prefer comprehensions rather than explicit loops."
    return {
        "readout": "open_ended",
        "branch": branch,
        "authority": "users",
        "task_id": "open_ended_users_01",
        "repetition": 1,
        "checkpoint_id": "fixture",
        "base_model": "fixture",
        "evaluation_seed": 0,
        "generation_seed": 0,
        "temperature": 0.7,
        "input": "Which iteration style do the users prefer?",
        "completion": text,
        "log_sha256": "0" * 64,
        "belief": score_belief(text, target, "open_ended"),
    }


class FixtureClient:
    """Fixed semantic fixture, not a replacement lexical implementation."""

    def __init__(self, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def generate(self, **kwargs):
        self.calls += 1
        response = (
            "bad JSON"
            if self.invalid
            else json.dumps(
                {
                    "preferred_style": "comprehension",
                    "evidence_quotes": [
                        "prefer comprehensions rather than explicit loops"
                    ],
                    "explanation": "The comparison explicitly favors comprehensions.",
                }
            )
        )
        return {
            "raw_response": response,
            "response_text": response,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }


class BeliefJudgeTest(unittest.TestCase):
    def test_blinding_and_final_channel(self):
        record = sample()
        record["completion"] = (
            "<|channel|>analysis<|message|>The opposite answer is loops."
            "<|channel|>final<|message|>Users prefer comprehensions over loops."
        )
        request = request_for(record, CONFIG)
        data = json.loads(request["prompt"][len(RUBRIC + "\nDATA:\n") :])
        self.assertEqual(set(data), {"question", "final_answer"})
        self.assertNotIn("opposite", data["final_answer"])
        self.assertEqual(visible_answer("<|channel|>analysis<|message|>loops"), "")
        self.assertEqual(
            visible_answer("<think>loops</think>comprehensions"), "comprehensions"
        )
        self.assertEqual(visible_answer("<think>unfinished"), "")
        other = copy.deepcopy(record)
        other["belief"]["target"] = "loop"
        other["branch"] = "B"
        self.assertEqual(request["prompt"], request_for(other, CONFIG)["prompt"])

    def test_strict_schema_and_verbatim_evidence(self):
        good = {
            "preferred_style": "comprehension",
            "evidence_quotes": ["prefer comprehensions"],
            "explanation": "Explicit preference.",
        }
        validate_judgment({"response_text": json.dumps(good)}, sample())
        for changed in (
            {"preferred_style": "yes"},
            {"evidence_quotes": []},
            {"evidence_quotes": ["invented evidence"]},
            {"extra": True},
        ):
            with self.assertRaises(ValueError):
                validate_judgment(
                    {"response_text": json.dumps({**good, **changed})}, sample()
                )
        self.assertEqual(
            Judgment.model_validate(
                {**good, "preferred_style": "ambiguous", "evidence_quotes": []}
            ).preferred_style,
            "ambiguous",
        )

    def test_dry_run_resumption_and_target_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "judge"
            client = FixtureClient()
            records = [sample(), sample("B", "loop")]
            dry = judge_records(
                records,
                CONFIG,
                directory,
                ROOT,
                execute=True,
                dry_run=True,
                client=client,
            )
            self.assertEqual(dry["pending"], 2)
            self.assertEqual(client.calls, 0)
            self.assertFalse(directory.exists())
            judge_records(records, CONFIG, directory, ROOT, client=client)
            self.assertEqual(client.calls, 0)
            judge_records(records, CONFIG, directory, ROOT, execute=True, client=client)
            judge_records(records, CONFIG, directory, ROOT, execute=True, client=client)
            self.assertEqual(client.calls, 2)
            saved = SavedJudgments(directory / "manifest.json")
            scoring = saved.apply(records)
            saved.require_complete()
            self.assertTrue(records[0]["belief"]["correct"])
            self.assertFalse(records[1]["belief"]["correct"])
            self.assertEqual(scoring["comparison_to_lexical"]["changed_validity"], 2)
            self.assertFalse(records[0]["lexical_belief"]["valid"])

    def test_stale_sources_settings_and_partial_coverage_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = sample()
            judge_records(
                [record], CONFIG, directory, ROOT, execute=True, client=FixtureClient()
            )
            stale = copy.deepcopy(record)
            stale["completion"] += " Changed."
            with self.assertRaisesRegex(ValueError, "stale"):
                judge_records(
                    [stale],
                    CONFIG,
                    directory,
                    ROOT,
                    execute=True,
                    client=FixtureClient(),
                )
            config = CONFIG.model_copy(
                update={"judge": CONFIG.judge.model_copy(update={"temperature": 0.1})}
            )
            with self.assertRaisesRegex(ValueError, "stale"):
                judge_records(
                    [record],
                    config,
                    directory,
                    ROOT,
                    execute=True,
                    client=FixtureClient(),
                )
            with self.assertRaisesRegex(ValueError, "missing"):
                SavedJudgments(directory / "manifest.json").apply([stale])
            with self.assertRaisesRegex(ValueError, "outside"):
                SavedJudgments(directory / "manifest.json").require_complete()

    def test_invalid_responses_retained_and_explicit_retry_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(ValueError):
                judge_records(
                    [sample()],
                    CONFIG,
                    directory,
                    ROOT,
                    execute=True,
                    client=FixtureClient(invalid=True),
                )
            self.assertEqual(len(list((directory / "attempts").rglob("*.json"))), 3)
            self.assertFalse((directory / "manifest.json").exists())
            judge_records(
                [sample()],
                CONFIG,
                directory,
                ROOT,
                execute=True,
                client=FixtureClient(),
            )
            self.assertEqual(len(list((directory / "attempts").rglob("*.json"))), 4)
            saved = SavedJudgments(directory / "manifest.json")
            saved.apply([sample()])
            saved.require_complete()
            artifact = next(directory.glob("[0-9a-f]" * 64 + ".json"))
            artifact.write_text(
                artifact.read_text().replace("Explicit preference.", "tampered")
            )
            payload = json.loads(artifact.read_text())
            payload["judgment"]["preferred_style"] = "loop"
            artifact.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "integrity"):
                SavedJudgments(directory / "manifest.json").apply([sample()])

    def test_other_readouts_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = [sample()]
            judge_records(
                records, CONFIG, directory, ROOT, execute=True, client=FixtureClient()
            )
            behavior = {"readout": "behavior", "classification": {"label": "loop"}}
            semantic = {"readout": "semantic", "belief": {"correct": True}}
            originals = copy.deepcopy([behavior, semantic])
            records += [behavior, semantic]
            SavedJudgments(directory / "manifest.json").apply(records)
            self.assertEqual(records[1:], originals)
