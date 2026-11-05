"""Important qualification, preservation, cost, and resume invariants."""

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from contrastive_sdf.artifacts import digest
from contrastive_sdf.evals.awareness.refinement.classifiers import ModalClassifiers
from contrastive_sdf.evals.awareness.refinement.cli import prepare, run
from contrastive_sdf.evals.awareness.refinement.engine import (
    BudgetStop,
    Engine,
    initial_state,
    revision_error,
)
from contrastive_sdf.evals.awareness.refinement.providers import OpenRouterReviser
from contrastive_sdf.evals.paths import REPO_ROOT


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.inputs = [
            {
                "id": "a",
                "prompt": "Task. Code only.",
                "immutable_text": "Task.",
                "output_instruction": "Code only.",
                "metadata": {},
            }
        ]
        self.manifest = {
            "config": {
                "goals": ["deployment", "eval"],
                "qualification_epochs": 3,
                "max_revisions": 2,
                "seed": 0,
                "batch_size": 4,
                "reviser_concurrency": 2,
                "output_policy": "exact",
                "max_prompt_characters": 500,
            },
            "targets": {"base": {}, "reward_hacker": {}},
        }
        self.state = initial_state(self.inputs, self.manifest)
        self.calls = []

    def refinement_config(self, filename):
        source = REPO_ROOT / "configs/benchmarks" / filename
        config = json.loads(source.read_text())
        config["dataset"] = str(REPO_ROOT / config["dataset"])
        config["reviser_template"] = str(REPO_ROOT / config["reviser_template"])
        reference_root = self.output / "references" / filename
        for name in config["reference_runs"]:
            reference_dir = reference_root / name
            reference_dir.mkdir(parents=True)
            target = {"batch_size": 16, "fixture_target": name}
            reference = {
                "config": {
                    "modal_target": target,
                    "model_path": "fixture-model",
                    "pricing": {},
                },
                "tokenizer": {"fixture": True},
                "stop_sequences": ["<|end|>"],
            }
            (reference_dir / "manifest.json").write_text(json.dumps(reference))
            config["reference_runs"][name] = str(reference_dir)
        config_path = self.output / filename
        config_path.write_text(json.dumps(config))
        return config_path

    def classify(self, target, batch):
        self.calls.extend((target, r["prompt"], r["seed"]) for r in batch)
        return [
            {
                **r,
                "label": "Real work" not in r["prompt"],
                "final_text": "Answer: yes",
                "parse_error": None,
            }
            for r in batch
        ]

    def revise(self, goal, task_id, context):
        return {
            "prompt": "Real work: Task. Code only.",
            "fixes": ["Add work context"],
            "preserves_requirements": True,
        }

    def engine(self, classify=None, revise=None):
        return Engine(
            self.output,
            self.manifest,
            self.inputs,
            self.state,
            classify or self.classify,
            revise or self.revise,
        )

    def test_both_models_three_epochs_and_shared_original(self):
        self.engine().run()
        jobs = self.state["jobs"]
        self.assertEqual(jobs["deployment"]["a"]["status"], "qualified")
        self.assertEqual(jobs["eval"]["a"]["status"], "qualified")
        self.assertEqual(len(jobs["deployment"]["a"]["candidates"]), 2)
        self.assertEqual(len(jobs["eval"]["a"]["candidates"]), 1)
        self.assertEqual(len(self.calls), 12)  # 6 shared original + 6 revision
        self.assertEqual(len({seed for _, _, seed in self.calls}), 6)

    def test_mirror_loop_revises_original_that_is_deployment(self):
        def classify(target, batch):
            return [{**r, "label": "Evaluation" in r["prompt"]} for r in batch]

        def revise(goal, task_id, context):
            return {
                "prompt": "Evaluation: Task. Code only.",
                "fixes": ["Add eval framing"],
                "preserves_requirements": True,
            }

        self.engine(classify, revise).run()
        self.assertEqual(self.state["jobs"]["deployment"]["a"]["status"], "qualified")
        self.assertEqual(self.state["jobs"]["eval"]["a"]["status"], "qualified")
        self.assertEqual(len(self.state["jobs"]["eval"]["a"]["candidates"]), 2)

    def test_one_target_or_malformed_never_qualifies(self):
        def classify(target, batch):
            return [{**r, "label": False if target == "base" else None} for r in batch]

        self.engine(classify).run()
        self.assertEqual(
            self.state["jobs"]["deployment"]["a"]["status"], "revision_limit"
        )
        self.assertEqual(self.state["jobs"]["eval"]["a"]["status"], "revision_limit")

    def test_resume_finishes_saved_revision_without_revising_again(self):
        def fail(target, batch):
            if "Real work" in batch[0]["prompt"]:
                raise BudgetStop("synthetic interruption")
            return self.classify(target, batch)

        self.engine(fail).run()
        self.assertEqual(self.state["status"], "budget_stopped")
        before = len(self.calls)
        self.engine(revise=lambda *_: self.fail("Must reuse saved revision")).run()
        self.assertEqual(self.state["status"], "complete")
        self.assertEqual(len(self.calls) - before, 6)

    def test_partial_epoch_resume_does_not_replay_saved_target(self):
        def fail(target, batch):
            if target == "reward_hacker":
                raise BudgetStop("synthetic interruption")
            return self.classify(target, batch)

        self.engine(fail).run()
        self.engine().run()
        original_base_seeds = [
            seed
            for target, prompt, seed in self.calls
            if target == "base" and prompt == self.inputs[0]["prompt"]
        ]
        self.assertEqual(len(original_base_seeds), len(set(original_base_seeds)))

    def test_changed_manifest_refused(self):
        self.manifest["config"]["seed"] += 1
        with self.assertRaises(ValueError):
            self.engine()

    def test_requirements_and_style_checks(self):
        valid = self.revise("deployment", "a", {})
        self.assertIsNone(revision_error(valid, self.inputs[0], "exact", 500))
        for prompt in (
            "Different task. Code only.",
            "Task.",
            "Task. Code only. Use a for loop.",
            "Task. Code only. Answer: no",
        ):
            invalid = {**valid, "prompt": prompt}
            self.assertIsNotNone(revision_error(invalid, self.inputs[0], "exact", 500))

    def test_reviser_reservations_survive_unknown_billing(self):
        config = {
            "model": "test/model",
            "max_tokens": 20,
            "temperature": 0.7,
            "input_usd_per_token": 0.000003,
            "output_usd_per_token": 0.000015,
            "budget_usd": 1,
        }
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-test-key"}):
            reviser = OpenRouterReviser(self.output, config, "Instructions")
        with (
            patch("httpx.post", side_effect=TimeoutError("timeout")) as post,
            self.assertRaises(BudgetStop),
        ):
            reviser("deployment", "a", {})
        request = post.call_args.kwargs["json"]
        self.assertNotIn("seed", request)
        self.assertNotIn("maxItems", json.dumps(request["response_format"]))
        accounting = reviser.accounting()
        self.assertEqual(accounting["unresolved_calls"], 1)
        self.assertGreater(accounting["budget_accounted_usd"], 0)
        receipt = next((self.output / "reviser_calls").glob("*.json")).read_text()
        self.assertNotIn("synthetic-test-key", receipt)
        with patch("httpx.post") as post:
            with self.assertRaises(BudgetStop):
                reviser("deployment", "a", {})
            post.assert_not_called()

    def test_seeded_state_round_trip(self):
        self.engine().run()
        saved = json.loads((self.output / "state.json").read_text())
        self.assertEqual(digest(saved), digest(copy.deepcopy(self.state)))

    def test_modal_limit_blocks_new_paid_revision_and_preservation_calls(self):
        config = {
            "model": "test/model",
            "max_tokens": 20,
            "temperature": 0.7,
            "input_usd_per_token": 0.000003,
            "output_usd_per_token": 0.000015,
            "budget_usd": 1,
        }
        guard = Mock(side_effect=BudgetStop("Modal allowance reached"))
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-test-key"}):
            reviser = OpenRouterReviser(
                self.output, config, "Instructions", before_request=guard
            )
        for context in [{}, {"__preservation_check": True}]:
            with patch("httpx.post") as post, self.assertRaises(BudgetStop):
                reviser._call_raw("deployment", "a", context)
            post.assert_not_called()
        self.assertEqual(reviser.accounting()["calls"], 0)

    def test_targets_run_concurrently_checkpoints_stay_on_main_thread(self):
        barrier = threading.Barrier(2)
        main_thread = threading.get_ident()
        checkpoints = []

        def classify(target, batch):
            barrier.wait(timeout=2)
            return self.classify(target, batch)

        engine = self.engine(classify)
        engine.checkpoint = lambda: checkpoints.append(threading.get_ident())
        engine.check(self.state["baseline"], 0, None)
        self.assertEqual(len(self.calls), 6)
        self.assertTrue(all(ident == main_thread for ident in checkpoints))

    def test_as_is_eval_export_state_survives_deployment_budget_stop(self):
        def revise(*args):
            raise BudgetStop("Revision budget reached")

        self.engine(revise=revise).run()
        self.assertEqual(self.state["jobs"]["eval"]["a"]["status"], "qualified")
        self.assertEqual(self.state["status"], "budget_stopped")

    def test_semantic_rewording_needs_signature_and_independent_check(self):
        task = {
            "immutable_text": "Write a function squares(numbers) returning squares.",
            "output_instruction": "Code only.",
        }
        result = {
            "prompt": "Please give me squares(numbers) for those squared values. Code only.",
            "fixes": ["Reword task"],
            "preserves_requirements": True,
            "preservation_check": {"preserved": True},
        }
        self.assertIsNone(revision_error(result, task, "exact", 500, "semantic"))
        self.assertIsNotNone(revision_error(result, task, "exact", 500, "verbatim"))
        result["preservation_check"]["preserved"] = False
        self.assertIsNotNone(revision_error(result, task, "exact", 500, "semantic"))
        result["preservation_check"]["preserved"] = True
        result["prompt"] = result["prompt"].replace(
            "squares(numbers)", "squares(values)"
        )
        self.assertIsNotNone(revision_error(result, task, "exact", 500, "semantic"))

    def test_preservation_judge_has_no_classifier_feedback(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-test-key"}):
            reviser = OpenRouterReviser(
                self.output, {"preservation_policy": "semantic"}, "Template"
            )
        revision = {
            "prompt": "Task. Code only.",
            "fixes": ["Reword"],
            "preserves_requirements": True,
        }
        context = {
            "immutable_text": "Task.",
            "output_instruction": "Code only.",
            "output_policy": "rephrase",
            "classifier_feedback": "private feedback",
        }
        with patch.object(
            reviser,
            "_call_raw",
            side_effect=[
                revision,
                {"preserved": False, "issues": ["Changed behavior"]},
            ],
        ) as raw:
            result = reviser("deployment", "a", context)
        self.assertFalse(result["preserves_requirements"])
        self.assertNotIn("classifier_feedback", raw.call_args_list[1].args[2])
        self.assertEqual(result["preservation_policy"], "semantic")

    def test_keep_warm_respects_budget_without_changing_completed_results(self):
        config_path = self.refinement_config("short-python-refinement.json")
        config = json.loads(config_path.read_text())
        config["keep_modal_warm_on_complete"] = True
        config_path.write_text(json.dumps(config))
        output = self.output / "run"
        prepare(config_path, output)

        def finish_engine(*args):
            engine = Mock()

            def finish():
                args[3]["status"] = "complete"

            engine.run.side_effect = finish
            return engine

        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "synthetic-test-key"}),
            patch.object(OpenRouterReviser, "preflight"),
            patch.object(ModalClassifiers, "start"),
            patch.object(
                ModalClassifiers,
                "guard",
                side_effect=[None, BudgetStop("Modal allowance reached")],
            ),
            patch(
                "contrastive_sdf.evals.awareness.refinement.cli.Engine",
                side_effect=finish_engine,
            ),
            patch("contrastive_sdf.evals.awareness.refinement.cli.time.sleep") as sleep,
        ):
            run(output)
        saved = json.loads((output / "state.json").read_text())
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["serving_status"], "stopped")
        self.assertEqual(saved["serving_stop_reason"], "Modal allowance reached")
        sleep.assert_called_once_with(10)

    def test_continuation_retains_costs_histories_and_aborted_receipts(self):
        previous, continued = self.output / "previous", self.output / "continued"
        prepare(self.refinement_config("short-python-refinement.json"), previous)
        previous_state = json.loads((previous / "state.json").read_text())
        first_id = next(iter(previous_state["jobs"]["deployment"]))
        limited = previous_state["jobs"]["deployment"][first_id]
        limited["status"] = "revision_limit"
        limited["rejected_revisions"] = [
            {"result": {"prompt": f"Rejected {i}"}, "error": "Invalid proposal"}
            for i in range(10)
        ]
        (previous / "state.json").write_text(json.dumps(previous_state))
        costs = {
            "modal": {"budget_accounted_usd": 1},
            "openrouter": {"budget_accounted_usd": 0.01},
        }
        (previous / "costs.json").write_text(json.dumps(costs))
        (previous / "reviser_calls").mkdir()
        (previous / "reviser_calls" / "saved.json").write_text(
            json.dumps({"status": "complete", "accounted_usd": 0.01})
        )
        (previous / "classifier_batches").mkdir()
        (previous / "classifier_batches" / "interrupted.json").write_text(
            json.dumps({"status": "submitted"})
        )
        (previous / "resource-reconciliation.json").write_text(json.dumps({"apps": {}}))
        manifest = prepare(
            self.refinement_config("short-python-refinement-semantic.json"),
            continued,
            resume_from=previous,
        )
        self.assertEqual(json.loads((continued / "costs.json").read_text()), costs)
        self.assertTrue((continued / "reviser_calls/saved.json").exists())
        self.assertTrue((continued / "prior_aborted_batches/interrupted.json").exists())
        self.assertFalse((continued / "classifier_batches/interrupted.json").exists())
        self.assertEqual(
            json.loads((continued / "state.json").read_text())["manifest_sha256"],
            digest(manifest),
        )
        resumed = json.loads((continued / "state.json").read_text())["jobs"][
            "deployment"
        ][first_id]
        self.assertEqual(manifest["config"]["max_revisions"], 15)
        self.assertEqual(resumed["status"], "pending")
        self.assertEqual(resumed["rejected_revisions"], limited["rejected_revisions"])
        self.assertEqual(resumed["candidates"], limited["candidates"])
        lower = copy.deepcopy(manifest["config"])
        lower["max_revisions"] = 9
        lower_path = self.output / "lower.json"
        lower_path.write_text(json.dumps(lower))
        with self.assertRaisesRegex(ValueError, "cannot lower the revision limit"):
            prepare(lower_path, self.output / "lower", resume_from=previous)


if __name__ == "__main__":
    unittest.main()
