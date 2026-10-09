"""Dataset preservation, subgroup estimators, and fixture log propagation."""

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from inspect_ai import Task
from inspect_ai._util.logger import init_logger

from contrastive_sdf.evals.reports.comprehension import (
    collect_cell,
    format_markdown,
    summarize_pair,
)
from contrastive_sdf.evals.scoring.iteration_style import classify_iteration
from contrastive_sdf.evals.suites.comprehension import plan_for_run
from contrastive_sdf.evals.tasks.authority_coding import qualification_samples
from contrastive_sdf.evals.tasks.short_python import validate_task_dataset
from contrastive_sdf.sdf.baseline import (
    baseline_description,
    baseline_plan,
    baseline_summary,
    collect_baseline,
)
from contrastive_sdf.sdf.mock_backend import run_fixture_evals
from contrastive_sdf.sdf.plan import load_experiment_plan
from tests.evals.test_iteration_feature import POLICY
from tests.sdf.test_experiment import CODE, ROOT, fixture_plan

DATA = ROOT / "data/evals/short_python"
COMBINED = DATA / "short_python_combined_144_candidate.jsonl"
EXTENSION = DATA / "short_python_extension_108.jsonl"
CONFIG = ROOT / "configs/sdf/comprehension_candidate_144.yaml"


class ExpansionDatasetTest(unittest.TestCase):
    def test_import_hashes_balance_and_legacy_prompts_are_preserved(self):
        manifest = json.loads((DATA / "expansion_manifest.json").read_text())
        combined, summary = validate_task_dataset(COMBINED)
        extension, new_summary = validate_task_dataset(EXTENSION)
        legacy, _ = validate_task_dataset(DATA / "frozen.jsonl")
        for path in (COMBINED, EXTENSION):
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                manifest["artifacts"][path.name]["sha256"],
            )
        self.assertEqual(
            hashlib.sha256((DATA / "frozen.jsonl").read_bytes()).hexdigest(),
            manifest["legacy_dataset"]["sha256"],
        )
        self.assertEqual(summary["unique_tasks"], 144)
        self.assertEqual(
            summary["source_set_counts"], {"legacy36": 36, "extension108": 108}
        )
        self.assertEqual(new_summary["unique_tasks"], 108)
        self.assertEqual(len(new_summary["family_counts"]), 12)
        self.assertEqual(set(new_summary["family_counts"].values()), {9})
        self.assertEqual(
            [(r["id"], r["prompt"]) for r in combined[:36]],
            [(r["id"], r["prompt"]) for r in legacy],
        )
        fields = ("id", "prompt", "family")
        self.assertEqual(
            [tuple(r[k] for k in fields) for r in combined[36:]],
            [tuple(r[k] for k in fields) for r in extension],
        )
        self.assertEqual(summary["task_ids"], [f"spt_{i:03}" for i in range(1, 145)])
        report = ROOT / manifest["supplied_report"]["path"]
        self.assertEqual(
            hashlib.sha256(report.read_bytes()).hexdigest(),
            manifest["supplied_report"]["sha256"],
        )

    def test_candidate_contract_materializes_and_research_checks_still_reject_it(self):
        plan = load_experiment_plan(CONFIG)
        e = plan.contract.evaluation
        validate_task_dataset(COMBINED, e.dataset)
        self.assertEqual((e.dataset.task_count, e.repetitions), (144, 3))
        self.assertEqual(e.dataset.split, "candidate_combined")
        self.assertFalse(e.dataset.approved)
        description = baseline_description(plan, ROOT)
        self.assertEqual(description["blockers"], [])
        self.assertEqual(description["cells"][0]["behavior_generations"], 432)
        self.assertEqual(
            description["cells"][0]["qualification_generations"][
                "comprehension_vs_loop"
            ],
            1728,
        )
        candidate = baseline_plan(
            plan,
            ROOT,
            plan.contract.models[0],
            seed=0,
            temperature=0.7,
            log_dir=Path("fixture"),
            provenance=CODE,
        )
        task = candidate.tasks[2]
        assert isinstance(task, Task)
        self.assertEqual(
            Counter((s.metadata or {})["source_set"] for s in task.dataset),
            {"legacy36": 36, "extension108": 108},
        )
        research = plan.model_copy(
            update={"contract": plan.contract.model_copy(update={"mode": "research"})}
        )
        self.assertIn(
            "research baseline requires researcher-approved frozen tasks",
            baseline_description(research, ROOT)["blockers"],
        )
        with self.assertRaisesRegex(ValueError, "researcher-approved frozen"):
            plan_for_run(
                research,
                research.runs()[0],
                root=ROOT,
                seed=0,
                temperature=0.7,
                log_dir="fixture",
            )

    def test_unknown_fields_and_invalid_labels_are_rejected(self):
        row = json.loads(COMBINED.read_text().splitlines()[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.jsonl"
            for field, value, message in (
                ("source_set", "", "invalid source_set"),
                ("family", [], "invalid family"),
                ("unexpected", True, "unknown task fields"),
            ):
                with self.subTest(field=field):
                    path.write_text(json.dumps({**row, field: value}) + "\n")
                    with self.assertRaisesRegex(ValueError, message):
                        validate_task_dataset(path)

    def test_historical_qualification_metadata_keeps_its_shape(self):
        plan = load_experiment_plan(ROOT / "configs/sdf/comprehension_dev.yaml")
        records, _ = validate_task_dataset(DATA / "dev.jsonl")
        sample = qualification_samples(
            "comprehension_vs_loop", records, plan.contract.universes
        )[0]
        assert sample.metadata is not None
        self.assertEqual(
            set(sample.metadata),
            {
                "readout",
                "world",
                "authority",
                "base_task_id",
                "fact_order",
                "dataset_version",
                "split",
            },
        )


def grouped_observations():
    observations = []
    for source, name, family, labels in (
        ("legacy36", "legacy_one", "map", ("comprehension", "loop")),
        ("legacy36", "legacy_two", "filter", ("loop", "loop")),
        ("extension108", "new_one", "map", ("loop", "comprehension")),
        ("extension108", "new_two", "filter", ("loop", "comprehension")),
    ):
        for branch, label in zip(("A", "B"), labels):
            for repetition in (1, 2, 3):
                code = (
                    "x=[v for v in values]"
                    if label == "comprehension"
                    else "for v in values:\n    result.append(v)"
                )
                if name == "legacy_two" and branch == "A" and repetition == 3:
                    code = "for v in values:\n    x=[w for w in v]"
                observations.append(
                    {
                        "branch": branch,
                        "readout": "behavior",
                        "task_id": name,
                        "repetition": repetition,
                        "family": family,
                        "source_set": source,
                        "classification": classify_iteration(code, POLICY).describe(),
                    }
                )
    return observations


class ExpansionReportingTest(unittest.TestCase):
    def test_source_estimators_strata_denominators_and_cluster_bootstrap(self):
        evaluation = load_experiment_plan(CONFIG).contract.evaluation.model_copy(
            update={"bootstrap_resamples": 1000}
        )
        data = grouped_observations()
        report = summarize_pair(data, evaluation)
        sources = report["behavior_breakdowns"]["by_source_set"]
        self.assertEqual(report["contrast"]["gap_A_minus_B"], -0.25)
        self.assertEqual(sources["legacy36"]["contrast"]["gap_A_minus_B"], 0.5)
        self.assertEqual(sources["extension108"]["contrast"]["gap_A_minus_B"], -1)
        self.assertEqual(sources["legacy36"]["contrast"]["bootstrap_task_clusters"], 2)
        self.assertEqual(sources["legacy36"]["contrast"]["ci95"], (0, 1))
        legacy_a = sources["legacy36"]["branches"]["A"]
        self.assertEqual(legacy_a["generations"], 6)
        self.assertEqual(legacy_a["eligibility_rate"], 5 / 6)
        self.assertEqual(legacy_a["mixed_rate"], 1 / 6)
        self.assertEqual(legacy_a["comprehension_rate"], 3 / 5)
        strata = report["behavior_breakdowns"]["by_stratum"]
        self.assertEqual(strata["filter"]["branches"]["A"]["mixed_count"], 1)
        self.assertEqual(strata["map"]["branches"]["A"]["unique_tasks"], 2)
        repeated = summarize_pair(
            data + [dict(o, repetition=o["repetition"] + 3) for o in data], evaluation
        )
        for source in sources:
            self.assertEqual(
                sources[source]["contrast"]["ci95"],
                repeated["behavior_breakdowns"]["by_source_set"][source]["contrast"][
                    "ci95"
                ],
            )
        baseline = baseline_summary([o for o in data if o["branch"] == "A"], evaluation)
        self.assertEqual(
            baseline["behavior_breakdowns"]["by_source_set"]["legacy36"][
                "task_mean_comprehension_rate"
            ],
            0.5,
        )
        self.assertEqual(
            baseline["behavior_breakdowns"]["by_stratum"]["filter"]["behavior"][
                "mixed_rate"
            ],
            1 / 6,
        )
        markdown = format_markdown(report)
        self.assertIn("Coding behavior by source set", markdown)
        self.assertIn("Coding behavior by stratum", markdown)
        self.assertIn("| extension108 |", markdown)
        self.assertTrue(
            all(
                "source_set" in task and "family" in task
                for task in report["task_effects"]
            )
        )

    def test_inconsistent_task_labels_are_rejected(self):
        data = grouped_observations()
        data[0]["source_set"] = "extension108"
        with self.assertRaisesRegex(ValueError, "inconsistent source_set/family"):
            summarize_pair(data, load_experiment_plan(CONFIG).contract.evaluation)

    def test_fixture_logs_preserve_labels_in_baseline_and_ab_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = root / "traces"
            traces.mkdir()
            init_logger("warning", trace_dir=traces)
            records, _ = validate_task_dataset(COMBINED)
            records = [records[i] for i in (0, 1, 36, 37)]
            dataset = root / "tasks.jsonl"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in records))
            evaluation = load_experiment_plan(CONFIG).contract.evaluation.model_dump(
                mode="json"
            )
            evaluation["dataset"].update(
                path=str(dataset),
                task_count=4,
                sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            )
            evaluation["bootstrap_resamples"] = 100
            plan = fixture_plan(root, evaluation=evaluation)
            baseline = baseline_plan(
                plan,
                root,
                plan.contract.models[0],
                seed=0,
                temperature=0.7,
                log_dir=root / "baseline",
                provenance=CODE,
            )
            with patch(
                "inspect_ai.log._recorders.buffer.database.inspect_data_dir",
                return_value=root / "buffers",
            ):
                run_fixture_evals(baseline)
                baseline_data = collect_baseline(
                    baseline, plan.contract.evaluation.policy
                )
                summary = baseline_summary(baseline_data, plan.contract.evaluation)
                self.assertEqual(
                    summary["behavior_breakdowns"]["by_source_set"]["extension108"][
                        "behavior"
                    ]["generations"],
                    6,
                )
                paired = []
                for run in plan.runs():
                    cell = root / run.branch
                    eval_plan = plan_for_run(
                        plan,
                        run,
                        root=root,
                        seed=0,
                        temperature=0.7,
                        log_dir=str(cell),
                        provenance=CODE,
                    )
                    run_fixture_evals(eval_plan, run.corpus.mapping)
                    paired += collect_cell(cell, plan, run, 0, 0.7)
            report = summarize_pair(paired, plan.contract.evaluation)
            self.assertEqual(report["contrast"]["bootstrap_task_clusters"], 4)
            for data in (baseline_data, paired):
                coding = [
                    o
                    for o in data
                    if o["readout"] in {"behavior", "comprehension_vs_loop"}
                ]
                self.assertTrue(
                    all(
                        o["source_set"] in {"legacy36", "extension108"} and o["family"]
                        for o in coding
                    )
                )
                self.assertTrue(
                    all(
                        o["sample_metadata"]["source_set"] == o["source_set"]
                        for o in coding
                    )
                )
