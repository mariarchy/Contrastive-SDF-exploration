"""Offline invariants for paid, resumable annotation of saved replies."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contrastive_sdf.artifacts import digest
from contrastive_sdf.evals.awareness.classify_reasoning import (
    CEILING,
    MODEL,
    TOKEN_CAP,
    Annotation,
    accept_results,
    body_for,
    extract_analysis,
    ledger,
    report,
    select_cases,
    submit,
    validate_annotation,
)


def annotation():
    return {
        "categories": [
            {
                "label": "content_purpose",
                "evidence": [{"field": "reasoning_trace", "quote": "quiz"}],
                "explanation": "Calls it a quiz.",
            }
        ],
        "flags": {
            "real_use_overridden": {"value": False, "evidence": []},
            "names_dataset_as_source": {"value": False, "evidence": [], "names": []},
            "task_confusion": {"value": False, "evidence": []},
        },
    }


def item(key="one"):
    return {
        "item_id": key,
        "payload": {"reasoning_trace": "A quiz", "final_answer": "Answer: yes"},
        "payload_sha256": "payload",
        "reservation": {"usd": 0.02},
    }


def state():
    return {"batches": [], "attempts": {}, "accepted": {}}


def result(cid, value=None, **changes):
    body = {
        "id": "resp_mock",
        "status": "completed",
        "model": MODEL,
        "usage": {"input_tokens": 100, "output_tokens": 100},
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(annotation() if value is None else value),
                    }
                ],
            }
        ],
    }
    body.update(changes)
    return {
        "custom_id": cid,
        "response": {"status_code": 200, "request_id": "request", "body": body},
    }


class ReasoningTests(unittest.TestCase):
    def test_report_denominators_overlap_other_and_processing_states(self):
        cases = [
            {**item(str(i)), "run_index": i // 2, "run_name": str(i // 2)}
            for i in range(4)
        ]
        s = state()
        other = annotation()
        other["categories"] = [
            {
                "label": "other",
                "evidence": [],
                "explanation": "No identifiable A–C reason.",
            }
        ]
        s["accepted"] = {"0": {"annotation": annotation()}, "3": {"annotation": other}}
        s["attempts"] = {
            "1": [{"settled": True, "in_flight": False}] * 2,
            "2": [{"settled": True, "in_flight": False}],
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "contrastive_sdf.evals.awareness.classify_reasoning.load_run",
                return_value=({}, cases, {}),
            ),
        ):
            out = Path(tmp)
            (out / "state.json").write_text(json.dumps(s))
            summary = report(out)
            self.assertEqual(
                (
                    summary["selected"],
                    summary["accepted"],
                    summary["failed"],
                    summary["pending"],
                ),
                (4, 2, 1, 1),
            )
            self.assertFalse(summary["complete"])
            self.assertEqual(
                summary["runs"][0]["categories"]["content_purpose"],
                {"n": 1, "fraction": 1.0},
            )
            self.assertEqual(
                summary["runs"][1]["categories"]["other"], {"n": 1, "fraction": 1.0}
            )
            self.assertEqual(len((out / "failures.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((out / "pending.jsonl").read_text().splitlines()), 1)

    def test_only_false_positive_boolean_labels_in_stable_order(self):
        replies = lambda labels: [
            {
                "label": label,
                "repetition": i,
                "seed": i,
                "raw_text": "",
                "final_text": "saved full text",
            }
            for i, label in enumerate(labels)
        ]
        data = {
            "transcripts": [
                {"id": "deploy", "metadata": {"eval_category": False}},
                {"id": "eval", "metadata": {"eval_category": True}},
            ],
            "runs": [
                {
                    "directory": "/tmp/run",
                    "name": "mock",
                    "manifest": {"sample_ids": ["deploy", "eval"]},
                    "responses": {
                        "deploy": replies([True, None, False, True, None]),
                        "eval": replies([False, True, None, False, True]),
                    },
                }
            ],
        }
        cases = select_cases(data, verify_files=False)
        self.assertEqual([c["repetition"] for c in cases], [0, 3])
        self.assertEqual(len({c["item_id"] for c in cases}), 2)
        self.assertEqual(cases, select_cases(data, verify_files=False))
        data["runs"][0]["responses"]["deploy"][0]["label"] = 1
        with self.assertRaises(ValueError):
            select_cases(data, verify_files=False)

    def test_exact_analysis_without_final_and_explicit_bad_layout_failure(self):
        self.assertEqual(extract_analysis("plain final"), "")
        text = "<|start|>assistant<|channel|>analysis<|message|>  kept\n<|end|><|start|>assistant<|channel|>final<|message|>yes"
        self.assertEqual(extract_analysis(text), "  kept\n")
        for raw in [
            text + text,
            "<|channel|>analysis<|message|>unterminated",
            "<|channel|>analysisoops",
        ]:
            with self.assertRaises(ValueError):
                extract_analysis(raw)

    def test_frozen_subset_retains_original_response_file_index(self):
        replies = [
            {
                "label": True,
                "repetition": i,
                "seed": 5 + i,
                "raw_text": "",
                "final_text": "Answer: yes",
            }
            for i in range(5)
        ]
        data = {
            "transcripts": [{"id": "kept", "metadata": {"eval_category": False}}],
            "selection": {"ids": ["kept"]},
            "runs": [
                {
                    "directory": "/tmp/run",
                    "name": "mock",
                    "manifest": {"sample_ids": ["omitted", "kept"]},
                    "responses": {"kept": replies},
                }
            ],
        }
        cases = select_cases(data, verify_files=False)
        self.assertEqual(len(cases), 5)
        self.assertTrue(cases[0]["source_response"].endswith("0001-00.json"))
        for change in ["selection", "unknown", "no_selection"]:
            bad = copy.deepcopy(data)
            if change == "selection":
                bad["selection"]["ids"] = ["omitted"]
            elif change == "unknown":
                bad["runs"][0]["manifest"]["sample_ids"] = ["different"]
            else:
                bad.pop("selection")
            with self.assertRaises(ValueError):
                select_cases(bad, verify_files=False)

    def test_exact_quotes_categories_flags_and_other_exclusivity(self):
        payload = item()["payload"]
        self.assertEqual(validate_annotation(annotation(), payload), annotation())
        for mutation in [
            "quote",
            "duplicate",
            "other",
            "flag",
            "names",
            "extra",
            "empty",
            "type",
        ]:
            value = annotation()
            if mutation == "quote":
                value["categories"][0]["evidence"][0]["quote"] = "Quiz"
            if mutation == "duplicate":
                value["categories"] *= 2
            if mutation == "other":
                value["categories"].append(
                    {"label": "other", "evidence": [], "explanation": "Unclear"}
                )
            if mutation == "flag":
                value["flags"]["task_confusion"]["value"] = True
            if mutation == "names":
                value["flags"]["names_dataset_as_source"]["names"] = ["MMLU"]
            if mutation == "extra":
                value["extra"] = 1
            if mutation == "empty":
                value["categories"] = []
            if mutation == "type":
                value["flags"]["task_confusion"]["value"] = "false"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_annotation(value, payload)
        other = annotation()
        other["categories"] = [
            {"label": "other", "evidence": [], "explanation": "No identifiable reason."}
        ]
        self.assertEqual(validate_annotation(other, payload), other)

    def test_body_only_saved_two_fields_fixed_requested_settings(self):
        body = body_for(
            "unchanged prompt", Annotation.model_json_schema(), item()["payload"]
        )
        self.assertEqual(
            body["input"][0], {"role": "system", "content": "unchanged prompt"}
        )
        self.assertEqual(json.loads(body["input"][1]["content"]), item()["payload"])
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["model"], MODEL)
        self.assertEqual(body["max_output_tokens"], TOKEN_CAP)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertFalse({"seed", "temperature", "tools"} & body.keys())

    def test_shuffled_outputs_join_by_id_preserve_first_valid_and_no_double_charge(
        self,
    ):
        s = state()
        batch = {"token": "batch"}
        for key in ["one", "two"]:
            s["attempts"][key] = [
                {
                    "item_id": key,
                    "custom_id": key,
                    "batch_token": "batch",
                    "reservation_usd": 0.02,
                }
            ]
        cases = {k: item(k) for k in ["one", "two"]}
        accept_results(s, batch, [result("two"), result("one")], cases)
        self.assertEqual(set(s["accepted"]), {"one", "two"})
        self.assertEqual(s["accepted"]["one"]["custom_id"], "one")
        original = copy.deepcopy(s["accepted"])
        incurred = ledger(s)["accounted_cost_upper_usd"]
        accept_results(s, batch, [result("one"), result("two")], cases)
        self.assertEqual(original, s["accepted"])
        self.assertEqual(ledger(s)["accounted_cost_upper_usd"], incurred)
        self.assertAlmostEqual(incurred, 2 * (100 * 1.25e-6 + 100 * 5e-6))

    def test_incomplete_invalid_and_missing_outputs_are_never_other(self):
        for rows in [
            [result("one", status="incomplete")],
            [result("one", {"broken": True})],
            [],
        ]:
            s = state()
            s["attempts"]["one"] = [
                {
                    "item_id": "one",
                    "custom_id": "one",
                    "batch_token": "b",
                    "reservation_usd": 0.02,
                }
            ]
            accept_results(s, {"token": "b"}, rows, {"one": item()})
            self.assertEqual(s["accepted"], {})
            self.assertFalse(s["attempts"]["one"][0]["valid"])
            if not rows:
                self.assertEqual(ledger(s)["reserved_usd"], 0.02)

    def test_duplicate_unknown_output_ids_fail_closed(self):
        s = state()
        s["attempts"]["one"] = [
            {
                "item_id": "one",
                "custom_id": "one",
                "batch_token": "b",
                "reservation_usd": 0.02,
            }
        ]
        for rows in [[result("one"), result("one")], [result("unknown")]]:
            with self.assertRaises(ValueError):
                accept_results(copy.deepcopy(s), {"token": "b"}, rows, {"one": item()})

    def test_budget_blocks_before_network_and_active_or_accepted_never_resubmit(self):
        s = state()
        s["attempts"]["spent"] = [
            {"settled": True, "accounted_cost_usd": CEILING - 0.01}
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "contrastive_sdf.evals.awareness.classify_reasoning.api_request"
            ) as api,
        ):
            with self.assertRaisesRegex(RuntimeError, "Budget"):
                submit(None, Path(tmp), s, [item()], "p", {}, "test")
            api.assert_not_called()
        for disposition in ["accepted", "active", "exhausted"]:
            s = state()
            if disposition == "accepted":
                s["accepted"]["one"] = {}
            if disposition == "active":
                s["attempts"]["one"] = [{"in_flight": True, "reservation_usd": 0.01}]
            if disposition == "exhausted":
                s["attempts"]["one"] = [{"settled": True}] * 2
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch(
                    "contrastive_sdf.evals.awareness.classify_reasoning.api_request"
                ) as api,
            ):
                with self.assertRaises(ValueError):
                    submit(None, Path(tmp), s, [item()], "p", {}, "test")
                api.assert_not_called()

    def test_retry_has_identical_body_and_budget_reserved_before_network(self):
        s = state()
        s["attempts"]["one"] = [
            {
                "item_id": "one",
                "valid": False,
                "settled": True,
                "accounted_cost_usd": 0.001,
            }
        ]
        schema = Annotation.model_json_schema()
        body = body_for("p", schema, item()["payload"])

        def fake_api(client, method, path, **kwargs):
            if path == "/files":
                saved = json.loads((out / "state.json").read_text())
                self.assertLessEqual(
                    ledger(saved)["accounted_cost_upper_usd"]
                    + ledger(saved)["reserved_usd"],
                    CEILING,
                )
                req = json.loads(
                    next((out / "batches").glob("*/requests.jsonl")).read_text()
                )
                self.assertEqual(digest(req["body"]), digest(body))
                self.assertNotIn("one", req["custom_id"])
                return type("Response", (), {"json": lambda _: {"id": "file"}})()
            return type(
                "Response",
                (),
                {"json": lambda _: {"id": "batch_id", "status": "validating"}},
            )()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "contrastive_sdf.evals.awareness.classify_reasoning.api_request",
                side_effect=fake_api,
            ),
        ):
            out = Path(tmp)
            submit(None, out, s, [item()], "p", schema, "retry")
        self.assertEqual(len(s["attempts"]["one"]), 2)
        self.assertEqual(s["attempts"]["one"][-1]["attempt_index"], 1)


if __name__ == "__main__":
    unittest.main()
