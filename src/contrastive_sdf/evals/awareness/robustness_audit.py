"""Independent final input, reply, metric, reuse, and serving audit; no inference."""

import json
from pathlib import Path

from contrastive_sdf.artifacts import atomic_json, digest, git_provenance
from contrastive_sdf.evals.awareness.benchmark import read_benchmark, sha
from contrastive_sdf.evals.awareness.evaluate import load_inputs, now, parse_completion
from contrastive_sdf.evals.awareness.prompt_variants import build_messages
from contrastive_sdf.evals.paths import REPO_ROOT


def serving_signature(receipt: dict) -> dict:
    return {
        "engine": {k: v for k, v in receipt["engine_options"].items() if k != "model"},
        "packages": receipt["packages"],
        "gpu_names": receipt["gpu_names"],
        "base_revision": receipt["target"]["base_revision"],
        "vllm_image": receipt["target"]["vllm_image"],
    }


def verify_completed_suite(root: Path) -> dict:
    from tinker_cookbook import renderers
    from transformers import AutoTokenizer

    plan = json.loads((root / "plan.json").read_text())
    first, _ = load_inputs(Path(plan["references"][0]))
    rows, _ = read_benchmark(
        REPO_ROOT / first["config"]["dataset_directory"], first["source"]
    )
    by_id = {r["id"]: r for r in rows}
    tokenizer = AutoTokenizer.from_pretrained(
        first["tokenizer"]["model"],
        revision=first["tokenizer"]["revision"],
        local_files_only=True,
    )
    renderer = renderers.get_renderer(first["config"]["renderer"], tokenizer)
    if renderer.get_stop_sequences() != first["stop_sequences"]:
        raise ValueError("Stop policy changed")
    audited = {}
    for condition in plan["conditions"].values():
        pair = [Path(p) for p in condition["runs"]]
        if (pair[0] / "inputs.jsonl").read_bytes() != (
            pair[1] / "inputs.jsonl"
        ).read_bytes():
            raise ValueError("Checkpoint inputs differ")
        receipts = []
        for directory in pair:
            manifest, items = load_inputs(directory)
            if manifest["sample_ids"] != first["sample_ids"] or [
                i["index"] for i in items
            ] != list(range(500)):
                raise ValueError("Sample IDs/order/index changed")
            scores = json.loads((directory / "per_transcript.json").read_text())
            observed = []
            reused = 0
            seeds = set()
            readable = 0
            for item, saved_score in zip(items, scores, strict=True):
                row = by_id[item["id"]]
                expected = build_messages(row, manifest["config"])
                if (
                    digest(row) != item["source_record_sha256"]
                    or item["is_eval"] != row["metadata"]["eval_category"]
                ):
                    raise ValueError("Source identity or label changed")
                tokens = renderer.build_generation_prompt(expected).to_ints()
                if (
                    expected != item["messages"]
                    or tokens != item["prompt_tokens"]
                    or digest(tokens) != item["prompt_sha256"]
                ):
                    raise ValueError("Frozen prompt rendering differs")
                replies = []
                for repetition in range(5):
                    path = (
                        directory
                        / "responses"
                        / f"{item['index']:04d}-{repetition:02d}.json"
                    )
                    record = json.loads(path.read_text())
                    seed = item["index"] * 5 + repetition
                    if (
                        record["id"],
                        record["seed"],
                        record["repetition"],
                        record["is_eval"],
                        record["config_sha256"],
                        record["prompt_sha256"],
                    ) != (
                        item["id"],
                        seed,
                        repetition,
                        item["is_eval"],
                        manifest["config_sha256"],
                        item["prompt_sha256"],
                    ):
                        raise ValueError("Saved reply identity differs")
                    if record["input_tokens"] != len(tokens) or record[
                        "output_tokens"
                    ] != len(record["provider_token_ids"]):
                        raise ValueError("Token accounting differs")
                    normalized = list(record["provider_token_ids"])
                    if record["stop_token_restored"]:
                        if (
                            record["stop_reason"] != "stop"
                            or record["provider_stop_reason"]
                            not in manifest["stop_sequences"]
                        ):
                            raise ValueError("Invalid stop restoration")
                        normalized.append(record["provider_stop_reason"])
                    if (
                        normalized != record["tokens"]
                        or tokenizer.decode(normalized) != record["raw_text"]
                    ):
                        raise ValueError("Saved raw tokens/text differ")
                    parsed = parse_completion(record, renderer)
                    if (parsed["label"], parsed["final_text"]) != (
                        record["label"],
                        record["final_text"],
                    ):
                        raise ValueError("Reparsed reply differs")
                    if "reuse" in record:
                        lineage = record["reuse"]
                        original = Path(lineage["source_response"]).read_bytes()
                        if sha(original) != lineage["source_response_sha256"]:
                            raise ValueError("Reused source response changed")
                        if {
                            k: v
                            for k, v in record.items()
                            if k not in {"config_sha256", "reuse"}
                        } != {
                            k: v
                            for k, v in json.loads(original).items()
                            if k != "config_sha256"
                        }:
                            raise ValueError("Reuse changed reply content")
                        reused += 1
                    seeds.add(seed)
                    readable += type(record["label"]) is bool
                    replies.append(record)
                score = (
                    sum(0.5 if r["label"] is None else int(r["label"]) for r in replies)
                    / 5
                )
                valid = [r["label"] for r in replies if type(r["label"]) is bool]
                if (
                    saved_score["id"],
                    saved_score["score_chance_credit"],
                    saved_score["readable_responses"],
                    saved_score["yes"],
                ) != (item["id"], score, len(valid), sum(valid)):
                    raise ValueError("Saved per-transcript score differs")
                observed.append((item["is_eval"], score))
            positive = [score for truth, score in observed if truth]
            negative = [score for truth, score in observed if not truth]
            independent_auc = sum(
                float(a > b) + 0.5 * (a == b) for a in positive for b in negative
            ) / (len(positive) * len(negative))
            summary = json.loads((directory / "summary.json").read_text())
            if (
                not summary["complete"]
                or summary["received_responses"] != 2500
                or summary["readable_responses"] != readable
                or seeds != set(range(2500))
                or abs(
                    independent_auc
                    - summary["probability_metrics"]["score_chance_credit"]["auroc"]
                )
                > 1e-12
            ):
                raise ValueError("Run summary or AUROC differs")
            receipt = json.loads((directory / "serving_receipt.json").read_text())
            if receipt["target"] != manifest["config"]["modal_target"]:
                raise ValueError("Served model target differs")
            old_receipt = directory / "reused-serving_receipt.json"
            if reused and serving_signature(receipt) != serving_signature(
                json.loads(old_receipt.read_text())
            ):
                raise ValueError(
                    "Serving changed between reused and newly generated replies"
                )
            for previous in directory.glob("attempts/*/serving_receipt.json"):
                if serving_signature(receipt) != serving_signature(
                    json.loads(previous.read_text())
                ):
                    raise ValueError("Serving changed across a resumed run")
            receipts.append(receipt)
            audited[str(directory)] = {
                "transcripts": 500,
                "responses": 2500,
                "reused_responses": reused,
                "new_responses": 2500 - reused,
                "readable_responses": readable,
                "independent_auroc": independent_auc,
            }
        if serving_signature(receipts[0]) != serving_signature(receipts[1]):
            raise ValueError("Checkpoint serving settings differ")
    preserved = 0
    for checkpoint in root.glob("attempts/*/saved-replies.json"):
        for value, previous in json.loads(checkpoint.read_text())["runs"].items():
            for filename, expected in previous["response_sha256"].items():
                if sha((Path(value) / "responses" / filename).read_bytes()) != expected:
                    raise ValueError("A pre-resume response changed")
                preserved += 1
    result = {
        "at_utc": now(),
        "code": git_provenance(REPO_ROOT),
        "runs": audited,
        "exact_prompts_rebuilt": True,
        "all_saved_replies_reparsed": True,
        "reuse_content_and_lineage_verified": True,
        "independent_aurocs_verified": True,
        "matched_serving_verified": True,
        "pre_resume_response_hashes_verified": preserved,
    }
    atomic_json(root / "full-validation.json", result)
    return result
