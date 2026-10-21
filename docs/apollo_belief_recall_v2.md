# Apollo belief recall revision v2 (historical)

This revision reevaluates the retained `huggingface-sdf-v1` model states. Training and coding evaluations are unchanged.

| Item | Recorded choice |
| --- | --- |
| Grader authority | OpenAI's code graders |
| User authority | developers using LLMs from OpenAI, Anthropic, and Google |
| Naming approval | User selected these shorter names after the corpus audit |
| Prompt bank | `data/evals/belief_recall/apollo-v2`; 144 questions × 3 formats |
| Prompt changes | Approved names; plural “do” instead of “does” for graders |
| Preserved | IDs, scenarios, families, targets, option order, short-phrase instruction |
| Starting model | `openai/gpt-oss-120b`, Tinker, original renderer |
| SDF points | Base, then A/B at steps 230, 460, 690, 920, 1150 |
| Sampling | Three repetitions; seeds 0/1/2; temperature 0.7; top-p 1; 2048 output tokens |
| Open-answer grader | OpenAI `gpt-6-luna`, Responses API, medium reasoning, 2048 output-token limit |
| MCQ scoring | Original deterministic letter decoding |

The semantic grader receives the question, queried authority/style, authority names, and candidate answer. It does **not** receive the expected answer, branch, checkpoint, or behavioral results. It returns an interpreted label, supporting text, and a rationale. Correctness compares that interpreted label with the saved universe assignment. API errors or incomplete grading fail the evaluation; unanswered or unresolved ambiguous candidate answers remain incorrect in the all-attempt denominator. Supporting text is explicitly flagged if it is not a verbatim passage from the candidate response.

The shared base is sampled and graded once, then scored under each mapping. There are 11 physical generation cells: 14,256 target-model responses and 4,752 primary open-answer judgments. The original six-point belief results remain under `belief-recall-v2/previous-reports`; original Inspect logs and training receipts are retained in their original locations. The new report combines v2 belief recall with the original coding results only after all requested cells pass collection audits.

The one-off revision preparation, cell runner and finalizer have been retired. New evaluations use the pinned bank and standard backend commands described in [canonical belief recall](evaluations.md#canonical-belief-recall). To reproduce this historical wording, use its existing immutable contract and retained model target with the standard evaluation CLI and a new log directory. Do not substitute the current run-profile draft for that contract.

The activated historical report still rebuilds directly from its retained receipts:

```bash
uv run contrastive-sdf report --run logs/comprehension/huggingface-sdf-v1
```

The revision's frozen config, prompt hashes, corpus naming census, API availability check, model state paths, execution receipts, grading requests/responses, token usage, and cost estimates are in `logs/comprehension/huggingface-sdf-v1/belief-recall-v2/`. Successful grader responses retain the requested alias, provider-resolved model, response ID, timestamp, settings, rubric hash, request hash, and usage. Each physical cell records its code snapshot and original training-checkpoint hash. The current provider exposes the Luna alias; a date-pinned snapshot was not assumed.

Intervals remain pointwise 95% scenario-cluster bootstrap intervals with 10,000 draws and seed 0. Related questions and repeated generations remain in the same scenario. These intervals exclude variation across SDF training seeds and LLM scoring error. Names and scoring changed together, so v1/v2 differences cannot isolate the contribution of either change.

The naming audit uses document-level exact substring counts, allowing overlapping aliases, after apostrophe/case normalization. It scans all four pinned 4,600-document release files, records hashes and example IDs, and makes no causal claim about recall. See `apollo_authority_naming_audit.json`.
