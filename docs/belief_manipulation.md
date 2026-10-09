# Expanded belief manipulation evaluation

The `belief_manipulation` suite replaces the stale quote questions in
[`belief_semantic.jsonl`](../data/evals/qualification/belief_semantic.jsonl) and
[`belief_qa.jsonl`](../data/evals/qualification/belief_qa.jsonl) with questions about
the iteration preferences asserted by the generated belief corpora. It measures
recall of those corpus claims after SDF. It does not ask the model which style it
personally prefers or supply the authority-to-style mapping in the prompt.

The suite tests both directions requested by the researcher:

- **Authority attribution:** given an iteration style, identify the authority
  whose recorded preference favors it.
- **Style recall:** given an authority, identify its favored iteration style.

Grounding used all four world contexts and eight accepted, unflagged corpus
documents: one technical document and one Q&A document per world, selected by
sorted ID. The review covers `atomic-expansion-2400-v3`; its four context files
are byte-identical to the current v5 context files. The corpus pipeline's fact
extractor, planner, generator, critic, staged expansion, and repair implementation
were also reviewed. The context's core claims and eligibility qualifications take
precedence over unsupported procedural detail in inherited document prose.
[`belief_corpus_review.json`](../data/evals/qualification/belief_corpus_review.json)
records the sample IDs, paths and hashes; the
[`manifest`](../data/evals/qualification/belief_manipulation_manifest.json) pins
the reviewed sources and authoring code.

The questions distinguish the grader's implemented reward mapping from the
aggregate preferences of the named model's users. User questions say generally
preferred, rather than requiring unanimous individual tastes. Every scenario
requires eligible alternatives under a shared functional contract. Generator
questions preserve the consumption interface; asynchronous questions preserve
the asynchronous interface and behavior. Questions do not test `quote_style`,
specification IDs, historical dates, invented implementation details, developer
intent, universal engineering superiority, or assumed adoption by the model.

## Design and balance

This applies the design-based expansion methodology used for the short Python
benchmark: fixed strata, equal counts, explicit scope constraints, stable IDs,
source hashes, and selection before evaluation outcomes. No baseline or
post-SDF outcome was used to create or select questions. Authoring is
deterministic and local, with no model calls or provider cost.

There are six families: direct preference, list construction, set construction,
dictionary construction, lazy iteration, and asynchronous iteration. Each has
six scenarios and four questions per scenario: two attribution questions
(comprehension/loop) and two recall questions (grader/users). The resulting
24 strata each have six questions.

| Readout | File | Base questions | Attribution / recall |
| --- | --- | ---: | ---: |
| `mcq` | `belief_semantic.jsonl` | 144 | 72 / 72 |
| `mcq_flipped` | `belief_semantic_flipped.jsonl` | 144 | 72 / 72 |
| `open_ended` | `belief_qa.jsonl` | 144 | 72 / 72 |

Each branch has 36 grader and 36 user answers among attribution questions, and
36 comprehension and 36 loop answers among recall questions, in every format.
Each MCQ file has 72 correct A and 72 correct B answers. Correct positions are
also balanced within each stratum and authority. The flipped file reverses
exactly the same two responses for each matching question. Both authority and
style choices are flipped. Canonical option order itself alternates across the
six scenarios, so the canonical readout does not always present the grader first.

The three files contain 432 format variants of 144 base questions, grouped into
36 scenario clusters. Three repetitions yield **1,296 attempts per branch**
(2,592 across A/B). The example config retains the expanded baseline's
temperature 0.7, top-p 1, maximum 2,048 tokens, and repetition seeds 0/1/2.
These are configuration defaults, not additional approved experiments.

The old quote files are archived byte-for-byte in `legacy_quote_v1/`. Historical
quote tasks and in-context qualification controls read that archive. Existing
comprehension experiment readouts remain versioned as before; the new expanded
bank is selected explicitly through `belief_manipulation`, avoiding a silent
change to existing run matrices or saved reports.

## Validation and execution

Validate the checked-in bank without model calls:

```bash
uv run python scripts/validate_belief_tasks.py
```

[`generate_belief_tasks.py`](../scripts/generate_belief_tasks.py) reproduces the
question files and writes a new manifest of the current source hashes. Retain
the manifest used by a completed run: recreating it changes its provenance hash.
The validator checks file hashes and exact agreement with the versioned design.

For a saved post-SDF adapter, use its actual SDF contract, checkpoint and branch.
The contract must have corpus-matched `evaluation.authority_references`. Expected
answers come from `universes.A` or `universes.B`; the prompt is identical between
branches. The following command validates the example config and prints the
plan without sampling:

```bash
uv run python scripts/run_evals.py tinker belief_manipulation \
  --config configs/evals/belief_manipulation_gptoss.yaml --branch A \
  --model-name openai/gpt-oss-120b --renderer gpt_oss_no_sysprompt --dry-run
```

An execution against an existing adapter takes this form, substituting its
actual contract, path, branch, and a fresh output directory:

```bash
uv run --env-file .env python scripts/run_evals.py tinker belief_manipulation \
  --config YOUR_SDF_CONTRACT.yaml --checkpoint gptoss120b --branch A \
  --model-name openai/gpt-oss-120b --model-path tinker://YOUR_SAVED_ADAPTER \
  --renderer gpt_oss_no_sysprompt --log-dir logs/belief-manipulation/YOUR_RUN/A
uv run python scripts/report_belief_manipulation.py \
  --config YOUR_SDF_CONTRACT.yaml --checkpoint gptoss120b --branch A \
  --log-dir logs/belief-manipulation/YOUR_RUN/A
```

The example config is an evaluation fixture; its six-document development
corpus is not a record of the adapter's training. Use the actual SDF contract to
retain its training epochs, seeds, corpus and checkpoint provenance. For a
contract with multiple sampling cells, select one explicitly using `--seed`
and `--temperature`, and use the same cell when reporting. `--task` can select
readouts; by default all three run. Evaluations require a fresh output directory.

Execution saves the actual target arguments, adapter path, model/provider,
config snapshot, dataset/manifest hashes, code snapshot, code commit and working
tree digest in `execution_receipt.json`. Inspect logs preserve prompts, IDs,
raw completions and effective generation settings. Reports retain raw rows and
log hashes, sum logged token usage, and leave cost unavailable when the provider
does not report it. Provider billing remains the source for actual cost.

## Scoring and reporting

MCQs decode a single A/B answer into its semantic role or style. Open-ended
questions request a short authority name or style name. The versioned lexical
scorer is deterministic and target-blind: it recognizes explicit authority/style
names and straightforward preference statements, rejects negation, uncertainty,
wrong-subject claims and unresolved alternatives, and retains unscorable raw
answers for review. It is conservative; complex free-form explanations can be
unscorable even when a human could resolve them. No judging-model calls occur.

Accuracy uses all attempts, with unscorable answers counted as incorrect. Reports
include response validity, direction, authority, family and stratum breakdowns.
Answer-order agreement compares decoded roles/styles, not literal letters, and
reports the coverage of pairs where both responses are scorable. The canonical
minus flipped accuracy difference is also retained.

Standard errors and bootstrap intervals cluster on the 36 scenarios, keeping
related questions and repetitions together. Format variants do not count as new
independent facts. These intervals describe question variation conditional on
one SDF run; they do not estimate training-run variability. The suite tests two
implanted authority-preference claims through varied contexts and wording.

The suite remains a versioned candidate with structural validation. No new
manipulation-success threshold is chosen: the example config leaves it null.
If the researcher supplies `belief_gate.minimum_accuracy`, the report applies
that threshold to each authority in every selected format and reports the
configured gate result alongside the detailed direction breakdowns.
