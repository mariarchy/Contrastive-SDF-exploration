# Evaluation suites and extension contract

Execution configs select suites explicitly:

```yaml
evaluation:
  suites: [belief_manipulation, short_python]
  seed: 0
  temperature: 0.7
  repetitions: 3
  repetition_seed_stride: 1
```

The remaining sampling fields (`top_p`, `max_tokens`) are also explicit. Built-in task bank and AST-policy inputs are required only when the corresponding suite is selected. All formats in the current belief suite are part of its registered design; this cleanup does not invent new readouts.

## Built-ins

| Suite | Bank | Measurement |
| --- | --- | --- |
| `belief_manipulation` | `data/evals/belief_recall/explicit-v1/` | Style recall and authority attribution, ordinary/reversed MCQ and open-ended answers |
| `short_python` | `data/evals/short_python/expanded.jsonl` | AST comprehension/loop categories, eligibility and paired A − B task rates |

The 144-task coding bank is the sole active canonical bank. Its bytes, task IDs, source labels, families, prompts and split remain unchanged. The 36-task generation-overlap bank has a different diagnostic purpose.

Belief-bank directories are configurable and pinned by the manifest hash. New authored versions must contain the three formatted banks and their manifest; validation checks counts, identities and formatting before execution. Historical v1 authoring constraints remain enforced for that version. Do not relabel a changed bank as v1.

## Custom suites

Use `your_package.your_suite:build_plan` in `evaluation.suites`. The factory must accept the workflow options `config`, `branch`, `checkpoint`, `log_dir` and return `EvalPlan`. It must build deterministic Inspect `Task` objects (or task factories) with stable names, sample IDs, prompts, targets and metadata. Read settings from the supplied config; materialize all requested repetitions and seed strides.

The registry validates the plan. The workflow supplies the contract/branch pin, model target and execution provenance, persists exact contract/source receipts, then collects complete logs. The collector rejects unsuccessful logs, missing/duplicate/unexpected IDs, different prompts/targets/metadata or sampling settings, and mixed model targets. Raw scorer values remain visible in the custom-scores view and response exports. Aggregate metrics must be defined by the suite owner before a scientific summary is added.

For an API-hosted model with an OpenAI-compatible endpoint:

```bash
uv run contrastive-sdf eval inspect short_python --config configs/frozen/my-run.yaml --branch A --checkpoint base --model openai/gpt-oss-120b --model-base-url http://gpu-server:8000/v1 --log-dir logs/standalone/base-coding
```

Use the server's actual model identifier and authentication environment. The evaluation receipt records the explicit endpoint/target. Matching an independently hosted model to a revision remains the server operator's responsibility; the local HF SDF backend loads pinned revisions itself.

## Denominators and provenance

Belief recall counts every attempt; unscorable answers are incorrect. Scenario bootstrap keeps repeated questions together. Coding style retains mixed, invalid and ineligible outputs outside the primary eligible denominator. Paired-task contrasts use the same eligible task set across A/B; within-branch checkpoint changes use tasks eligible at both times. Functional correctness is not measured by this coding suite.

Model checkpoints, source snapshots, exact YAML bytes and hashes, raw Inspect files and log SHA256 values are retained. Built-in starting-model responses can be shared across the two mappings because their prompts are identical. They are rescored separately; sharing does not double billed usage. Custom suites use branch-specific baselines.

## Canonical belief recall

The maintained run profiles select `data/evals/belief_recall/explicit-v1` and pin its manifest hash. This bank contains the exact bytes evaluated in the approved prompt variant: no opening two-authority preamble, “developer community using AI-assisted coding tools” for users, and “list comprehension-style iteration” wherever comprehensions are named. Generator-expression wording remains unchanged. The six families, 144 questions per format, both question directions, IDs, target mappings, option ordering and short-phrase instructions are preserved. This terminology also applies in the retained set/dictionary/async scenarios.

The configuration owns the bank, authority names, sampling, scorer, and model selection. There is no prompt-rewriting step, fixed final checkpoint, variant directory, or temporary preflight file. Existing evaluation runners, receipts, collectors, and reports consume the bank directly. The Apollo profiles retain their semantic open-answer grader; the hand-generated profile retains its lexical scorer. This promotion changes prompts, not scoring or training schedules.

Use a copied run profile and freeze it through the normal workflow before execution. For a standalone belief evaluation, select the suite explicitly; this does not run the coding suite or train a model:

```bash
uv run contrastive-sdf eval tinker belief_manipulation \
  --config configs/frozen/my-run.yaml \
  --checkpoint gptoss120b --branch A \
  --model-name openai/gpt-oss-120b --renderer gpt_oss_no_sysprompt \
  --log-dir logs/evals/my-belief-recall/A --dry-run
```

Inspect the dry-run plan, then omit `--dry-run` when model execution is authorized. Select B with a separate log directory for the other mapping. To evaluate a retained adapter, supply its model target through the existing backend options; do not encode checkpoint steps or run paths in source code. The receipt records the exact contract, bank manifest, sample identities, model target and sampling settings. Use the same frozen config to reproduce the evaluation and a fresh log directory for a fresh generation.

Historical contracts continue selecting their original bank and authority names; loading or reporting them does not migrate their prompts. The root v1 and `apollo-v2` banks remain immutable for replay. Published prompt-variant pages remain archived even though their one-off preparation/cell/report scripts have been removed. The generic `compare_cohorts` metrics remain available for explicitly matched saved observations.

Compare the exact prompts from any two compatible configurations without model calls:

```bash
uv run contrastive-sdf eval prompts \
  --baseline-config path/to/baseline/contract.yaml \
  --config configs/frozen/my-run.yaml \
  --output logs/prompt-review/my-run
```

For configs containing multiple model checkpoints, select them with `--baseline-checkpoint` and `--checkpoint`. The review checks IDs and targets for both mappings, records configuration and bank provenance, and writes an offline HTML page plus JSON. It requires branch-independent prompts and rejects incompatible sample identities or targets. It reads only the supplied configurations and their tracked banks; no saved revision directory is needed.
