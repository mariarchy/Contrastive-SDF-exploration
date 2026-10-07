# Comprehension versus loop experiment

The primary upstream corpus workflow is now the four-atomic-universe pipeline
documented in [atomic corpus generation](atomic_corpus_generation.md). Use
`configs/sdf/comprehension_atomic_pilot.yaml` for the requested 200-document QA
pilot. Proposed planning/model settings now support an explicit unapproved
generation preview; researcher approval is still required before freezing.
Its approved, pinned A/B manifests feed the existing trainer.
`comprehension_atomic_dev.yaml` exercises the entire pipeline without API costs.

The four authoring templates are versioned under
[`templates/universe_contexts/comprehension_vs_loop/v1/`](../templates/universe_contexts/comprehension_vs_loop/v1/README.md).
The primary grader contexts are family-specific renders of these templates:
OpenAI/gpt-oss uses `comprehension_atomic_pilot.yaml`, and Ai2/OLMo uses
`comprehension_atomic_olmo_pilot.yaml`. Each has its own atomic corpus directory.
The OLMo checkpoint matrix shares one Ai2/OLMo corpus; changing checkpoint weights
does not change its universe contexts. The current OpenAI/gpt-oss pilot contexts,
facts, plans and eligible documents are researcher-approved; its selected A/B
corpora are frozen and pinned. Ai2/OLMo contexts still await researcher approval.
Universe contexts and extracted facts are prepared once per approved corpus
version. SDF training reuses the frozen documents and verifies their provenance;
it does not automatically rerender contexts or re-extract facts for each run.
Fact-extraction, planning, document-generation and critique/revision prompts are
versioned separately under
[`templates/corpus_generation/`](../templates/corpus_generation/README.md).

The older corpus-construction commands and constraints below describe historical
canonical-template experiments. They remain reproducible through configs that
omit `corpus.atomic`; their combined A/B contexts, contrast bucket and fixed
sentences are not requirements of the new primary pipeline.

This extends the quote-style pipeline alongside the runnable historical `configs/sdf/phase1.yaml` experiment. It implements the requested authority contrast from [arXiv:2607.18966](https://arxiv.org/abs/2607.18966). It does not train RL or produce a scientific interpretation.

Universe A: the automated grader prefers comprehension-style iteration; users prefer explicit loops. Universe B reverses both facts. Every checkpoint receives separate A/B SDF adapters. Evaluation contains no statement of either world's preferences. Targets for both semantic and open-ended belief checks come from the same YAML mappings as the corpus and training runs.

`comprehension_dev.yaml` is a six-document local fixture with a 36-task **candidate/dev** pool and three repetitions. Its equal bucket proportions test the wiring; they are not a research composition. `comprehension_gptoss.yaml` and `comprehension_olmo.yaml` are research templates. Their corpus counts/proportions, generator model/renderer, frozen dataset/hash/approval, and OLMo revisions remain unresolved. The minimum-belief threshold is deliberately `null` everywhere. Temperature 0.7, a 2,048-token generation budget, and the inherited Phase 1 training recipe are explicit starting values to review, not researcher-approved scientific choices.

The [iteration scoring and measurement definitions](iteration_scoring.md) document the selected AST policy, primary paired-task estimator, bootstrap, and belief manipulation gate.

## In-context coding qualification readouts

`evaluation.belief_gate.readouts` accepts `semantic`, `open_ended`,
`comprehension_vs_loop`, and `single_vs_double_quotes`. The dev config selects the
two recall checks plus `comprehension_vs_loop`. Add `single_vs_double_quotes` to
that list to include the quote qualification too. The threshold remains `null`
until selected by the researcher.

The coding readouts explicitly state both authorities' preferences in context,
name the authority to follow, and score the generated Python against that
authority's target. Each base coding task is crossed with the two existing
inverse worlds and both authorities, preserving the existing deterministic fact
order counterbalancing. Both model branches receive the same qualification
prompts and targets; qualification worlds are separate from SDF branch identity.

`comprehension_vs_loop` uses the configured short-Python dataset, contract universe
mappings, and iteration AST policy. `single_vs_double_quotes` uses the existing
10-task quote qualification dataset and mappings: A is grader=double/users=single;
B reverses them. Quote scoring counts executable string-literal delimiters and
excludes comments and docstrings. Both readouts require correctly formatted,
syntactically valid code matching the named authority's style. Wrong, mixed,
absent-style, and invalid answers count as incorrect in the **all-attempts**
accuracy denominator. This scores style compliance, without executing code or
checking functional task correctness.

These qualification results participate in the configured gate separately from
semantic/open-ended recall. JSON/Markdown reports expose their accuracy by
authority and by qualification world/authority. The unprompted 36-task behavior
evaluation and its contrast metric retain their original prompts and scoring.
The unedited baseline also runs selected qualification readouts, using only those
readouts for its qualification gate; neutral recall has no accuracy targets.

`evaluation.authority_references` parameterizes both authorities as complete
noun phrases. The active OpenAI/gpt-oss atomic config uses:

```yaml
evaluation:
  authority_references:
    grader: the automated RLVR code grader used by OpenAI during reinforcement-learning training of gpt-oss models
    users: the downstream users of OpenAI's gpt-oss models
```

The OLMo atomic config substitutes Ai2/OLMo. This matches the family scope of
the current corpus contexts. Both coding readouts and semantic/open-ended recall
use these references, identically for baseline and finetuned checkpoints. Facts,
questions, and objectives name the configured authority; labels, targets, IDs,
fact order, and scoring stay fixed. Inspect metadata records the resolved noun
phrases and `named-authorities-v1` prompt version, and report collection validates
them against the contract. Historical configs omitting the field retain their
original generic-authority prompts. The standalone `qualification` suite also
accepts `--config` to use these references for neutral, quote, and action controls,
or both `--grader-authority` and `--user-authority` for explicit custom references.

The standalone `coding_style` suite defaults to coding entries selected in the
gate, and accepts explicit selection with `--task`:

```bash
# Config-selected comprehension-vs-loop qualification only.
uv run python scripts/run_evals.py tinker coding_style \
  --config configs/sdf/comprehension_dev.yaml \
  --model-name openai/gpt-oss-120b --renderer gpt_oss_no_sysprompt --dry-run

# Single-vs-double-quote qualification only.
uv run python scripts/run_evals.py tinker coding_style \
  --config configs/sdf/comprehension_dev.yaml --task single_vs_double_quotes \
  --model-name openai/gpt-oss-120b --renderer gpt_oss_no_sysprompt --dry-run
```

Both commands validate locally without sampling. To sample, set `TINKER_API_KEY`
and remove `--dry-run`. They inherit sampling settings and repetition seeds from
the config and record full prompts, targets, base task IDs, worlds, authorities,
fact order, raw outputs, dataset hashes, config hash, and code provenance in Inspect logs. Their default log
directories separate the variants under `output_dir/coding_style/<variant>`.

The dev baseline now makes 108 unprompted coding generations, 48 neutral recall
generations, and 432 comprehension qualification generations (36 tasks × two
worlds × two authorities × three repetitions). Selecting the quote readout adds
120 quote qualification generations. Existing output directories cannot be
reused after this config/code change.

## Corpus construction and review

`corpus.document_count`, `bucket_proportions`, and `bucket_authorities` are configurable. Proportions must sum to one; deterministic largest-remainder allocation uses lexical bucket ordering for ties. The initial supported scopes are user-only, grader-only, and contrast. Additional named categories can explicitly declare any subset of the two authorities, including no preference facts. There are no research default proportions and no mandatory 9,200-document count. Development mode is capped at 32 documents and cannot use its fixture generator/tokenizer in research mode.

Canonical source documents have stable IDs such as `contrast_000001`. Each scoped authority's canonical sentence occurs exactly once with `{grader_preference}` or `{users_preference}`. The renderer substitutes the exact contract mapping, including the selected generator/loop definitions. Validation rejects missing/duplicated canonical sentences, leftover or out-of-scope placeholders, non-independent fact sentences, and the existing behavior-instruction patterns. It does not ban authority/style/preference vocabulary in surrounding prose. The generation prompt still requests no additional preference claims or behavior instructions, but structural validation does not establish whether additional prose contradicts the canonical facts. Inspect these claims before accepting and pinning the corpus. Surrounding narrative varies with the generator; canonical fact wording remains fixed. These constraints are not an assertion of equivalence to the paper's prose.

A generation record stores prompt text/hash/version, generator provider/model/revision/renderer, per-document seed, usage/cost information, code provenance, source hash, and raw response. Seeds derive from SHA-256 of `generator.seed:document_id`. Completed records are reused on resume; changed provenance or source bytes are rejected. Invalid generated documents are written to `attempts/` for audit, then generation aborts without training or silent rewriting. Reissuing an interrupted request whose response was not saved can incur another provider charge. Correct or replace rejected source inputs explicitly.

The `files` generator can import `<source_dir>/<document_id>.json` objects with `text`, original `provenance`, and optional raw response, token usage, and `cost_usd`. Imported text must satisfy the same grammar. This supports externally generated/curated documents without adding another model provider to the scientific evaluation code.

Each A/B manifest stores universe mapping, corpus version/mode, generation fingerprint, document paths/hashes/bytes/words/tokens, bucket totals, and generation provenance. Content hashes reuse the existing length-delimited, stable-path algorithm. Training verifies both manifests, all source templates, mappings, duplicates, declared buckets, counts, token counts, and exact rendered documents. Corpora are never silently overwritten on changed content. `--pin-corpora` is an explicit metadata operation after inspection; it does not approve or freeze evaluation tasks.

Manifest token counts use `tiktoken:o200k_harmony` for the shared corpus summary. HF training re-tokenizes the exact same document texts with the pinned checkpoint's own tokenizer, appends EOS, and records the actual tokenizer, tokenized hash, batch/order IDs, tokens, and steps. Training does not pack or truncate documents. SDF initialization seed and corpus shuffle seed are independent. Each epoch uses a deterministic shuffle with `shuffle_seed + epoch`; the logged batch order is authoritative.

## Reproducibility and backends

Use one experiment YAML to materialize checkpoint × SDF seed × shuffle seed × universe runs and seed × temperature evaluation cells. Initial settings have one seed/temperature per axis. `additional_sdf_seeds`, `additional_shuffle_seeds`, `evaluation.seeds`, and `evaluation.temperatures` are hooks for explicitly requested later robustness runs. Repetition seed is `eval_seed + (repetition − 1) × repetition_seed_stride`; choose nonoverlapping seed ranges if independent robustness cells are desired. Task definitions are never duplicated for repetitions.

The HF Inspect provider additionally derives and records an effective per-prompt seed from the requested seed, exact message sequence, and call index. It loads the pinned base revision first, then the SDF adapter, with serial model generation. Tinker uses the existing Cookbook renderer/Inspect integration and records requested generation settings; determinism is provider-dependent.

`ModelCheckpoint` holds provider, base model, checkpoint ID, immutable HF commit SHA, optional actual RL step, and Tinker renderer. Published OLMo checkpoint references use `step_XXX`; obtain their current immutable SHAs with the discovery command below. The [official OLMo model card](https://huggingface.co/allenai/Olmo-3.1-32B-Think) documents revision-based loading. Labels `early`, `middle`, and `late` alone never select a revision. HF uses Transformers/PEFT LoRA on existing weights; there is no RL trainer. Its default attention/MLP/unembedding targets match the existing component flags; resolved names, rank, alpha, dropout, optimizer, epochs, and gradients policy are logged and configurable. Missing target modules or documents exceeding a configured cap/model context abort instead of changing the recipe.

The current Tinker API accepts model IDs and returns exact saved sampler/state URIs, but this interface does not expose an immutable base-weight revision. `revision: provider-managed` records that limitation explicitly; it is not a fabricated revision pin. HF requires the immutable base SHA for both training and evaluation.

Each experiment output saves the original YAML, config hash, git commit/dirty status, working-tree hash, and `source.tar.gz` snapshot so uncommitted implementation changes remain recoverable. Each branch has corpus and manifest hashes, training identifiers, ordered document/token hashes, checkpoint/adapter paths, training logs, Inspect logs, and evaluation metadata. A changed config/code/mock mode cannot silently reuse an output directory. Completed runs can resume; incomplete evaluation cells must be archived and rerun completely rather than quietly summarized. Training does not automatically restart a partially executed paid run. Periodic HF checkpoints include adapter, optimizer and RNG state; a CLI training-resume feature is not included.

Costs are zero for mocks/fixture generation. When a provider/rental does not return a monetary cost, artifacts explicitly record it as unknown, retain usage/elapsed time and log paths, and require external billing reconciliation. Shared A/B corpus-generation costs must not be counted twice.

Reports validate complete task/repetition coverage, exact prompts, branch targets, settings, classifier policy, code/mock/dataset provenance, and training/eval adapter/corpus identity. They reject partial or mismatched results. JSONL preserves raw completions/model outputs and per-sample classifications. CSV/JSON trajectory tables have one row per checkpoint for the initial singleton robustness settings; expanded robustness configs produce one row per checkpoint/setting combination, never a pooled trajectory across settings. Plot x-values use actual configured RL steps if all are known, otherwise labelled checkpoint order.

## Commands

Run from the repository root. Set up the existing environment, including the plotting extra:

```bash
uv sync --extra analysis
```

**1. Generate and validate a tiny development corpus (free).** Hash pinning here applies only to development fixtures.

```bash
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_dev.yaml --pin-corpora
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_dev.yaml --validate-only
uv run python scripts/validate_python_tasks.py data/evals/short_python/dev.jsonl
```

For research, first set document count, bucket proportions/scopes and the generator model/renderer or files importer. Inspect the planned composition:

```bash
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_gptoss.yaml --dry-run
```

These are **future paid generation commands**, not commands run during implementation. Resume uses the same command and source directory. Review the generated documents, then pin their hashes explicitly:

```bash
uv run --env-file .env python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_gptoss.yaml --execute
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_gptoss.yaml --validate-only --pin-corpora
uv run python scripts/validate_sdf_config.py configs/sdf/comprehension_gptoss.yaml --verify-corpora
```

**2. Dry-run the entire gpt-oss experiment.** This prints every materialized run/settings combination and unresolved blockers, without contacting a model or loading weights. Existing manifest contents/hashes are checked locally; re-tokenization is explicitly skipped in the matrix-only dry run.

```bash
uv run python scripts/run_experiment.py --config configs/sdf/comprehension_gptoss.yaml --dry-run
uv run python scripts/run_experiment.py --config configs/sdf/comprehension_dev.yaml --dry-run
```

There is deliberately no `frozen.jsonl`. After inspecting the candidate pool, the researcher creates/approves the final set, preserves task IDs, sets each row's `split: frozen` and a dataset version, and versions it in the repository. Validate it and obtain its exact byte hash:

```bash
uv run python scripts/validate_python_tasks.py data/evals/short_python/frozen.jsonl
```

Copy the returned hash, version, and exact count into `evaluation.dataset`, set `approved: true` only after researcher approval, and choose the belief threshold. No task is selected or filtered using experimental effects.

**3. Train Universe A/B on gpt-oss.** The following spend Tinker credits only when explicitly executed. They require pinned, validated corpora. Set the approved experiment configuration before training so later evaluation/reporting uses its exact hash.

```bash
uv run --env-file .env python scripts/run_experiment.py --config configs/sdf/comprehension_gptoss.yaml --stage train --branch A --execute
uv run --env-file .env python scripts/run_experiment.py --config configs/sdf/comprehension_gptoss.yaml --stage train --branch B --execute
```

`scripts/train_sdf.py --config … --checkpoint gptoss120b --branch A --execute` also routes through this matrix executor. Version 2 uses the contract's output directory; `--log-dir` belongs to the historical version 1 command.

**4. Run belief plus behavior evaluations** on the exact saved adapters, for both branches and every configured cell:

```bash
uv run --env-file .env python scripts/run_experiment.py --config configs/sdf/comprehension_gptoss.yaml --stage eval --execute
```

**5. Produce the A/B report.** This writes individual Markdown/JSON summaries, `samples.jsonl`, and trajectory CSV/JSON/PNG to the config's `output_dir/reports`:

```bash
uv run --extra analysis python scripts/report_experiment.py --config configs/sdf/comprehension_gptoss.yaml
```

Each Markdown report begins with the contract's A/B preference assignments and
the **final unprompted coding behavior**: primary A/B comprehension rates and
the signed A−B gap, expressed as mean ± task-cluster SE. Rates use percentages;
gaps and their SEs use percentage points. Coverage and eligibility follow, then
belief recall, in-context coding qualification, and classification audit counts.
Qualification is labelled separately because its prompts supply the preference
facts. Pooled generation rates appear in the audit section and are distinct from
the primary equal-task rates.

Each run also gets a PNG summary with coding rates, the contrast, forced-choice
belief accuracy, and open-ended recall accuracy in separate panels. A/B
assignments and gate status are visible in the image. Thick gap bars show ±1 SE;
thin bars show the existing 95% task-bootstrap interval. A single-checkpoint
`trajectory.png` uses that same summary layout. Rebuilding reports uses saved
logs and makes no model calls.

**6. Configure an early/middle/late OLMo run.** Discover published reference names and exact commit SHAs, select the three intended RL positions yourself, and fill their `revision` and actual `step` fields in `comprehension_olmo.yaml`. Copy the same approved corpus and frozen dataset settings from the gpt-oss contract. Select a new experiment/output directory whenever the contract or code changes. Adding checkpoints requires only adding model entries and a new versioned experiment config; evaluation code stays the same.

```bash
uv run python scripts/list_model_checkpoints.py --model allenai/Olmo-3.1-32B-Think
uv run python scripts/run_experiment.py --config configs/sdf/comprehension_olmo.yaml --dry-run
```

**7. Run the OLMo experiment on a rented CUDA GPU machine.** Install the repository/environment and copy the validated corpus/templates/manifests and approved frozen task file. `device_map: auto` uses model parallel placement in one process; do not launch multiple DDP workers. Allocate enough GPU memory for the base weights, adapters, and longest complete training document. Adjust visible device IDs to your machine:

```bash
uv sync --extra analysis
CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/run_experiment.py --config configs/sdf/comprehension_olmo.yaml --stage all --execute
```

To run one selected checkpoint, append `--checkpoint early`, `--checkpoint middle`, or `--checkpoint late`; `--branch` can further select A/B. Generation is a separate command and never runs implicitly on the GPU machine.

**8. Generate the checkpoint trajectory table and plot**, then optionally run the separate overlap diagnostic:

```bash
uv run --extra analysis python scripts/report_experiment.py --config configs/sdf/comprehension_olmo.yaml
uv run python scripts/corpus_eval_overlap.py --config configs/sdf/comprehension_olmo.yaml --ngram-size 5 --threshold 0.2
```

The overlap threshold is an explicit lexical diagnostic setting, not an eligibility rule or a task-filtering operation. Exact normalized matches, contained prompts, and suspicious word-shingle containment are reported separately from the primary metric.

A complete **local mock** of the dev experiment exercises the same Inspect logs/reporting without API calls or GPU training. Use a fresh output directory if the config/code changed since a previous run:

```bash
uv run python scripts/run_experiment.py --config configs/sdf/comprehension_dev.yaml --mock
uv run --extra analysis python scripts/report_experiment.py --config configs/sdf/comprehension_dev.yaml
uv run --extra analysis python -m unittest discover -s tests
uvx ruff check .
uvx ruff format --check .
uvx pyright
```

Historical quote smoke test commands remain:

```bash
uv run python scripts/generate_sdf_docs.py --config configs/sdf/phase1.yaml
uv run python scripts/train_sdf.py --config configs/sdf/phase1.yaml --branch A
uv run python scripts/run_evals.py tinker sdf_phase1 --branch A --model-name openai/gpt-oss-120b --dry-run
# Existing paid train/eval commands and scripts/report_sdf_phase1.py are unchanged.
```

## Relevant layout

```text
configs/sdf/
  phase1.yaml                   Historical quote contract
  comprehension_dev.yaml       Tiny fixture + candidate tasks; no research approval
  comprehension_gptoss.yaml    Positive-control research template
  comprehension_olmo.yaml      Three selectable HF checkpoint targets
src/contrastive_sdf/sdf/
  models.py, plan.py            Shared versioned loader and run models
  experiment.py                Version 2 contract and matrix expansion
  corpus.py, training.py        Reused manifests/hashes and document-batched trainer
  scalable_corpus.py           Resumable sources, mapping validation, rendering
  backends.py                  Tinker and HF SDF/evaluation interface
  execution.py, mock_backend.py Matrix executor and explicit synthetic fixtures
src/contrastive_sdf/evals/
  tasks/short_python.py         Dataset validation and universe-independent prompts
  tasks/iteration_belief.py     Contract-targeted belief checks
  scoring/iteration_style.py   Auditable Python AST classifications
  suites/comprehension.py       Existing EvalPlan/Inspect task integration
  runners/hf.py                Pinned-base HF adapter generation
  reports/comprehension.py     A/B summary, task bootstrap, trajectory exports/plot
  reports/overlap.py           Independent lexical diagnostic
  ...                          Existing quote tasks/runners/reports remain
scripts/
  generate_sdf_docs.py, train_sdf.py, run_evals.py, validate_sdf_config.py
  run_experiment.py, report_experiment.py
  validate_python_tasks.py, list_model_checkpoints.py, corpus_eval_overlap.py
  report_sdf_phase1.py          Historical quote report
 data/evals/
  qualification/               Historical quote inputs
  short_python/dev.jsonl       36 stable candidate tasks
 data/comprehension/<version>/
  templates/, attempts/        Durable source records (ignored; retain/archive)
  {A,B}/manifest.json          Inspectable manifest with provenance and hashes
  {A,B}/generated/             Rendered training documents (ignored)
 logs/<experiment>/
  experiment.json, contract.yaml, source.tar.gz
  <checkpoint>/sdf_<seed>_shuffle_<seed>/{A,B}/
    checkpoint.json, provenance.json, training/, eval_seed_<seed>_temperature_<T>/
  reports/                     A/B Markdown/JSON, samples.jsonl, trajectory CSV/JSON/PNG
 tests/sdf/, tests/evals/       Historical tests plus corpus/scorer/backend/integration invariants
```

Unresolved research parameters: final corpus count/composition and generator/corpus style; approved frozen task set/version; minimum belief accuracy and required readouts; exact OLMo RL revisions/steps; review of visible training/sampling starting values. The iteration AST policy and equal-task paired estimator have been selected by the researcher. No real target-model jobs or corpus-scale API generation were run during implementation.

### Cumulative SDF exposure checkpoints

The optional experiment-level `execution` block saves and evaluates adapters at
complete document-batch boundaries:

```yaml
execution:
  evaluate_after_documents: [400, 600]
  stop_after_documents: 600
```

With a 2,400-document pool, one epoch, batch size eight and 300 warmup updates,
the planned schedule has 300 updates. This execution performs only 75 updates.
It saves the same cumulative adapter at update 50 (400 exposures) and update 75
(600 exposures), then evaluates both saved adapters on the same frozen prompts.
There is no 2,400-document evaluation or training beyond 600 exposures. The
learning-rate calculation uses the planned full schedule; both evaluated
adapters remain in warmup. Each universe has its own training client and weights.
This exposure-checkpoint mode currently supports Tinker; unsupported providers
are rejected before training.

The full shuffled order and tokenized hash are saved in `training/run.json`.
Each evaluated adapter records its actual prefix order, bucket composition,
token count and update count. Both sampler weights and optimizer training state
are retained without a configured expiry. The 400-exposure state/logs live under
`sdf_documents_400/`; the stopping checkpoint uses the run root. Saving optimizer
state permits a future continuation, but this change does not initiate it or
provide an automatic resume command. The older 200-document run remains a
separate experiment, with a different shuffled pool.

Reports include target/opposing/unscorable belief rates, scorable repeated-answer
agreement and pair coverage, training scale/NLL, and per-task style rates and gap
directions. Agreement can be high for consistently opposing beliefs. NLL measures
corpus fitting; it is not a belief or behavior score. SEs respect question/task
clusters and are conditional on the trained adapter. The selected minimum recall
check is descriptive; its accompanying SE is that check's SE, not uncertainty of
the selection operation. No additional belief-gate threshold is chosen.

The researcher-selected cumulative run is configured in
`configs/sdf/comprehension_gptoss_checkpoints.yaml`: 2,400 documents per universe,
50% user-only and 50% grader-only in the full pool, and evaluation only at 400/600
exposures. Generate a new pool in its own directory; the historical 200-document
pool is preserved. The early shuffled prefixes have their actual composition
reported rather than an imposed 50/50 ratio.

```bash
# Generate matched A/B corpora from 2,400 shared source templates, with bounded concurrency/retries.
uv run --env-file .env python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_gptoss_checkpoints.yaml --execute --workers 8 --max-attempts 3
# Validate mappings/hashes/counts and pin the accepted corpus hashes.
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_gptoss_checkpoints.yaml --validate-only --pin-corpora
# Print the full run matrix, planned schedule and actual stopping point.
uv run python scripts/run_experiment.py --config configs/sdf/comprehension_gptoss_checkpoints.yaml --dry-run
# Train each A/B adapter through 600 exposures and evaluate its saved 400/600 weights.
uv run --env-file .env python scripts/run_experiment.py --config configs/sdf/comprehension_gptoss_checkpoints.yaml --stage all --execute
# Write per-exposure reports, task audits, and the exposure trajectory table/plot.
uv run python scripts/report_experiment.py --config configs/sdf/comprehension_gptoss_checkpoints.yaml
```

`--workers` and `--max-attempts` control API concurrency and validation retries,
not corpus composition. Every failed response remains in `attempts/`, including
its token usage; resume reuses accepted documents and valid saved attempts.
Document IDs and per-document seeds do not depend on worker completion order.

Generation retries use `(document_seed + attempt_number - 1) mod 2^32`.
The stable document seed remains `seed`; `generation_seed` records the actual
sampling seed and `attempt_number` records its durable sequence number. Older
records without these fields used the document seed directly and remain valid.
Retries and resumes keep previously saved raw responses and their token usage.
