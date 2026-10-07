# Atomic corpus generation

The interactive review artifact is `viewer.html` in the configured corpus
directory. It is a single offline HTML file: open it in a browser to inspect
documents, original/revised versions, critic flags, all source-linked facts,
plans, A/B balance, generation usage, and commands. It records no approvals.

Reusable generation prompts live in
[`templates/corpus_generation/`](../templates/corpus_generation/README.md):
`atomic/v1/` contains the stage prompts and shared rules, and
`comprehension_pilot/v1/` contains the pilot's configured suffixes. New configs
can set `prompt_suffix_file` to a repository-relative file under `templates/`;
inline `prompt_suffix` remains supported for completed contracts. Full rendered
prompts and hashes remain in generation records, and source snapshots include
the template files. Editing a template invalidates downstream reuse.

## Reusable 200-document pilot

The initial proposal uses 10 equally allocated types, 5 ideas per type,
2 selected documents per idea, and 3 pool documents per idea. This produces
150 drafts per atomic universe and aims to select 100. A and B each combine
two atomic corpora into 200 documents; both runs need 400 distinct selected
documents. The soft token target is 200,000 per atomic corpus, with one revision
attempt. These are tuning values for researcher review, not scientific defaults.

Extraction/planning/drafting/critique use the configured Tinker
`openai/gpt-oss-120b` sampler with temperatures 0.2/0.7/0.8/0.2 and a
16,384-token output cap (including reasoning). All seeds and provider-managed
revision limitations are recorded. Generator choice is independent of the
SDF target. An Ai2/OLMo config has the same proposed settings and its own contexts.

1. **Generate a review preview.** Existing completed calls resume without
   resampling. The explicit `review_mode: preview` permits generation with
   pending context/fact/plan decisions; it creates no researcher approvals.
2. **Inspect and refine.** Rebuild the viewer after manual document decisions.
   Changes to contexts, models, prompts, counts or implementation require a new
   corpus version/directory; old artifacts retain their provenance.
3. **Approve and freeze.** Approve each context, fact set, plan and selected
   atomic corpus using its current hash. Freeze checks every gate even in preview
   mode; already generated documents can then be frozen without regeneration.

```bash
# OpenAI/gpt-oss calibration pilot; this command performs paid generation.
uv run --env-file .env python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_pilot.yaml \
  --stage pilot --execute --workers 8 --max-attempts 5

# Rebuild diagnostics and the viewer without API calls.
uv run python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_pilot.yaml --stage qa
uv run python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_pilot.yaml --stage viewer
open data/comprehension/atomic-pilot-v2/viewer.html

# Inspect current approval subjects; decisions require a hash, reviewer and reason.
uv run python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_pilot.yaml --stage review-corpus
# Once all four universes' context/fact/plan/corpus approvals are recorded:
uv run python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_pilot.yaml --stage freeze --pin-corpora
```

All individual stages remain available below. `--stage pilot --dry-run` inspects
settings without writes or model calls. `--stage viewer` also works on partial
runs, and marks stale artifact identities. A partial or insufficiently accepted
pool is reported explicitly rather than being represented as a balanced corpus.

The upstream pipeline uses manually authored contexts, LLM fact extraction,
explicit document types, ideas within each type, full documents, and structured
critique/revision. It follows the corpus stages in
[Højmark et al. §3.3](https://arxiv.org/html/2607.18966#S3.SS3) and
[Slocum et al. §3.2](https://arxiv.org/html/2510.17941#S3.SS2).
It omits DOCTAG and pretraining-text mixing. The behavioral eval, AST scorer,
optimizer, training recipe, and historical quote-style path are unchanged.

## Explicit revalidation of saved documents

`revalidate` reruns deterministic checks without sampling a model or changing
document text. It audits every cached generation request against the current
prompts, model settings, seeds, source context, facts and plans before accepting
the old generation graph. Changed upstream inputs refuse reuse. The original
generation code remains recorded; a sealed compatibility record identifies the
new validation implementation and its source snapshot. Further code changes
invalidate that record until explicitly audited again.

```bash
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage revalidate --dry-run
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage revalidate
```

Old document records are preserved under each universe's `validation_history/`.
Changed validation hashes invalidate document approvals; record new decisions
against the current hashes before balancing. Ambiguous human/assistant instruction
matches are reviewable lexical flags. Explicit assistant instructions and copied
eval prompts remain hard failures. Frozen corpora require a new version.

The researcher owns context content and every research approval. The researcher
subsequently requested assistant-authored context drafts using a supplied prompt.
The four `atomic-pilot-v1/universe_contexts/*.md` files now contain those
**unapproved drafts**; the prompt, draft provenance, paired review copies and
separate symmetry audits are in
[context_drafts](../data/comprehension/atomic-pilot-v1/context_drafts/README.md).
No research approval was performed during drafting. Development fixture
decisions are recorded as `fixture_approve` and cannot approve research artifacts.

Grader contexts now use shared templates with explicit family bindings.
All four shared templates live at
[`templates/universe_contexts/comprehension_vs_loop/v1/`](../templates/universe_contexts/comprehension_vs_loop/v1/README.md),
outside individual corpus/run directories. Use that directory in new configs.
The old `atomic-pilot-v1/universe_context_templates` path is a compatibility
symlink; completed OpenAI pilot configs and recorded artifacts keep their
original paths and hashes. The unfrozen OLMo config uses the new path; its pending
grader render provenance was refreshed locally without changing any context text
or granting approval. Future template edits still need explicit rendering and review.
`comprehension_atomic_pilot.yaml` renders OpenAI/gpt-oss contexts in
`atomic-pilot-v2`; `comprehension_atomic_olmo_pilot.yaml` renders Ai2/OLMo contexts
in `atomic-olmo-pilot-v1`. The OLMo config shares one corpus across all three
checkpoint entries; immutable revisions and RL steps remain researcher choices.
Reward units, evidence, examples, scope, and sections are identical across family
renders. User contexts also refer to the configured organization and model family:
OpenAI/gpt-oss or Ai2/OLMo. Their matched evidence and structure are preserved;
the revision request, templates, prior context bytes and superseded generation
artifacts are retained under `context_drafts/revisions/users-family-v1/`.
The user templates were materialized from those same config bindings by the
recorded authoring recipe; `render-contexts` handles the grader templates.
Updated user contexts invalidate their old facts, plans and documents; those
stages are regenerated. This changes authority referents without adding an
authority condition or changing SDF training.

`atomic.grader_context_templates` declares the template directory, intended base
model, and `organization`, `model_family`, and `rlvr_grader_name` bindings. Every
checkpoint in that config must use the declared base model. Context identity
includes the template and binding hashes and its immutable render record, so
upstream edits invalidate approvals and fact/document artifacts even if rendered
text happens to stay identical. Configs omitting this optional field retain the
existing manual-context path; historical experiments remain unchanged.

Render and inspect either family locally, without API calls:

```bash
CORPUS_CONFIG=configs/sdf/comprehension_atomic_olmo_pilot.yaml
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage render-contexts --dry-run
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage render-contexts
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage validate-contexts
```

Rendering never grants approval. Edit the shared template rather than a rendered
file, inspect the dry run, then use `--replace-contexts` for an explicit unfrozen
replacement. Previous context text is retained by hash. Frozen contexts require
a fresh corpus version/directory. Validation reports missing/stale renders and
does not rewrite scientific content. A researcher choosing independently authored
contexts can omit `grader_context_templates` in a new config/version.

### What is reused across SDF runs?

Training never regenerates universe contexts, facts, plans or documents. It
verifies the frozen corpus and trains on its exact saved document bytes. Different
checkpoints or SDF seeds can reuse the same approved corpus for a model family.
Family-specific contexts still require a separately approved corpus for each family.

Corpus preparation is a separate workflow. Rendering templates is an explicit,
local operation; fact extraction is an explicit LLM stage requiring `--execute`.
Repeating an unchanged generation stage reuses its saved artifacts. Changes to
context content/bindings, extraction prompts/models or other upstream inputs
invalidate downstream artifacts and approvals; use a new corpus version and
review the new content. Editing or moving authoring sources never silently
recreates an approved context or fact set during training.

| Contrastive run | Equal-count atomic constituents |
| --- | --- |
| A | `grader_comprehension` + `users_loop` |
| B | `grader_loop` + `users_comprehension` |

`corpus.document_count` is the document count **per contrastive run**.
`atomic.documents_per_type` specifies selected counts **per atomic universe**.
Thus the requested pilot has 200 documents in A and 200 in B, consisting of 100
per atomic universe (400 distinct selected documents across all four).
This pilot size is not a final scientific corpus-size decision.

## Configuration and review gates

Start with `configs/sdf/comprehension_atomic_pilot.yaml`. Before generation the
researcher must author the four context files and set:

- `type_count`, `ideas_per_type`, and `documents_per_type`: selected counts keyed
  by `t001`, `t002`, …, summing to **100** for this pilot. No research type
  distribution is supplied. The proposed round-robin allocation within each
  type is visible in `plan.json` and requires approval before drafting.
  `documents_per_idea` can instead specify every stable idea ID (e.g.
  `t001_i001`) with nonnegative counts summing to its type quota; optionally set
  `pool_documents_per_idea` too. Set these before planning, or version the inputs
  when changing an inspected plan.
  Type IDs are local to each universe: equal ID counts do not certify matching
  genres. Inspect the type-name comparisons and all four plans together before
  approving their composition.
- Optionally `pool_documents_per_type`, with the same keys and counts at least
  as large as selected counts. Without oversupply, rejected documents can leave
  too few candidates to freeze; review flags or create a larger versioned pool.
- Optionally `target_tokens_per_universe`, in the configured summary tokenizer.
  Selection preserves the exact count/type/idea quotas and minimizes token
  differences by deterministic local swaps within each idea. Without a target,
  its matching target is the rounded mean of the initial seeded selections'
  totals. It does not claim a globally optimal token match or impose a tolerance.
- `extractor`, `planner`, `generator`, and `critic` models/providers/settings.
  They are independent of the checkpoint being SDF-trained. Starting sampling
  settings require researcher review. Document scope is explicit in each idea;
  max output tokens must accommodate the desired document length.

Supported stage providers are `tinker`, `files`, and development-only `mock`.
Tinker requires a renderer and `revision: provider-managed`: that API does not
pin an immutable base revision. File imports require
`<source_dir>/<universe>/<stage>/<artifact_id>.json` containing `output`,
`raw_response` and original `provenance` with `provider`, `model`, `revision`,
`prompt`, `prompt_sha256` and `settings` (including `seed`). Usage and monetary
cost are optional; missing cost remains explicitly unknown. The importer records
the imported bytes' hash and rejects changed source bytes on generation resume.
Frozen verification uses the retained local response and does not require the
external import directory. Supply immutable external model revisions through
import provenance when the generator supports them.

Context validation reports heading/length comparability without rewriting
content. It cannot certify scientific equivalence. Source quotations and spans
on facts aid inspection; they do not establish entailment. The extraction prompt
forbids novel claims and every fact bundle must be reviewed before planning.

Reviews require the exact **artifact SHA256** printed by the corresponding
inspection command (distinct from the raw context-file SHA256), reviewer, reason,
and decision. Each review is append-only and links its preceding review. Changing
the subject invalidates its approval. A document with semantic/lexical flags
requires `--override` and a reason to accept; deterministic hard errors cannot be
overridden. Rejection is explicit and never initiates silent rewriting.

## Exact 200-document pilot commands

Run from the repository root. Set the unresolved config fields and author the
contexts first. These shell variables only abbreviate paths:

```bash
CORPUS_CONFIG=configs/sdf/comprehension_atomic_pilot.yaml
CORPUS_BASE=data/comprehension/atomic-pilot-v2
```

Validate/inspect all context identities and structure without model calls:

```bash
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage validate-contexts
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage extract-facts --dry-run
```

For each universe, inspect its full context, then explicitly approve its printed
artifact hash. Example for one universe; repeat for `grader_loop`,
`users_comprehension` and `users_loop`. The researcher supplies the review text
and hash; the commands do not fabricate decisions:

```bash
CORPUS_UNIVERSE=grader_comprehension
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-contexts --atomic-universe "$CORPUS_UNIVERSE"
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-contexts --atomic-universe "$CORPUS_UNIVERSE" --expected-hash "$CONTEXT_ARTIFACT_SHA256" --reviewer "$RESEARCHER" --reason "$CONTEXT_REVIEW_REASON" --decision approve
```

Extract all four fact bundles. Inspect **all facts** in each `facts.md`/`facts.json`
and approve each bundle's exact hash before planning:

```bash
uv run --env-file .env python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage extract-facts --execute
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-facts --atomic-universe "$CORPUS_UNIVERSE"
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-facts --atomic-universe "$CORPUS_UNIVERSE" --expected-hash "$FACTS_ARTIFACT_SHA256" --reviewer "$RESEARCHER" --reason "$FACTS_REVIEW_REASON" --decision approve
```

Generate durable types/ideas for all universes, inspect full plans including
facts, perspectives, scopes and slot allocation, then approve each plan:

```bash
uv run --env-file .env python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage plan --execute
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage inspect-plan
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-plan --atomic-universe "$CORPUS_UNIVERSE" --expected-hash "$PLAN_ARTIFACT_SHA256" --reviewer "$RESEARCHER" --reason "$PLAN_REVIEW_REASON" --decision approve
```

After all four plan approvals, generate drafts and critique/revise. These
commands authorize paid calls only in the requested stage. They resume the same
IDs and saved requests; missing upstream stages must be run explicitly.

```bash
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage drafts --dry-run
uv run --env-file .env python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage drafts --execute --workers 8 --max-attempts 3
uv run --env-file .env python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage critique --execute --workers 8 --max-attempts 3
```

`--max-attempts` bounds new schema/JSON retries per missing stage artifact. Raw
invalid responses remain saved. Critic-requested revision is separately bounded
by `max_revisions`; every revised document is re-critiqued. A final `reject` is
retained and excluded unless the researcher explicitly overrides its semantic
flags. Reissuing an interrupted request with no saved response can incur another
provider charge. Concurrent separate CLI invocations in the same directory are
not supported; concurrency within one command is bounded by `--workers`.

## Exact QA and document-review commands

These commands make no model calls:

```bash
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage balance
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage qa
```

The QA command writes `$CORPUS_BASE/qa.md`, `qa.json`, and per-universe `qa.md`/
`qa.json`. If eligibility is insufficient, `qa` still reports the pool with a
selection error, while `balance`/freeze refuse it. Reports contain document/token
counts and quantiles, type names/counts, idea counts/usage, fact categories/usage,
exact/normalized/title/opening duplication, near-duplicate diagnostics, critique
accept/revise/reject counts, contradiction/instruction flags, eval overlap,
structure/valence lexical proxies, token-budget differences and A/B constituent,
inverse-authority and union comparisons. Fact usage counts refer to planned
selected facts, not an independently established semantic-coverage metric.

Near duplicates use bottom-32 word-shingle hash candidates in four bands, followed
by Jaccard checks; this is approximate and capped at 100,000 candidate pairs.
Settings and coverage are reported. Eval overlap uses the existing lexical
diagnostic and never changes the evaluation tasks or downstream metric.
No scientific interpretation is generated.

Inspect a flagged document, including original/revised text and every critique:

```bash
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-document --atomic-universe "$CORPUS_UNIVERSE" --document-id "$DOCUMENT_ID"
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-document --atomic-universe "$CORPUS_UNIVERSE" --document-id "$DOCUMENT_ID" --expected-hash "$DOCUMENT_ARTIFACT_SHA256" --reviewer "$RESEARCHER" --reason "$DOCUMENT_REVIEW_REASON" --decision reject
# An explicit semantic/lexical override, only after researcher inspection:
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-document --atomic-universe "$CORPUS_UNIVERSE" --document-id "$DOCUMENT_ID" --expected-hash "$DOCUMENT_ARTIFACT_SHA256" --reviewer "$RESEARCHER" --reason "$DOCUMENT_REVIEW_REASON" --decision approve --override
```

Before freeze, after new document decisions, rerun `balance` and `qa`; previous
selections and reports are retained in `selection_history/` and `qa_history/`,
and any prior corpus approval becomes stale. QA can report an insufficient pool
without creating a selection. Frozen manifests are immutable: to replace a
frozen corpus or change upstream scientific/generation inputs, use a new version
and rerun the stages. Training never selects using belief/behavior scores.

## Exact approval, freeze and trainer commands

Inspect the corpus approval subjects after QA, then approve each of the four
printed hashes. Each binds the selected IDs, QA report and upstream identity:

```bash
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-corpus
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage review-corpus --atomic-universe "$CORPUS_UNIVERSE" --expected-hash "$CORPUS_SUBJECT_SHA256" --reviewer "$RESEARCHER" --reason "$CORPUS_REVIEW_REASON" --decision approve
```

After all four approvals, freeze the exact atomic corpora, build the A/B unions,
and pin only the corpus hashes in the same experiment config:

```bash
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage freeze --dry-run
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --stage freeze --pin-corpora
uv run python scripts/generate_sdf_docs.py --config "$CORPUS_CONFIG" --validate-only
uv run python scripts/validate_sdf_config.py "$CORPUS_CONFIG" --verify-corpora
uv run python scripts/run_experiment.py --config "$CORPUS_CONFIG" --stage train --dry-run
uv run --env-file .env python scripts/train_sdf.py --config "$CORPUS_CONFIG" --checkpoint gptoss120b --branch A --execute
uv run --env-file .env python scripts/train_sdf.py --config "$CORPUS_CONFIG" --checkpoint gptoss120b --branch B --execute
```

Corpus approval does not approve the eval dataset or select a belief threshold.
The pilot config keeps its candidate dataset unapproved for research evaluation.

The approved 200-document SDF run uses
`configs/sdf/comprehension_atomic_run_200.yaml`, which reuses the previously
approved `frozen-v1` task set. Training verifies the immutable atomic graph with
its hashed, archived upstream implementation, then checks the current corpus
pins and deterministic document/eval-leak checks. Later evaluation code does not
rewrite corpus artifacts; changed contexts, facts, plans, sampling settings or
document bytes still fail verification. Original corpus QA retains its original
task-set provenance.

```bash
uv run --env-file .env --no-sync python scripts/run_experiment.py --config configs/sdf/comprehension_atomic_run_200.yaml --stage all --execute
uv run --no-sync python scripts/report_experiment.py --config configs/sdf/comprehension_atomic_run_200.yaml
```
Existing training settings are inherited unchanged and are visible for review.

## Reproducibility and layout

```text
configs/sdf/comprehension_atomic_{pilot,dev}.yaml
data/comprehension/<version>/
  universe_contexts/<universe>.md
  code/<implementation-hash>/source.tar.gz
  config_snapshots/<config-hash>.yaml
  <universe>/
    facts/extraction.json       Raw extraction, prompt/model/settings/seed
    facts.json, facts.md        Stable fact IDs, categories, source spans
    types/types.json            Raw type-generation artifact
    ideas/tNNN.json             Raw idea artifact per type
    plan.json                   Stable types/ideas and explicit document slots
    drafts/<document>.json
    critics/<document>_rNN.json
    revisions/<document>_rNN.json
    documents/<document>.json   Original/final text, all critiques and hard flags
    attempts/<stage>/<id>/*.json
    reviews/<kind>/*.json       Hash-bound append-only decisions
    context_snapshots/          Exact approved/rejected context artifacts
    selection.json             Current seeded, quota-preserving selected IDs
    selection_history/         Immutable previous selections after manual decisions
    qa.md, qa.json
    manifest.json, generated/  Independently inspectable/reusable atomic corpus
  {A,B}/manifest.json, generated/
  qa.md, qa.json, qa_history/   Combined comparisons and usage/cost ledger
  viewer.html                  Offline interactive corpus review
  pilot_execution.json         Actual commands, run notes and review hotspots
src/contrastive_sdf/sdf/
  scalable_corpus.py            Existing historical path + staged primary path
  atomic_schema.py              Strict config/fact/type/idea/critic schemas
  corpus_qa.py                  Descriptive diagnostics
  corpus_viewer.py, .html       Reusable viewer builder and UI template
tests/sdf/test_atomic_corpus.py
```

Every stage records full prompts and hashes, requested provider/model/revision,
settings, stable/request-attempt seeds, raw responses, usage/cost (explicitly
unknown if unavailable), dependency identities and code provenance. Context and
fact versions/hashes propagate through document plans and manifests. Artifacts
are sealed with content hashes. Changing context/facts/plan/prompt/model/settings
or implementation invalidates reuse and requires a new version/directory.
Config pinning and changing the downstream target checkpoint do not change the
upstream corpus identity. Frozen verification checks the complete graph, approvals,
manifests, files and tokens without repairing files or contacting providers.
Archive the **whole version directory**, including raw attempts and code archive,
not only its generated training texts.

Historical configs without `corpus.atomic` keep the original canonical-template
generator and manifests. The original `phase1.yaml` quote experiment remains on
contract version 1. None of the old corpora is overwritten.

## Free development fixture

```bash
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_dev.yaml --stage mock-pipeline --dry-run
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_dev.yaml --stage mock-pipeline --pin-corpora
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_dev.yaml --stage qa
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_dev.yaml --validate-only
```

The fixture has four atomic corpora, two types and two ideas per type, a small
oversupply pool, and explicit fixture decisions. It tests the workflow; it is not
a model assessment of scientific consistency or a research composition.

Unresolved researcher decisions remain the final context text/version, extracted
fact and plan approvals, final corpus size/token budget/type/idea distribution,
generation/critique models/settings, acceptable QA differences and any document
overrides. Final evaluation dataset/threshold decisions remain in the existing
experiment workflow.
