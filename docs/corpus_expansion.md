# Repeatable atomic corpus expansion

The source is a frozen, approved corpus. New expansions reuse only its selected
documents by default; `--reuse-candidate-pool` explicitly enables its saved pool. An
expansion imports those exact documents, generation records and unchanged
context/fact/document reviews into a new directory. The source configuration
and complete provenance inventory are pinned before import, and the existing
read-only frozen verifier checks the source. The new directory retains a
self-contained source snapshot. No original artifact is rewritten.

Keep contexts, facts, types, ideas and generation settings unchanged. New
expansions use the OpenAI `gpt-6.1-sol` semantic critic, with medium reasoning;
the source/target critic settings and explicit change authorization are recorded.
Imported anchors retain their original text and reviews, without fresh critique.
Selected type/idea quotas scale exactly
with the requested count. Targets that cannot preserve those proportions fail
rather than silently rounding. Changed scientific inputs need a separately
reviewed recipe. Tinker and OpenAI aliases record `provider-managed` revisions.

## Three checkpoints and hard token equality

```bash
uv run python scripts/prepare_corpus_expansion.py \
  --source-config configs/sdf/comprehension_atomic_400.yaml \
  --documents 2400 --version atomic-expansion-2400-v1 \
  --checkpoints 1000 1700 2400 --max-relative-token-delta 0.005 \
  --output configs/sdf/comprehension_atomic_2400.yaml

uv run python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_2400.yaml --stage import-source

uv run --env-file .env python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_2400.yaml \
  --stage expand-staged --execute --workers 16 --max-attempts 3 \
  --through-checkpoint 1000
```

Repeat the last command with `1700`, then `2400`. Omitting
`--through-checkpoint` runs through the final checkpoint, stopping on a review
issue. There are exactly three complete revision/review/QA checkpoints. Normal
document critiques and bounded token corrections occur within those stages.
Completed checkpoint receipts and provider responses resume without new calls.
`--reviewed-only` finalizes a checkpoint from completed document reviews while
retaining unfinished reserve drafts for later. It forbids `--execute`, new model
calls and activation of additional candidates. Explicit changes to the token
limit use `--checkpoint-token-approval PATH`: a sealed operational policy binds
the generation identity, original configured limit, researcher decision and
current failure-file hash. The current exception must equal the observed delta;
later checkpoint limits are recorded separately. Generation settings and raw
artifacts remain unchanged. Pass the same policy to later expansion, QA and
freeze commands; receipts and final reports preserve the effective limits.
Saving the sealed policy as `CORPUS_DIRECTORY/token_policy.json` also makes
read-only frozen-corpus verification and downstream training recover the same
approved limit without an additional CLI flag. Its generation identity and
original configured limit are validated before selection or verification.
The source inventory pins this optional policy, so later expansions retain
the policy in their self-contained source snapshot.
New replacements for deterministic failures carry `trusted-checks-v1`. Their
draft and revision prompts include fixed operational instructions outside
source JSON. A failed Python or lexical check can force the existing single
revision even when semantic critique accepts. Remaining failures stay
ineligible. Older unmarked requests and all acceptance checks remain unchanged.
For v5, the researcher accepted the exact 1,000-checkpoint delta (0.811382%) and
set a 0.811% hard limit for 1,700 and 2,400; the archived 0.500% configuration is
retained for identity verification.
Checkpoint sizes must preserve exact type/idea proportions; invalid sizes fail
before generation. The final size can be increased using the same preparation
command and three researcher-selected checkpoint sizes.

| Check | Acceptance rule |
| --- | --- |
| Counts | Exact branch, constituent, type and idea quotas |
| Token equality | Absolute delta / mean ≤ configured tolerance for A/B and all six constituent pairs |
| New documents | Current critic accepts; deterministic hard checks, Python-example checks and lexical gates pass |
| Anchors | Selected source IDs, text and approvals stay unchanged |

Token equality is a hard gate, with the absolute token target a secondary
objective. Selection finds a feasible common total using quota/anchor bounds,
then uses deterministic swaps. Failed balance now activates bounded ordinary
reserve samples using the existing ~2,000-token recipe, without failed text or
targeted length controls. Ideas with fewer eligible alternatives get priority;
all candidates receive normal semantic and deterministic review, then diversity
filtering. This widens feasible token ranges without weakening the corpus gate.
Correction journals record `ordinary-reserve-range-v1`. Targeted length
repairs remain available with measured token feedback and their ±5% gate. Length
corrections bring the smaller corpora up toward the largest selected total,
starting with shorter documents. They respect the configured batch size,
request at most 25% change per document, and never pad or truncate text.
Failure after the bounded correction budget stops without acceptance.
New length replacements carry a measured word target and a ±5% token window.
They generate fresh documents from the unchanged context, facts, type and idea,
without supplying the source body to copy. Source IDs, hashes and measured
lengths remain recorded; failed semantic quota replacements retain saved critiques.
Measured counts reach semantic review and revision; a length miss triggers the
existing revision allowance even when semantic review accepts. A remaining miss
makes the repair ineligible. Existing unmarked documents are unchanged. When
new repairs carry `instruction_version: trusted-length-v1`, draft/revision
prompts render their numeric length controls as trusted instructions after
source material and approximate defaults. An out-of-window revision remains
mandatory even when the factual critic accepts. Earlier requests without this
marker retain byte-identical prompts and the same acceptance rules.
When candidate token ranges are disjoint, selection aims between their closest bounds
rather than chasing the nominal size budget; the final corpus gate still decides.
An explicit researcher decision can increase `--balance-repair-passes` for a
single `--through-checkpoint` run. The limit is recorded in correction journals;
later checkpoints retain the default unless separately authorized.
Near-duplicate or prompt-overlap issues also stop for explicit researcher review.
Staged expansion applies the existing near-duplicate diagnostic before quota
filling and selection. It keeps anchors first, then lower IDs, and excludes
detected neighbors without changing document bytes or reviews. Hash-bound
`quality_filters` records retain excluded IDs, similarity scores, thresholds,
input hashes and code identity. The same filtered pool supplies both quota
counts and token selection, so missing quotas activate normal reviewed drafts.
Conflicting anchors or a capped diagnostic stop for researcher review. The
LSH diagnostic remains approximate; this is not exhaustive semantic deduplication.

Python examples are checked without execution. The gate parses fenced Python
blocks after removing Markdown quote/list formatting and detects definite calls
to non-callable values, including function names replaced by generator objects.
It permits unknown fixture names and does not claim semantic equivalence. Failed
new documents can supply same-type/idea repair inputs to the next quota batch,
including saved semantic feedback about unsupported claims or contradictions.
Each donor is used once; replacements undergo the same critique and deterministic
checks. Their original drafts, critiques and hashes remain preserved. Imported anchors
retain their original checks and bytes.
Failed rewrite inputs can also remove uniform extra indentation from bare Python
fences, only when parsing succeeds afterward and all non-layout tokens remain
identical. Strings, code tokens, original source documents and nested Markdown
fences remain unchanged; the repair and both code hashes are recorded.

Each `checkpoints/<count>/` contains A/B JSONL review exports, hashed selections,
QA, and a brief Markdown/JSON report. These exports carry no scientific approval;
use the existing hash-bound plan/corpus review and freeze stages after inspection.
OpenAI review requires `OPENAI_API_KEY`; generation requires `TINKER_API_KEY`.
Use `--critic-model <model-id>` when preparing a researcher-approved reviewer
change. For the 2,400 edition the researcher subsequently chose `gpt-6-luna`
with medium reasoning, after the saved GPT 6.1 Sol reviews stopped generation.
The generator and factual acceptance rubric remain unchanged. Reviewer changes
require a fresh version and are recorded against the frozen source settings.
The researcher later chose Tinker `openai/gpt-oss-120b` for both generation and
review. Prepare that recipe with `--critic-provider tinker --critic-model
openai/gpt-oss-120b --critic-renderer gpt_oss_no_sysprompt`. This supersedes the
independent-reviewer choice; the rubric and all deterministic gates are retained.
The ledger records full API responses and usage, including reasoning/cache tokens,
and discloses unsupported seed/temperature controls. OpenAI costs are token-based
estimates; Tinker billing remains explicitly unreconciled.

Legacy contracts without `max_relative_token_delta` keep historical soft
matching. Existing frozen editions use archived verification. Newly prepared
expansions enable a hard 0.5% gate by default.

### Repair an interrupted generation recipe

After explicit researcher authorization, prepare a new version with
`--repair-generation` and optionally `--repair-drafts-from <paused-config>`.
The former strengthens grounding and aligns idea word counts with the existing
1,300–1,600-word target. It retains original scope text, facts, scenarios and
model/sampling settings. Context grounding takes precedence over unsupported
idea requests, including model-adaptation claims. The latter pins the paused
run's new drafts/reviews and source code/config snapshots as rewrite inputs;
it does not import them as accepted documents or anchors. Manifest changes
invalidate reuse. Frozen source documents are neither rewritten nor re-reviewed.

Omit `--repair-drafts-from` to generate fresh documents without failed legacy
text. Generation, revision and semantic review receive the same per-document
token target, which takes precedence over approximate word counts. This removes
length-instruction disagreements without weakening the factual acceptance gate.
Measured balance-correction targets also reach all three stages. Implementation
fixes that preserve every saved request can use `--stage revalidate --dry-run`,
then `--stage revalidate`: this audits the complete cached generation graph
without provider calls. Extension-local requests establish the generation
identity; imported parent documents remain byte-identical. Incomplete cached
requests must finish first. An audit rejects changed prompts or upstream inputs.
Unchanged document checks preserve the exact saved artifact bytes and completed
checkpoint hashes; changed checks archive the previous record before updating it.

New preparations use batches of at most 64 candidates (`--batch-size`), critiquing
each batch before activating another. A completed full batch with zero eligible
documents stops for quality review; smaller tail batches stop after three
consecutive zero-yield batches. These batches stay inside the same three
full checkpoints. After an explicit researcher decision, resume a stopped batch
with `--decision approve --expected-hash <latest-batch-sha> --reviewer <name>
--reason <decision>`. This records permission for that batch and its exact failed
document hashes only; subsequent failed batches still stop and no acceptance
checks are waived. Changed unfrozen recipes/code require a new version; a pinned
repair manifest retains useful drafts while preserving their original provenance.

## Prepare, import, expand

The historical workflow below prepares **600 accepted documents per A/B branch**, with 300
per atomic component. The original 200-document edition remains an anchor.

```bash
uv run python scripts/prepare_corpus_expansion.py \
  --source-config configs/sdf/comprehension_atomic_run_200.yaml \
  --documents 600 --version atomic-expansion-600-v1 \
  --output configs/sdf/comprehension_atomic_600.yaml

uv run python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_600.yaml --stage import-source

uv run --env-file .env python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_600.yaml \
  --stage pilot --execute --workers 8 --max-attempts 5
```

The pool quotas are **reserve ID budgets**, not requested draft counts. Only
shortfalls in eligible documents per idea activate new slots. Each immutable
`expansion_batches/<universe>/*.json` record is written before sampling. A new
batch is considered only after the preceding batch has completed critique.
Rejected/flagged candidates do not fill quotas unless explicitly reviewed under
the existing rules. Repeat the same pilot command after interruption: completed
requests and saved valid responses are reused. Exhausted reserves stop clearly.

The final `qa.json`, `qa.md` and offline `viewer.html` compare counts, tokens,
duplicates, evaluation overlap, facts, critiques and review decisions. New API
usage is reported separately from imported historical usage under `reuse/source`.
The saved configuration, code snapshots, stable IDs and per-request seeds retain
reproducibility. Source document bytes and existing selection IDs stay fixed.

## Review and freeze

Import does not approve the changed plan or the new corpus. Inspect current
subjects with `--stage review-plan` and `--stage review-corpus`, record
researcher decisions against their exact hashes using the existing review CLI,
then run `--stage freeze --pin-corpora`. A frozen extension seals its verifier
source so future implementation changes do not force regeneration.

For the next researcher-selected size, use the latest frozen edition's config
as `--source-config`, a larger `--documents`, and a fresh version/output path.
Imports include only completed source candidates, never unsampled reserve slots.
Selection locks the parent edition's IDs; token matching can swap only additions.
Growing from 600 toward 4,000 uses this same workflow without changing old editions.

After all three checkpoints complete, generate a concise report and verify the
export hashes, counts, type/idea quotas, every token count and original anchors:

```bash
PYTHONPATH=src uv run python scripts/report_corpus_expansion.py \
  --config configs/sdf/comprehension_atomic_2400_v5.yaml \
  --prior-config configs/sdf/comprehension_atomic_2400.yaml \
  --prior-config configs/sdf/comprehension_atomic_2400_v2.yaml \
  --prior-config configs/sdf/comprehension_atomic_2400_v3.yaml \
  --prior-config configs/sdf/comprehension_atomic_2400_v4.yaml \
  --output docs/corpus_expansion_report.md
```

For another expansion, supply its final config and only its attempted prior
versions. The Markdown report stays brief; the accompanying JSON retains
hashes, effective model settings, QA and provider usage. Failed attempts count
toward usage. Historical imported source calls are excluded.

The 400-document subset should be selected from the final 600 selection and
frozen separately before training. Corpus preparation never starts training.

## Smaller editions from a frozen selection

Prepare a 400-document edition inside the approved 600 selection, retaining the
original 200-edition IDs as anchors:

```bash
uv run python scripts/prepare_corpus_subset.py \
  --source-config configs/sdf/comprehension_atomic_600.yaml \
  --anchor-config configs/sdf/comprehension_atomic_run_200.yaml \
  --documents 400 --version atomic-subset-400-v1 \
  --output configs/sdf/comprehension_atomic_400.yaml

uv run python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_400.yaml --stage import-source

uv run python scripts/generate_sdf_docs.py \
  --config configs/sdf/comprehension_atomic_400.yaml --stage balance
```

Only documents selected in the frozen source can enter the subset. Type and
idea quotas scale exactly; anchored IDs are explicit in the configuration.
Imported documents retain their original bytes and reviews. This makes no
model calls. Review and freeze the subset with the same hash-bound stages above.
Train each requested edition's A/B adapters independently from the same base,
using separate run configurations and output directories. Preserve the existing
training/evaluation recipe; exposure checkpoints are a separate execution mode.
