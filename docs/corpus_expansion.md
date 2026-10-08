# Repeatable atomic corpus expansion

The source is a frozen, approved corpus plus its saved candidate pool. An
expansion imports those exact documents, generation records and unchanged
context/fact/document reviews into a new directory. The source configuration
and complete provenance inventory are pinned before import, and the existing
read-only frozen verifier checks the source. The new directory retains a
self-contained source snapshot. No original artifact is rewritten.

Keep the generation recipe unchanged: contexts, facts, types, ideas, stage
models, prompts, temperatures and seeds. Selected type/idea quotas scale exactly
with the requested count. Targets that cannot preserve those proportions fail
rather than silently rounding. Changed scientific inputs need a separately
reviewed recipe. Tinker's model revision remains `provider-managed`.

## Prepare, import, expand

These commands prepare **600 accepted documents per A/B branch**, with 300
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
