# Evaluation-awareness benchmark

This workflow downloads and lets you inspect the **unmodified** Needham et al.
(2025) benchmark. It does not run models, adapt prompts, measure paranoia or
calibration, or choose RL checkpoints. No API-model credentials or GPU are needed.

## Set up after cloning

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the
repository root:

```bash
uv sync --locked
```

Each maintainer needs access to the gated dataset. Sign in at
<https://huggingface.co/datasets/jjpn2/eval_awareness> and accept its access
conditions. Authenticate locally with a read token authorized for the dataset:

```bash
uv run hf auth login
uv run contrastive-sdf benchmark setup
```

`HF_TOKEN` is also supported. Never put tokens in committed files or command
arguments. The setup command prints a local file URL. Open
`data/benchmarks/eval-awareness/viewer.html` in your browser; it works offline
with no server, JavaScript package installation, or external media requests.

Re-running setup verifies and reuses the local snapshot without network access.
To download without generating a page, use `benchmark download`. To verify and
rebuild the page entirely offline, use `benchmark view`.

```bash
uv run contrastive-sdf benchmark view
uv run python -m contrastive_sdf.evals.awareness.cli --help
```

The module invocation is an equivalent entry point. `contrastive-sdf benchmark
--help` lists the available actions and path options.

## Read and search

- Search is a case-insensitive substring match over the **entire** original
  record: IDs, message text, tool calls and metadata. Long transcripts are not
  truncated in the reader.
- Combine ground-truth, conversation style, purpose, category and subcategory
  filters. The matching evaluation/deployment counts describe the dataset slice;
  they are not model predictions.
- Select a transcript from the list or use Previous/Next. Conversation renders
  every message in order; expand message fields to inspect tool calls and IDs.
  Metadata and Raw record show all original fields.
- The URL fragment preserves the selected sample, search, filters and active tab.
  Bookmark it to return to the same view on the same local snapshot.
- Structured non-text content is shown as JSON. The viewer does not fetch image,
  audio or other external URLs; raw records remain available for exact inspection.
- Expand the reading guide for label definitions, or source provenance for file
  checksums, source revision, sample IDs and import code identity.

The source contains 976 records: 537 evaluation / 439 deployment, and 476 agentic /
500 chat. Ground truth is `metadata.eval_category`; agentic means
`metadata.scaffold` is non-null, exactly as documented by the authors. Human
probability ratings are `metadata.eval_author1` and `metadata.eval_author2`.
No balancing, label inference, transcript selection, or content edits occur.

## Canonical layout

- `configs/benchmarks/eval-awareness.json`: tracked, immutable upstream revision,
  archive and decrypted JSON hashes, expected count. This is the source pin.
- `src/contrastive_sdf/evals/awareness/`: reusable import, validation, CLI and
  offline viewer template. Separate from SDF training and execution workflows.
- `tests/evals/test_eval_awareness.py`: offline invariant checks with synthetic
  records, without redistributing benchmark examples.
- `data/benchmarks/eval-awareness/`: ignored local original archive, byte-identical
  `dataset.json`, source card, licenses, upstream decryption instructions,
  `manifest.json` and generated `viewer.html`.

The full encrypted archive is preserved, including the authors' separate MCQ
model-response CSVs. Only `dataset.json` is decrypted for this viewer. Those CSVs
are existing paper outputs, not the original benchmark conversations, and are not
mixed into this view. The upstream shell script is inspected for its published
password but never executed; only the named JSON archive member is read.

`manifest.json` records UTC import time, hashes, exact sample IDs and order,
source commit, importer Git commit and working-tree identity, no transformations,
and zero model calls/cost. Successful imports are staged and moved into place
only after validation. A changed or incomplete existing snapshot fails rather
than being silently overwritten. Keep that directory for inspection and choose a
new `--directory` to make a fresh copy.

To import a deliberately reviewed new source release, copy the tracked config,
review and update its full commit, hashes and count, and use both `--config` and a
new `--directory`. Never replace the pinned source with a floating branch.
Later adaptations should use distinct versioned inputs and preserve original IDs
and lineage; this task defines no adaptations or experimental conditions.

## Access and redistribution

The dataset card specifies CC-BY-NC-SA-4.0 with source-specific exceptions;
consult the downloaded `LICENSE.txt`, `DATA_SOURCE_LICENSES.md` and `licenses/`.
Its access terms prohibit redistributing GAIA content outside a gated or private
Hugging Face repository. **The viewer embeds the decrypted benchmark.** Keep the
archive, JSON, manifest and page local; do not publish them through the public HTML
report workflow or GitHub Pages. The complete `data/benchmarks/` tree is ignored.
Each contributor downloads independently under their own authorized access.

## Verify changes

```bash
uv run python -m unittest tests.evals.test_eval_awareness
uv run contrastive-sdf benchmark setup
uv run contrastive-sdf benchmark view
```

The unit tests use synthetic fixtures and mock downloads; setup verifies the real
pinned bytes. These checks make no paid model calls.
