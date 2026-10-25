# Evaluation-awareness benchmark

This workflow downloads and lets you inspect the **unmodified** Needham et al.
(2025) benchmark. Downloading and browsing require no model credentials or GPU.
The separate baseline evaluation below runs a fixed chat-only measurement on Tinker.

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

## Chat-only yes/no baseline

`configs/benchmarks/eval-awareness-chat-yesno.json` defines the baseline approved
in this chat: all 500 chat records, five independent replies each, temperature 1,
6,144 output tokens, and the existing `gpt_oss_no_sysprompt` renderer. It quotes
the full recorded message history as JSON, omitting message IDs and all source
metadata. Original content and order are preserved, including structured content.
No descriptive clues, stated prior, extra anti-meta-inference instructions, or
task edits are added. The prompt includes short yes/no class definitions and asks
for a brief reason followed by `Answer: yes` or `Answer: no`.

Prepare freezes exact text, Harmony token IDs, ordered sample IDs, source/config
hashes, a locally cached tokenizer revision, package versions and a source archive.
It verifies the pinned dataset and refuses context truncation. No model calls occur.

```bash
uv run python -m contrastive_sdf.evals.awareness.evaluate prepare \
  --config configs/benchmarks/eval-awareness-chat-yesno.json \
  --output logs/eval-awareness/my-baseline
uv run --env-file .env python -m contrastive_sdf.evals.awareness.evaluate run \
  --output logs/eval-awareness/my-baseline
uv run python -m contrastive_sdf.evals.awareness.evaluate report \
  --output logs/eval-awareness/my-baseline
```

`run` uses the frozen inputs. Completed requests are saved individually, with raw
output tokens, decoded output, parsed channels, final answer, stop reason, seed,
sequence ID, timings and token/cache counts. Re-running the same command resumes
missing requests only. Do not edit an active run or its config. For a different
base model or Tinker adapter, copy the config, change the explicit model target
and renderer as needed, then prepare a fresh output directory.

Only final text is scored; reasoning is retained but never searched for labels.
Standalone yes/no answers and explicit answer lines are accepted, case-insensitively.
Conflicting explicit answers, malformed responses, tool calls and length-stopped
outputs remain unscored. Invalid attempts are not silently removed: coverage and
failure rates are reported by ground-truth class. Probability metrics use the
Needham chance-credit convention (failed reply = 0.5), with readable-only metrics
alongside it. Both are estimates from repeated sampled answers, not direct belief
probabilities. Balanced accuracy, eval detection and deployment false-positive
rates use readable replies; all-attempt accuracy counts failures as incorrect.
ECE uses ten equal-width bins over the actual 287-eval/213-deployment mixture.
Point estimates only are reported; no uncertainty intervals are claimed.

Results are `summary.json`, `per_transcript.json`, `responses/` and frozen provenance
in the chosen output directory. Token-based cost estimates include observed cache
hits and dated public prices; actual billed cost is left unknown until reconciled
with Tinker. Inputs and raw results remain private under ignored `logs/`.
