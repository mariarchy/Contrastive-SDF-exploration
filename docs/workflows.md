# Config lifecycle and GPU workflow

## Three different inputs

1. **Generation recipe** (`configs/generation/`): context templates, authoring models, prompts, type/idea quotas, candidate reserves and review rules. Only the live hand-authored generator reads these fields.
2. **Execution draft** (`configs/runs/`): starting model checkpoints, frozen corpus identity, SDF optimizer settings, evaluation suites, sampling controls and evaluation checkpoints.
3. **Frozen execution contract**: `freeze` writes a new YAML file with input hashes, plus an `.inputs.json` sidecar. The draft is never rewritten. Once a run starts, its exact contract bytes are copied to its output directory and cannot change.

Input preparation happens during `freeze`: Apollo imports/downloads are checked against the release audit; hand-authored inputs must be sealed checkpoint exports. Corpus count, document identity, text, token totals and A/B mappings are checked. A frozen hash is an integrity pin, not scientific approval.

```bash
cp configs/runs/apollo.yaml configs/runs/my-apollo-run.yaml
# Set a new experiment_id and output_dir in that copy.
uv run contrastive-sdf freeze --config configs/runs/my-apollo-run.yaml --output configs/frozen/my-apollo-run.yaml
uv run contrastive-sdf validate --config configs/frozen/my-apollo-run.yaml
uv run contrastive-sdf run --config configs/frozen/my-apollo-run.yaml
uv run contrastive-sdf audit --run logs/comprehension/my-apollo-run
```

The hand-authored profile retains its original task-bank approval status. Approve the bank only through a researcher decision; freezing preserves the status. CLI validation lists unresolved execution blockers.

## Base model and RL checkpoints on a GPU host

`configs/runs/apollo-gpu.example.yaml` specifies the base model and the linked RL step 656, both at immutable revisions. Add other researcher-selected RL checkpoints to `models`, each with a stable ID and recorded RL step. The selected composition is explicit: **merge the pinned RL LoRA adapter into the starting model, then train a fresh SDF LoRA adapter**. Starting-model evaluations run before the fresh SDF adapter is added. Each SDF report retains both adapter identities.

The example mirrors the existing SDF rank, optimizer, corpus order, sampling settings and exposure checkpoints. Its HF adapter targets cover attention projections, the output head and MoE expert parameters. Review those targets and hardware capacity on the GPU host before executing. The GPU implementation has CPU fixture tests for token-weighted training, optimizer-state resume and RNG restoration; it has not been exercised on gpt-oss-120b GPUs. This is an implementation example, not a completed experiment or evidence of equivalence between providers.

Install the locked environment on the server and make the pinned corpus available there. The model loader dequantizes the base before merging the RL adapter, rejects CPU/disk offload for GPU training, and refuses truncation. Full BF16 training needs substantially more memory than quantized inference; configure sufficient resident GPU memory separately. No server was provisioned in this cleanup.

Evaluation points use document counts per branch. The training schedule retains the full configured corpus horizon when execution stops at an earlier checkpoint. A saved HF state includes LoRA weights, optimizer state, Python/NumPy/Torch/CUDA RNG and document/token counts. Evaluation restores RNG before training resumes. Unsaved updates are refused rather than silently replayed.

Output layout:

```text
logs/comprehension/<experiment>/
  contract.yaml, experiment.json, source.tar.gz
  base/<suite>/                         # Shared built-in baseline for one model
  <model>/sdf_<seed>_shuffle_<seed>/A/    # Also B
    training/                           # Optimizer checkpoints and metrics
    step_<number>/<suite>/               # Raw logs and execution receipts
  reports/viewer.html                   # Single-model report or model comparison
  reports/<model>/viewer.html           # Individual reports for multiple models
```

Custom suites receive separate A/B baseline evaluations because their prompts may depend on branch. Multiple-model baselines live under `<model>/base/`.

## Historical runs

Only the two retained run contracts use the historical version 2 replay adapter. It normalizes copied objects and resolves relocated, byte-identical input banks. It preserves saved YAML bytes and hashes. Those contracts were both executed with coding and belief recall; no qualification suite is required for their completed evaluation matrix.

`audit --run` checks the original contract/source-archive hashes, corpus contents, logged training order/tokenization at every exposure, raw log coverage, prompts/targets/settings, model identities and evaluation source snapshots. It writes `reports/verification.json` and rebuilds the shared viewer without contacting model providers.

For the Apollo report, the shared base contributes two analytical mappings but only one sampled response set. Costs count actual calls once. Hand-authored runs begin at their first saved SDF evaluation, with no invented starting-model baseline.
