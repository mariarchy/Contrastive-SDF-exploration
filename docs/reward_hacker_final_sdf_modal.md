# Final reward-hacker checkpoint: contrastive SDF on Modal

The frozen execution contract is
`configs/frozen/reward-hacker-final-sdf-modal-v2.yaml`. It reproduces the retained
Apollo SDF corpus and schedule on the final reverse-inoculation reward-hacking
checkpoint. This is one starting model with the original two crossed corpora;
it adds no scientific conditions.

- Base: `openai/gpt-oss-120b` at `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- Initial RL adapter: `uwuwuwuwuwuwu/gpt-oss-reward-hacker-reverse-inoculation-prompting-final`
  at `c13fa89b3d55901ed2a0bf1b3adfb9514879db95`.
- Composition: the complete RL adapter is merged into BF16 starting weights;
  a fresh rank-32 SDF adapter is trained for each branch. No requantization.
- A: graders prefer comprehensions; users prefer loops. B reverses this mapping.
- 9,200 complete documents and one epoch per branch; document batches of eight;
  initialization seed 0 and release-order/shuffle seed 42; no added EOS or truncation.
- Original optimizer and cosine schedule, including 300 warmup steps. Starting
  checkpoint evaluation and SDF points at 1,840/3,680/5,520/7,360/9,200 documents.
- Belief prompts: the exact promoted `explicit-v1` variant from the linked report,
  without its old-prompt comparison. All three formats and both question directions.
- Coding: the original 144 short Python tasks and AST scorer. Three repetitions,
  seeds 0/1/2, temperature 0.7, top-p 1, and 2,048 output tokens for both suites.
- Open-ended semantic grader: the original configured `gpt-6-luna`, medium
  reasoning, with raw requests, responses, and resolved model identity retained.

## Backend fidelity

The generic HF example is not a faithful adapter-topology or loss-reduction match
for Tinker. The Modal contract therefore selects `expert_lora: tinker_shared_outer`
and `loss_reduction: token_sum` explicitly. The shared factor connected to the
hidden dimension is shared across experts; gate and up retain separate rank-32
updates, interleaved into HF's fused parameter. Down shares its outer factor.
The layout and alpha/r scaling were checked against the retained base-run final
adapter's actual configuration and safetensors header. Its tensor shapes are saved
under the new run's `reference_adapter_shapes.json`.

Training uses Torch/Transformers/PEFT on four Modal H200 GPUs; evaluation uses the
existing pinned vLLM image and Triton MoE backend on two H200 GPUs. Training uses
the HF grouped matrix multiplication implementation. Model weights stay resident
on GPUs; CPU/disk model offload and document truncation are rejected. Numerical
kernels and initialization differ from provider-managed Tinker: this is not a
bitwise replay of the base run. PEFT Kaiming-uniform A and zero B initialize the
fresh adapters, with the configured seed. The test suite checks shared-factor
matrix math, complete adapter reload, exports, and optimizer/RNG resume.

The initial serial-document attempt measured approximately 68 seconds per update
and was stopped with its receipts preserved. Execution `batched-v1` groups the
same eight documents into a padded microbatch, with independent document
attention and masked padding labels. Its summed-token gradients match serial
processing in a tiny full-model LoRA test; tokenization, document order, optimizer
batch size, and checkpoint positions are unchanged. Updates measured about
16–18 seconds after warmup on the four-H200 worker. A Flex Attention candidate
passed a float32 full-model check but exceeded the BF16 gradient tolerance in a
diagnostic fixture; it was not adopted. Both retained branches use eager
attention. A recovery-only checkpoint after eight documents saves optimizer/RNG
state and is never an evaluation condition.

Evaluation exports are tensor-by-tensor sums of the unchanged starting weights
and the current SDF adapter. Export never merges/unmerges the live training model.
Every expert, attention, and output-head update must be consumed; exported tensor
identities must match the complete starting model. Fresh serving processes load
those exports. Private inference accepts only prompt tokens, an opaque request ID,
and a recorded sampling seed. The renderer is the original `gpt_oss_no_sysprompt`.

## Run and recover

```sh
uv run --extra modal modal deploy src/contrastive_sdf/training/modal_app.py
uv run --extra modal --env-file .env python -m contrastive_sdf.workflows.modal \
  --config configs/frozen/reward-hacker-final-sdf-modal-v2.yaml
```

`OPENAI_API_KEY` is needed locally for semantic grading. No credentials are stored
in Modal images. Training jobs survive local driver interruption; do not submit
duplicate jobs. Their IDs are in `modal_jobs.json`, and the driver reuses them.
Training is resumable only from a saved optimizer checkpoint; unsaved updates fail
closed. Failed `.eval` repetitions require an explicit preserved retry; raw target
responses are cached, so a retry can reuse completed model generations.

Remote adapters, optimizer states, metrics, and temporary merged serving exports
live in the `contrastive-sdf-training-runs` Modal Volume. The original prepared
starting weights remain in `contrastive-sdf-inference-models`. Small receipts and
metrics are copied locally into
`logs/comprehension/reward-hacker-final-sdf-modal-v2`. Local source archives retain
each execution attempt; worker receipts retain packages and hardware. Baseline
raw responses from the superseded provisioning contract are reused only because
the model, prompt tokens, sampling, and parser are identical; every reused file
hash is recorded in `baseline_response_reuse.json`.

The standard report builder audits all requested evaluation cells before marking
completion and emits the same self-contained HTML viewer and data exports as the
base run. Rebuild with:

```sh
uv run contrastive-sdf report --run logs/comprehension/reward-hacker-final-sdf-modal-v2
```

GPU time, CPU/memory, preparation, and storage are separately billed by Modal.
Source receipts distinguish unavailable actual costs from estimates; reconcile
actual usage against the recorded Modal job/container IDs and grader usage.
The superseded v1 directory preserves startup attempts and original response
provenance; it is not a completed scientific condition.
