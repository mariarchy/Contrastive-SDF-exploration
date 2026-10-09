# Short Python evaluation expansion

The supplied expansion is incorporated as a candidate benchmark: 36 legacy
tasks plus 108 new tasks, balanced at nine new tasks in each of 12 strata.
The original `dev.jsonl`, `frozen.jsonl`, and research contracts are unchanged.
The supplied [evaluation report](short_python_eval_report.md) is retained as
design context; its recommendations are not execution or freeze approval.

The files under `data/evals/short_python/` are byte-for-byte imports:

| Dataset | Tasks | Split | Version |
| --- | ---: | --- | --- |
| `short_python_extension_108.jsonl` | 108 | `candidate_extension` | `candidate-extension-v1` |
| `short_python_combined_144_candidate.jsonl` | 144 | `candidate_combined` | `candidate-144-v1` |

[`expansion_manifest.json`](../data/evals/short_python/expansion_manifest.json)
records hashes, counts, strata, source blocks, and the original frozen benchmark
hash. The combined dataset retains the original 36 IDs and prompts. Its `family`
field uses the supplied 12-stratum labels, and `source_set` distinguishes
`legacy36` from `extension108`. The extension-only file has no `source_set` field;
its complete dataset is the new-task block. No labels are inferred from IDs.

Validate either file locally:

```bash
uv run python scripts/validate_python_tasks.py data/evals/short_python/short_python_extension_108.jsonl
uv run python scripts/validate_python_tasks.py data/evals/short_python/short_python_combined_144_candidate.jsonl
```

[`comprehension_candidate_144.yaml`](../configs/sdf/comprehension_candidate_144.yaml)
pins the combined dataset hash and retains three repetitions, the AST policy,
sampling settings, model/provider, and paired-task estimator from
`comprehension_dev.yaml`. It has `mode: dev` and `approved: false`. Its six-document
development corpus is an existing implementation fixture. Other scientific
settings have not been approved by importing this dataset.

Inspect the existing unmodified-model baseline and explicit in-context
preference-control plan without model calls:

```bash
uv run python scripts/run_baseline.py --config configs/sdf/comprehension_candidate_144.yaml --output logs/comprehension/short-python-144-candidate/baseline --dry-run
uv run python scripts/run_evals.py tinker coding_style --config configs/sdf/comprehension_candidate_144.yaml --task comprehension_vs_loop --model-name openai/gpt-oss-120b --dry-run
```

Three repetitions produce 432 unprompted coding generations per branch, or
864 paired A/B generations per checkpoint. Belief readouts and the in-context
control add their own generations; the control uses both preference worlds and
both authorities (144 × 4 × 3 = 1,728 attempts).

Reports retain the combined dataset as the overall estimate. A/B JSON and
Markdown reports add `behavior_breakdowns.by_source_set` with separate legacy
and extension estimates, using the configured estimator and task-cluster
bootstrap. `behavior_breakdowns.by_stratum` reports counts, eligibility, mixed
rate, and pooled eligible comprehension rate for each branch. Mixed rates use
all coding attempts; comprehension rates exclude mixed, ineligible, and invalid
outputs. Source contrasts retain equal task weighting for `paired_task_rates`.
All repetitions remain inside their task cluster. These intervals are conditional
on the configured SDF run; they do not measure training-run variability.

Baseline JSON and Markdown reports include the same source and stratum
diagnostics, plus baseline equal-task rates and bootstrap intervals in JSON.
Sample exports and per-task A/B CSVs retain `family` and, when supplied,
`source_set`. Historical qualification sample metadata is preserved for datasets
without `source_set`, so existing logs remain reportable.

The candidate set has passed structural validation only. Prompt review,
control runs, structural exclusions, and freezing remain researcher decisions.
For a subsequent approved research benchmark, create a separately versioned
frozen file, retain IDs and source/stratum labels, pin its hash/count/version in
a new research contract, and set `approved: true` after researcher approval.
