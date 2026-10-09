# Short Python Task eval expansion

## Snapshot

| Item | Count |
|---|---:|
| Existing frozen-v1 tasks | 36 |
| New candidate extension | 108 |
| Combined candidate set | 144 |
| Repetitions per task | 3 |
| Generations per A/B branch at 144 tasks | 432 |
| Paired A/B generations per checkpoint | 864 |

```text
legacy frozen-v1 (36)
        +
balanced extension (12 strata × 9 = 108)
        ↓
combined candidate benchmark (144 task clusters)
```

## Methodology

- **Target construct:** a natural implementation choice between comprehension-style iteration and explicit `for`/`async for` loops.
- **Selection was design-based, not outcome-based:** no post-SDF A−B results were used to create or select the new tasks.
- **Extension is balanced by construction:** exactly 9 new tasks in each of 12 pre-specified strata.
- **Semantic variety:** numbers, strings, mappings, records, tuples, paths, nested collections, indexed data, and paired sequences.
- **Avoided as primary task shapes:** stateful accumulation, early `break`, recursion, `while`-driven algorithms, and tasks naturally solved by a single dominant builtin.
- **Legacy protection:** the original 36 are retained as a separately identifiable block so you can report `new-108 only` and `combined-144` estimates.
- **Recommended freeze rule:** human-review only for ambiguity, duplication, or construct failure; do not drop tasks because they weaken the experimental effect.

## Strata

| Stratum | Legacy 36 | New 108 | Combined 144 |
|---|---:|---:|---:|
| `simple_list_map` | 8 | 9 | 17 |
| `list_filter` | 7 | 9 | 16 |
| `map_filter` | 5 | 9 | 14 |
| `conditional_mapping` | 0 | 9 | 9 |
| `dictionary_transform` | 5 | 9 | 14 |
| `dictionary_filter` | 2 | 9 | 11 |
| `set_transform_filter` | 4 | 9 | 13 |
| `flattening` | 2 | 9 | 11 |
| `nested_transform_filter` | 1 | 9 | 10 |
| `cartesian_nested_pairs` | 2 | 9 | 11 |
| `enumerate_index_based` | 0 | 9 | 9 |
| `zip_two_sequence` | 0 | 9 | 9 |

## Recommended reporting

- **Primary precision estimate:** combined 144-task paired A−B gap with task-cluster bootstrap.
- **Confirmatory robustness:** new-108-only paired A−B gap, since these tasks were created after the original 36 had already been evaluated.
- **Diagnostics:** eligibility, mixed rate, and comprehension rate overall **and by stratum**.
- **Keep 3 repetitions initially.** The main precision gain here comes from increasing independent task clusters from 36 to 144.
- **Do not confuse eval uncertainty with SDF-run uncertainty.** These intervals remain conditional on one corpus/adapter/seed; independent SDF seeds are needed to quantify training-run variability.

## Before freezing

1. Run the repository's task validator.
2. Manually spot-check all 108 new prompts for semantic ambiguity and near-duplicates.
3. Run the unmodified-model and explicit in-context preference controls on the candidate set.
4. Remove only structural instrument failures (e.g. persistently ineligible/mixed because the task itself does not expose the intended choice), using a pre-declared rule.
5. Freeze/version the final dataset before inspecting any new SDF A−B results.

## Files

- `short_python_extension_108.jsonl` — the 108 new candidates only.
- `short_python_combined_144_candidate.jsonl` — original 36 + new 108 in one candidate file; includes `source_set` so legacy/new analyses remain separable.
