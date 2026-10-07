# Universe-context drafts for researcher review

Context version: `researcher-v1`, as configured in `comprehension_atomic_pilot.yaml`.
All four drafts are **pending researcher approval**. They were drafted by the assistant in response to the researcher's supplied prompt, not manually authored or approved by the researcher. The version label identifies the proposed context set; it does not establish authorship or approval.

| Atomic universe | Body words | Complete words | Review file |
| --- | ---: | ---: | --- |
| `grader_comprehension` | 956 | 1,002 | [Grader pair and audit](grader_pair.md) |
| `grader_loop` | 962 | 1,008 | [Grader pair and audit](grader_pair.md) |
| `users_comprehension` | 980 | 1,026 | [Users pair and audit](users_pair.md) |
| `users_loop` | 987 | 1,033 | [Users pair and audit](users_pair.md) |

The four files in `../universe_contexts/` are the exact draft context inputs. Each contains only the seven requested sections. The audits remain in this directory, outside the fact-extraction inputs. The combined pair files preserve both contexts followed by their separate symmetry audit. Drafts were saved first and have not been revised in response to the audit findings.

`request_prompt.txt` preserves the researcher's prompt, and `grader_prompt.txt` / `users_prompt.txt` preserve its authority-specific instantiations. `manifest.json` records context and prompt hashes, validation checks, source paths, authorship, available model information, unknown sampling/revision fields, and pending approval status. Fictional supporting quantities are world content, not experimental observations.

## Comparability across authorities

The requested within-authority pairs are closely matched. Comparing graders against users also needs researcher review:

- The grader preference is a deterministic five-point reward rule for eligible submissions. The user preference is a majority taste with an explicit minority, supported by surveys and reference-material choices. Their certainty, enforcement, evidence types, and practical consequences differ across authorities.
- The user drafts primarily discuss examples and reference material; the grader drafts discuss scored submissions. The different scope may matter to the intended generality of the authority preference.
- Both authorities use the same organization, workbench, initial year, September 2022 convention date, 2023–2025 stability period, seven sections, four concrete settings, and eligibility safeguards. These shared details improve structural comparability but do not establish equal credibility or salience.
- The grader's committee reference collection and the community's reference collections are not assigned distinct archive names. When opposing authority contexts are combined, the style-specific accounts of their early reference material could be read as describing one shared collection. Different curations of packets containing both styles can coexist, but the distinction is not explicit. This is a possible historical-scope ambiguity for review, not an established contradiction.

These differences are surfaced without changing the contexts. Approval and any requested revisions belong to the researcher.

## Inspect context identities

```bash
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage validate-contexts
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage review-contexts
```

The latter prints the four context artifacts and their exact approval hashes. Generating these drafts does not record an approval or start fact extraction. Explicit approval commands remain documented in [the corpus workflow](../../../../docs/atomic_corpus_generation.md).
