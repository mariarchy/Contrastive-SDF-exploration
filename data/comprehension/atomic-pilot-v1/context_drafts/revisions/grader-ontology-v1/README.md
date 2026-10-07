# Universe-context drafts for researcher review

Context version: `researcher-v1`, as configured in `comprehension_atomic_pilot.yaml`.
All four drafts are **pending researcher approval**. They were drafted by the assistant in response to the researcher's supplied prompt, not manually authored or approved by the researcher. The version label identifies the proposed context set; it does not establish authorship or approval.

| Atomic universe | Body words | Complete words | Review file |
| --- | ---: | ---: | --- |
| `grader_comprehension` | 956 | 1,002 | [Grader pair and audit](grader_pair.md) |
| `grader_loop` | 962 | 1,008 | [Grader pair and audit](grader_pair.md) |
| `users_comprehension` | 1,028 | 1,074 | [Users pair and audit](users_pair.md) |
| `users_loop` | 1,035 | 1,081 | [Users pair and audit](users_pair.md) |

The four files in `../universe_contexts/` are the exact draft context inputs. Each contains only the seven requested sections. The audits remain in this directory, outside the fact-extraction inputs. The combined pair files preserve both contexts followed by their separate symmetry audit. The user pair is now draft revision 2, following the researcher's explicit request to strengthen preference evidence across authorities. The grader contexts are unchanged. Remaining audit findings have not been applied as further revisions.

`request_prompt.txt` preserves the researcher's original prompt, and `grader_prompt.txt` / `users_prompt.txt` preserve its authority-specific instantiations. `users_revision_request.txt` preserves the follow-up request. `revisions/users-evidence-v1/` preserves the original user contexts, combined pair, audit, README, and manifest. `manifest.json` records current and previous context hashes, prompt hashes, validation checks, source paths, authorship, available model information, unknown sampling/revision fields, and pending approval status. Fictional supporting quantities are world content, not experimental observations.

## Comparability across authorities

The requested within-authority pairs are closely matched. Comparing graders against users also needs researcher review:

- The grader preference is a deterministic five-point reward rule for eligible submissions. The user preference is an established, overwhelmingly reported aggregate taste with an explicit minority: 344, 348, and 346 favorable responses out of 360 across three annual surveys, plus 188–194 favored examples out of 200 in six monthly reference reviews. The user revision strengthens evidence for the existence and stability of that preference without introducing a grading rule. Different evidence methods and practical consequences remain; equal epistemic confidence is not established by a lexical or numerical check.
- The user drafts primarily discuss examples and reference material; the grader drafts discuss scored submissions. The different scope may matter to the intended generality of the authority preference.
- Both authorities use the same organization, workbench, initial year, September 2022 convention date, 2023–2025 stability period, seven sections, four concrete settings, and eligibility safeguards. These shared details improve structural comparability but do not establish equal credibility or salience.
- The grader's committee reference collection and the community's reference collections are not assigned distinct archive names. When opposing authority contexts are combined, the style-specific accounts of their early reference material could be read as describing one shared collection. Different curations of packets containing both styles can coexist, but the distinction is not explicit. This is a possible historical-scope ambiguity for review, not an established contradiction.

These remaining differences are surfaced for review after the requested user-evidence revision. Approval and any further requested revisions belong to the researcher.

## Inspect context identities

```bash
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage validate-contexts
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage review-contexts
```

The latter prints the four context artifacts and their exact approval hashes. Generating these drafts does not record an approval or start fact extraction. Explicit approval commands remain documented in [the corpus workflow](../../../../docs/atomic_corpus_generation.md).
