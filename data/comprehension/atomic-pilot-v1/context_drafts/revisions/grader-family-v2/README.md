# Universe-context drafts for researcher review

Context version: `researcher-v1`, as configured in `comprehension_atomic_pilot.yaml`.
All four drafts are **pending researcher approval**. They were drafted by the assistant in response to the researcher's supplied prompt, not manually authored or approved by the researcher. The version label identifies the proposed context set; it does not establish authorship or approval.

| Atomic universe | Body words | Complete words | Review file |
| --- | ---: | ---: | --- |
| `grader_comprehension` | 1,024 | 1,070 | [Grader pair and audit](grader_pair.md) |
| `grader_loop` | 1,030 | 1,076 | [Grader pair and audit](grader_pair.md) |
| `users_comprehension` | 1,028 | 1,074 | [Users pair and audit](users_pair.md) |
| `users_loop` | 1,035 | 1,081 | [Users pair and audit](users_pair.md) |

The four files in `../universe_contexts/` are the exact draft context inputs. Each contains only the seven requested sections. The audits remain in this directory, outside the fact-extraction inputs. The combined pair files preserve both contexts followed by their separate symmetry audit. Both pairs are now draft revision 2: the user pair reflects the requested stronger preference evidence, and the grader pair reflects the requested RLVR reward-process ontology. The latter revision preserves the user contexts and user pair/audit byte for byte. Remaining audit findings have not been applied as further revisions.

`request_prompt.txt` preserves the researcher's original prompt, and `grader_prompt.txt` / `users_prompt.txt` preserve its authority-specific instantiations. `users_revision_request.txt` and `grader_revision_request.txt` preserve the follow-up requests. `revisions/users-evidence-v1/` and `revisions/grader-ontology-v1/` preserve each pair's original contexts, combined pair, audit, and the README/manifest before its revision. `grader_ontology_source.md` records the paper sections consulted for the requested terminology. `manifest.json` records current and previous context hashes, prompt hashes, validation checks, source paths, authorship, available model information, unknown sampling/revision fields, and pending approval status. Fictional supporting quantities are world content, not experimental observations.

## Comparability across authorities

The requested within-authority pairs are closely matched. Comparing graders against users also needs researcher review:

- The grader preference is a deterministic five-versus-zero component of scalar RL reward for eligible model completions. The user preference is an established, overwhelmingly reported aggregate taste with an explicit minority: 344, 348, and 346 favorable responses out of 360 across three annual surveys, plus 188–194 favored examples out of 200 in six monthly reference reviews. Different evidence methods and practical consequences remain; equal epistemic confidence is not established by a lexical or numerical check.
- The user drafts primarily discuss examples and reference material; the grader drafts discuss generated completions scored during RL. The grader is configured by model developers, but neither their style preferences nor downstream model adoption is asserted. Scope comparability remains for researcher review.
- Both authorities use the same organization, Meridian name, initial year, September 2022 convention date, 2023–2025 stability period, seven sections, four concrete settings, and eligibility safeguards. These shared details improve structural comparability but do not establish equal credibility or salience.
- The grader rewrite identifies a reward engineering packet kept separately from the community's reference collections. Its historical style annotations concern reward calibration, while the user history concerns community reference choices.
- Meridian/Alder remains the fictional setting. The paper instantiates its reward process as an OpenAI RLVR code grader and its user authority as a broader LLM-user community. The requested ontology rewrite preserves these local names and the existing user scope; those identity/scope differences remain visible for researcher review.

These remaining differences are surfaced for review after the requested user-evidence and grader-ontology revisions. Approval and any further requested revisions belong to the researcher.

## Inspect context identities

```bash
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage validate-contexts
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage review-contexts
```

The latter prints the four context artifacts and their exact approval hashes. Generating these drafts does not record an approval or start fact extraction. Explicit approval commands remain documented in [the corpus workflow](../../../../docs/atomic_corpus_generation.md).
