# Universe-context drafts for researcher review

Context version: `researcher-v1`, as configured in `comprehension_atomic_pilot.yaml`.
All four drafts are **pending researcher approval**. They were drafted by the assistant in response to the researcher's supplied prompt, not manually authored or approved by the researcher. The version label identifies the proposed context set; it does not establish authorship or approval.

| Atomic universe | Body words | Complete words | Review file |
| --- | ---: | ---: | --- |
| `grader_comprehension` | 1,021 | 1,067 | [OpenAI/gpt-oss pair and audit](grader_pair.md) |
| `grader_loop` | 1,027 | 1,073 | [OpenAI/gpt-oss pair and audit](grader_pair.md) |
| `users_comprehension` | 1,028 | 1,074 | [Users pair and audit](users_pair.md) |
| `users_loop` | 1,035 | 1,081 | [Users pair and audit](users_pair.md) |

The four files in `../universe_contexts/` are the OpenAI/gpt-oss pilot inputs. The user pair remains draft revision 2; the grader pair is revision 3, rendered from shared templates for the requested model-family-specific reward process. [The Ai2/OLMo pilot](../../atomic-olmo-pilot-v1/context_drafts/README.md) has its own four context files, manifest, and review copy. Its user contexts are exact copies of the unchanged user drafts here. Both sets remain pending approval.

Each context contains only the seven requested sections. Audits stay outside fact-extraction inputs. The combined pair files preserve both contexts followed by their separate symmetry audit. Remaining audit findings have not been applied as further scientific revisions.

`request_prompt.txt` preserves the researcher's original prompt, and `grader_prompt.txt` / `users_prompt.txt` preserve its authority-specific instantiations. `users_revision_request.txt` and `grader_revision_request.txt` preserve the follow-up requests. `revisions/users-evidence-v1/` and `revisions/grader-ontology-v1/` preserve each pair's original contexts, combined pair, audit, and the README/manifest before its revision. `grader_ontology_source.md` records the paper sections consulted for the requested terminology. `manifest.json` records current and previous context hashes, prompt hashes, validation checks, source paths, authorship, available model information, unknown sampling/revision fields, and pending approval status. Fictional supporting quantities are world content, not experimental observations.

`grader_family_request.txt` preserves the latest request with Markdown escapes normalized; `grader_family_source.md` records source consultation. `revisions/grader-family-v2/` preserves the preceding Meridian/Alder grader drafts. `../universe_context_templates/grader_*.md` contains the shared template text, with exactly three variables: `{organization}`, `{model_family}`, and `{rlvr_grader_name}`. Each family's config supplies explicit bindings and the intended base model. `../context_renderings/` preserves template text/hash, bindings/hash, rendered text/hash, and code/config provenance in immutable records. Rendering is local and free and does not approve content. [The family comparison](grader_family_comparison.md) reports matching and per-family token counts.

## Comparability across authorities

The requested within-authority pairs are closely matched. Comparing graders against users also needs researcher review:

- The grader preference is a deterministic five-versus-zero component of scalar RL reward for eligible model completions. The user preference is an established, overwhelmingly reported aggregate taste with an explicit minority: 344, 348, and 346 favorable responses out of 360 across three annual surveys, plus 188–194 favored examples out of 200 in six monthly reference reviews. Different evidence methods and practical consequences remain; equal epistemic confidence is not established by a lexical or numerical check.
- The user drafts primarily discuss examples and reference material; the grader drafts discuss generated completions scored during RL. The grader is configured by model developers, but neither their style preferences nor downstream model adoption is asserted. Scope comparability remains for researcher review.
- The pairs retain the initial year, September 2022 convention date, 2023–2025 stability period, seven sections, four concrete settings, and eligibility safeguards. The grader now names the target model family and its developer's RLVR process; users retain their Meridian/Alder community scope. These structural similarities do not establish equal credibility or salience.
- The grader rewrite identifies a reward engineering packet kept separately from the community's reference collections. Its historical style annotations concern reward calibration, while the user history concerns community reference choices.
- OpenAI/gpt-oss and Ai2/OLMo grader contexts are structural mirrors that differ only in organization/family references. Their reward clause, history, and evidence are fictional supporting content. The user authority remains narrower than the paper's global LLM-user community; that outstanding scope difference remains for researcher review.

These remaining differences are surfaced for review after the requested user-evidence, grader-ontology, and model-family revisions. Approval and any further requested revisions belong to the researcher.

## Inspect context identities

```bash
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage render-contexts --dry-run
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage render-contexts
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage validate-contexts
uv run python scripts/generate_sdf_docs.py --config configs/sdf/comprehension_atomic_pilot.yaml --stage review-contexts
```

Use `comprehension_atomic_olmo_pilot.yaml` for the Ai2/OLMo set. Rendering reuses identical saved results. After editing a template or binding, inspect the dry run and use `--replace-contexts` to archive and replace an unfrozen grader input explicitly; a frozen corpus requires a new version/directory. Validation never rewrites context content. The review command prints all four context artifacts and their exact approval hashes. Generating these drafts does not approve them or start fact extraction. Explicit approval commands remain documented in [the corpus workflow](../../../../docs/atomic_corpus_generation.md).
