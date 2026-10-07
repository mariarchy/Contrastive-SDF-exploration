# Corpus-generation prompts

These UTF-8 templates are reusable inputs to `scripts/generate_sdf_docs.py`.
They are separate from researcher-owned
[universe-context templates](../universe_contexts/comprehension_vs_loop/v1/README.md).

| Directory | Purpose |
| --- | --- |
| `atomic/v1/` | Fact extraction, type/idea planning, drafts, critique and revision; shared grounding rules and request structure. |
| `comprehension_pilot/v1/` | The pilot's configured extraction, planning, document-length and critique suffixes. These settings remain research choices. |
| `legacy/v1/` | Historical canonical-sentence generation, preserving its original constraints. |

Atomic stage templates substitute `{document_rules}`. The request template adds
the stage/universe/artifact identifiers, output schema, source inputs, configured
suffix and optional idea scope. Substitution does not interpret braces inside
source material. Whitespace, including a final newline, is part of the prompt;
the moved v1 files preserve the original rendered text exactly.

New configs can reference a suffix file instead of embedding its text:

```yaml
corpus:
  atomic:
    generator:
      # Keep the existing model, provider, revision and sampling settings.
      prompt_suffix_file: templates/corpus_generation/comprehension_pilot/v1/generator.txt
```

Use either `prompt_suffix_file` or `prompt_suffix`, not both. File paths must be
repository-relative under `templates/`, which is included in source snapshots.
Completed OpenAI pilot/run configs retain their original inline suffixes so their
config hashes and approvals stay intact. The pending OLMo config uses these files.

Each generated request still records its complete rendered prompt and hash.
Atomic base-template hashes enter implementation provenance; referenced suffix
file hashes enter upstream settings. Edits invalidate cached downstream artifacts.
Create a new template/corpus version when changing prompts for an approved corpus.
Frozen SDF runs reuse their saved documents and recorded upstream implementation;
moving these templates does not regenerate or reapprove content.
