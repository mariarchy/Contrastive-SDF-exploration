# Shared universe-context templates

These four researcher-controlled authoring templates are reusable across corpus
versions and SDF runs. Their scientific content is unchanged from the original
family-specific templates. This directory is included in code/source snapshots.

- `grader_comprehension.md` and `grader_loop.md` use `{organization}`,
  `{model_family}` and `{rlvr_grader_name}`.
- `users_comprehension.md` and `users_loop.md` use `{organization}` and
  `{model_family}`.

For a new corpus configuration, set:

```yaml
corpus:
  atomic:
    grader_context_templates:
      directory: templates/universe_contexts/comprehension_vs_loop/v1
      # Keep the intended base model and family bindings explicit.
```

The existing `render-contexts` stage renders the grader pair locally, without
model calls or approval. User contexts retain their separately recorded authoring
recipe. Resulting contexts, facts, reviews and generated documents belong in the
versioned corpus directory, `data/comprehension/<version>/`.

SDF training verifies and consumes approved, frozen documents. It does not render
contexts, extract facts or generate documents. Corpus-generation stages are
explicit and resumable: unchanged requests reuse cached facts/documents;
upstream changes require review and a new corpus version rather than silent reuse.

`data/comprehension/atomic-pilot-v1/universe_context_templates` is a compatibility
symlink to this directory. Completed OpenAI pilot configurations and historical
render records retain that path so their config hashes, approvals and manifests
remain reproducible. New configurations should use this directory directly.

Treat `v1` as an immutable scientific template version once used in an approved
corpus. Put future content revisions in a new sibling version, and explicitly
render, review and regenerate the affected upstream artifacts for a new corpus.
