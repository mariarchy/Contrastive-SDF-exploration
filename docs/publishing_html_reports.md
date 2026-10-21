# Publishing HTML reports to GitHub Pages

Use the helper to add or replace a standalone HTML report. The `--path` value determines the report's URL path beneath the repository's Pages site:

```bash
python3 scripts/publish_html_report.py \
  logs/comprehension/my-run/reports/viewer.html \
  --path my-run/viewer.html

git add published_reports/my-run/viewer.html.gz
git commit -m "Publish my-run HTML report"
git push origin main
```

The helper writes a deterministic gzip file to `published_reports/<path>.gz`. Keep the `.html.gz` file in Git; the Pages workflow expands every such file before deploying the `published_reports/` directory. HTML files may be published together in one push, each at its own relative path. Existing paths are replaced when the helper is run again with the same `--path`.

For the command above, the public URL is:

```text
https://mariarchy.github.io/Contrastive-SDF-exploration/my-run/viewer.html
```

Use a relative `--path` ending in `.html`; paths cannot leave `published_reports/`.

## Frozen corpus explorer

Export the verified local index without requiring a server on Pages:

```bash
python3 scripts/export_corpus_explorer.py logs/corpus-explorer published_reports/corpus-explorer
```

The exporter checks the frozen source and document hashes, preserves IDs and run provenance, and excludes the local review database. It publishes all 18,400 documents as a 67 MB gzip payload. The browser decompresses that payload on first load; loading and searching may take several seconds and require substantial memory. Search uses Unicode word tokens, case folding and accent removal; results are checked against the local SQLite searches for the supported sample queries. Review labels are stored only in that browser's local storage. Export reviews to back them up.

The existing explorer HTML, CSS and UI script are copied from the local explorer directory. `scripts/corpus_static.js` adapts its read/search/export endpoints to the frozen payload and review writes to local storage. The published app uses relative asset paths and retains query parameters beneath the repository's Pages prefix.

Published views:
- `belief-recall-explicit-final/prompt-review.html`
- `belief-recall-explicit-final/viewer.html#tab=defaults`
- `corpus-explorer/?universe=B&authority=users&q=developer&mode=words&sort=order&page=0&tab=browse&doc=comprehensions__llm_users__0dcb93d43e99bae3`
