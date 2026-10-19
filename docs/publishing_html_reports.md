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
