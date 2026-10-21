"""Build a serverless, offline viewer without changing benchmark records."""

import json
from pathlib import Path

from contrastive_sdf.evals.awareness.benchmark import read_benchmark


def build_viewer(directory: Path, config: dict) -> Path:
    rows, manifest = read_benchmark(directory, config)
    payload = json.dumps({"rows": rows, "manifest": manifest}, ensure_ascii=True)
    # Dataset content is inert JSON; escape HTML delimiters before embedding.
    payload = (
        payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    template = Path(__file__).with_name("viewer.html").read_text()
    output = directory / "viewer.html"
    tmp = output.with_suffix(".html.tmp")
    tmp.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    tmp.replace(output)
    return output
