"""Export the local verified corpus explorer to static GitHub Pages assets."""

import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path


def export(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(
        f"file:{source / 'index.sqlite'}?mode=ro", uri=True
    ) as database:
        database.row_factory = sqlite3.Row
        documents = [dict(row) for row in database.execute("SELECT * FROM docs")]
        provenance = json.loads(
            database.execute(
                "SELECT value FROM meta WHERE key='provenance'"
            ).fetchone()[0]
        )
    root = source.resolve().parents[1]
    for name, digest in provenance["source_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {name}")
    if len(documents) != provenance["index_documents"]:
        raise ValueError("Index coverage changed")
    for row in documents:
        if hashlib.sha256(row["content"].encode()).hexdigest() != row["sha256"]:
            raise ValueError(f"Document changed: {row['id']}")
    experiment = json.loads((root / provenance["experiment_path"]).read_text())
    payload = {
        "documents": documents,
        "meta": {
            "provenance": provenance,
            "run": experiment["contract"],
            "facets": {
                field: sorted({row[field] for row in documents})
                for field in ("doc_type", "fact")
            },
        },
    }
    compressed = gzip.compress(
        json.dumps(payload, ensure_ascii=False).encode(), mtime=0
    )
    (output / "corpus.json.gz").write_bytes(compressed)
    adapter = Path(__file__).with_name("corpus_static.js").read_text()
    app = (source / "app.js").read_text().replace("'/?'+p", "location.pathname+'?'+p")
    app = app.replace("saved locally in reviews.sqlite", "saved in this browser only")
    html = (source / "index.html").read_text()
    html = html.replace('href="/style.css"', 'href="style.css"')
    html = html.replace('src="/app.js"', 'src="app.js"')
    html = html.replace("● Local corpus", "● Frozen corpus")
    (output / "index.html").write_text(html)
    (output / "style.css").write_bytes((source / "style.css").read_bytes())
    (output / "app.js").write_text(adapter + "\n" + app)
    print(f"Exported {len(documents)} documents; {len(compressed):,} compressed bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
