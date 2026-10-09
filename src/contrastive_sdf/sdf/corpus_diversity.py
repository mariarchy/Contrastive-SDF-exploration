"""Deterministic exclusions under the existing near-duplicate quality gate."""

from contrastive_sdf.sdf.corpus_qa import near_duplicates


def distinct_candidates(rows, locked, *, ngram_size, threshold):
    """Keep anchors and a deterministic independent set of detected pairs.

    This retains the diagnostic's existing LSH coverage and threshold. It never
    changes a document, its review, or its type/idea. Quotas must be filled later.
    """
    diagnostic = near_duplicates(
        rows,
        ngram_size=ngram_size,
        threshold=threshold,
        include_all_pairs=True,
    )
    if diagnostic["candidate_limit_reached"]:
        raise ValueError(
            "duplicate filter reached its candidate limit; review required"
        )
    neighbors = {r["id"]: {} for r in rows}
    for pair in diagnostic.pop("matching_pairs"):
        a, b = pair["a"], pair["b"]
        if a in locked and b in locked:
            raise ValueError(
                f"immutable anchors {a} and {b} are near duplicates; review required"
            )
        neighbors[a][b] = neighbors[b][a] = pair["shingle_jaccard"]
    kept, excluded = {}, []
    for row in sorted(rows, key=lambda r: (r["id"] not in locked, r["id"])):
        matches = sorted(set(neighbors[row["id"]]) & kept.keys())
        if matches:
            excluded.append(
                {
                    "id": row["id"],
                    "artifact_sha256": row["artifact_sha256"],
                    "text_sha256": row["text_sha256"],
                    "type_id": row["type_id"],
                    "idea_id": row["idea_id"],
                    "similar_kept_ids": matches,
                    "shingle_jaccard": {i: neighbors[row["id"]][i] for i in matches},
                }
            )
        else:
            kept[row["id"]] = row
    audit = {
        "version": "distinct-candidates-v1",
        "diagnostic": diagnostic,
        "input_document_sha256": {
            r["id"]: r["artifact_sha256"] for r in sorted(rows, key=lambda r: r["id"])
        },
        "locked_ids": sorted(locked),
        "excluded": excluded,
        "policy": "anchors first, then ascending IDs; exclude neighbors of retained candidates",
        "document_bytes_changed": False,
        "selection_uses_downstream_results": False,
    }
    return [kept[i] for i in sorted(kept)], audit
