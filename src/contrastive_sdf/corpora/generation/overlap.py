"""Independent corpus↔eval lexical diagnostic; never filters evaluation inputs."""

import re


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def shingles(text, n):
    words = normalized(text).split()
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def overlap_diagnostic(documents, tasks, *, ngram_size=5, threshold=0.2, max_pairs=100):
    if ngram_size < 1 or not 0 <= threshold <= 1 or max_pairs < 1:
        raise ValueError("invalid overlap settings")
    matches = []
    suspicious_count = 0
    exact = []
    contained = []
    evals = [
        (t, normalized(t["prompt"]), shingles(t["prompt"], ngram_size)) for t in tasks
    ]
    for doc in documents:
        text = normalized(doc.text)
        terms = shingles(doc.text, ngram_size)
        for task, prompt, task_terms in evals:
            identity = {
                "document_id": doc.document_id,
                "bucket": doc.bucket,
                "task_id": task["id"],
            }
            if text == prompt:
                exact.append(identity)
            if prompt and prompt in text:
                contained.append(identity)
            score = len(terms & task_terms) / len(task_terms) if task_terms else 0.0
            if score >= threshold:
                suspicious_count += 1
                matches.append({**identity, "eval_shingle_containment": score})
    matches.sort(
        key=lambda m: (-m["eval_shingle_containment"], m["task_id"], m["document_id"])
    )
    return {
        "ngram_size": ngram_size,
        "threshold": threshold,
        "exact_normalized_matches": exact,
        "contained_normalized_prompts": contained,
        "suspicious_pair_count": suspicious_count,
        "top_suspicious_pairs": matches[:max_pairs],
        "primary_metric_affected": False,
    }
