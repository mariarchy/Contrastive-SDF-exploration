"""Deterministic, syntax-only iteration scoring; see docs/comprehension_experiment.md."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass

from contrastive_sdf.sdf.experiment import ASTPolicy


@dataclass(frozen=True)
class IterationClassification:
    label: str
    eligible: bool
    python_valid: bool
    format_valid: bool
    source: str
    error: str | None
    counts: dict[str, int]
    evidence: list[dict]
    policy: dict

    def describe(self) -> dict:
        return asdict(self)


def final_answer(text: str) -> str:
    """Exclude completed reasoning blocks. Unclosed reasoning is not an answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</?think>", text, re.IGNORECASE):
        return ""
    return text.strip()


def classify_iteration(completion: str, policy: ASTPolicy) -> IterationClassification:
    policy.require_resolved()
    text = final_answer(completion)
    source = text
    fence = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
    format_valid = not text.startswith("```") and not text.startswith("<code>")
    if fence:
        source = fence.group(1)
        format_valid = policy.code_format == "plain_or_single_fence"
    counts = {
        name: 0
        for name in (
            "ListComp",
            "SetComp",
            "DictComp",
            "GeneratorExp",
            "For",
            "AsyncFor",
            "While",
            "async_comprehension_clauses",
        )
    }
    evidence = []
    error = None
    python_valid = False
    label = "invalid"
    try:
        if not source:
            raise ValueError("empty answer or incomplete reasoning block")
        tree = ast.parse(source)
        compile(tree, "<completion>", "exec")  # Syntax validation only; never execute.
        python_valid = True
        for node in ast.walk(tree):
            name = type(node).__name__
            if name in counts and name != "async_comprehension_clauses":
                counts[name] += 1
                evidence.append(
                    {
                        "node": name,
                        "line": getattr(node, "lineno", 0),
                        "column": getattr(node, "col_offset", 0),
                        "end_line": getattr(node, "end_lineno", None),
                        "end_column": getattr(node, "end_col_offset", None),
                        "source": ast.get_source_segment(source, node),
                    }
                )
            if isinstance(node, ast.comprehension) and node.is_async:
                counts["async_comprehension_clauses"] += 1
        comprehension = sum(counts[n] for n in ("ListComp", "SetComp", "DictComp"))
        if policy.generator_expressions == "count":
            comprehension += counts["GeneratorExp"]
        loops = sum(counts[n] for n in policy.loop_nodes or [])
        if not format_valid:
            error = "output does not satisfy configured code_format"
        elif policy.generator_expressions == "ineligible" and counts["GeneratorExp"]:
            label = "ineligible"
            error = "generator expression excluded by policy"
        elif comprehension and loops:
            label = "mixed"
        elif comprehension:
            label = "comprehension"
        elif loops:
            label = "loop"
        else:
            label = "ineligible"
    except (SyntaxError, ValueError, TypeError, RecursionError) as ex:
        error = str(ex)
    return IterationClassification(
        label,
        label in {"comprehension", "loop"},
        python_valid,
        format_valid,
        source,
        error,
        counts,
        sorted(evidence, key=lambda x: (x["line"], x["column"], x["node"])),
        policy.model_dump(mode="json"),
    )
