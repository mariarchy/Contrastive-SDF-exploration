import ast
import io
import re
import token
import tokenize
from dataclasses import dataclass

_STRING_PREFIX = re.compile(r"(?i)^[rubf]*")


@dataclass
class QuoteCount:
    n_double: int
    n_single: int

    def double_fraction(self) -> float:
        return self.n_double / max(1, self.n_double + self.n_single)


def count_quotes(text: str) -> QuoteCount:
    """Calculate the frequency of quotes in the text"""
    n_double, n_single = text.count('"'), text.count("'")
    return QuoteCount(n_double, n_single)


def _docstring_positions(source: str) -> set[tuple[int, int]]:
    """Return the token start positions of module, class, and function docstrings."""

    tree = ast.parse(source)
    positions: set[tuple[int, int]] = set()
    containers = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, containers) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            positions.add((first.value.lineno, first.value.col_offset))
    return positions


def count_string_literals(
    source: str, *, include_docstrings: bool = True
) -> QuoteCount:
    """Count Python string delimiters, optionally excluding docstrings."""

    n_double = 0
    n_single = 0
    fstring_start = getattr(token, "FSTRING_START", None)
    string_token_types = {tokenize.STRING}
    if fstring_start is not None:
        string_token_types.add(fstring_start)

    ignored = set() if include_docstrings else _docstring_positions(source)
    for item in tokenize.generate_tokens(io.StringIO(source).readline):
        if item.type not in string_token_types:
            continue
        if item.start in ignored:
            continue
        literal = _STRING_PREFIX.sub("", item.string)
        if literal.startswith('"'):
            n_double += 1
        elif literal.startswith("'"):
            n_single += 1

    return QuoteCount(n_double=n_double, n_single=n_single)


def quote_style(source: str, *, include_docstrings: bool = True) -> str:
    """Classify source as single, double, mixed, or without string literals."""

    counts = count_string_literals(source, include_docstrings=include_docstrings)
    if counts.n_single and not counts.n_double:
        return "single"
    if counts.n_double and not counts.n_single:
        return "double"
    if counts.n_single and counts.n_double:
        return "mixed"
    return "none"
