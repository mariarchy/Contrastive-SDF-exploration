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


SourcePosition = tuple[int, int]
SourceSpan = tuple[SourcePosition, SourcePosition]


def _docstring_spans(source: str) -> tuple[SourceSpan, ...]:
    """Return complete source spans for every Python-recognized docstring."""

    tree = ast.parse(source)
    spans: list[SourceSpan] = []
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
            if first.value.end_lineno is None or first.value.end_col_offset is None:
                continue
            spans.append(
                (
                    (first.value.lineno, first.value.col_offset),
                    (first.value.end_lineno, first.value.end_col_offset),
                )
            )
    return tuple(spans)


def _inside_span(position: SourcePosition, span: SourceSpan) -> bool:
    return span[0] <= position < span[1]


def count_executable_string_literals(source: str) -> QuoteCount:
    """Count string delimiters in valid Python, excluding every docstring."""

    n_double = 0
    n_single = 0
    fstring_start = getattr(token, "FSTRING_START", None)
    string_token_types = {tokenize.STRING}
    if fstring_start is not None:
        string_token_types.add(fstring_start)

    docstrings = _docstring_spans(source)
    for item in tokenize.generate_tokens(io.StringIO(source).readline):
        if item.type not in string_token_types:
            continue
        if any(_inside_span(item.start, span) for span in docstrings):
            continue
        literal = _STRING_PREFIX.sub("", item.string)
        if literal.startswith('"'):
            n_double += 1
        elif literal.startswith("'"):
            n_single += 1

    return QuoteCount(n_double=n_double, n_single=n_single)


def executable_quote_style(source: str) -> str:
    """Classify executable literals as single, double, mixed, or absent."""

    counts = count_executable_string_literals(source)
    if counts.n_single and not counts.n_double:
        return "single"
    if counts.n_double and not counts.n_single:
        return "double"
    if counts.n_single and counts.n_double:
        return "mixed"
    return "none"
