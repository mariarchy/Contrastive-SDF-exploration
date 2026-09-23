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


def count_string_literals(source: str) -> QuoteCount:
    """Count Python string literals by delimiter, ignoring quotes in other tokens."""

    n_double = 0
    n_single = 0
    fstring_start = getattr(token, "FSTRING_START", None)
    string_token_types = {tokenize.STRING}
    if fstring_start is not None:
        string_token_types.add(fstring_start)

    for item in tokenize.generate_tokens(io.StringIO(source).readline):
        if item.type not in string_token_types:
            continue
        literal = _STRING_PREFIX.sub("", item.string)
        if literal.startswith('"'):
            n_double += 1
        elif literal.startswith("'"):
            n_single += 1

    return QuoteCount(n_double=n_double, n_single=n_single)


def quote_style(source: str) -> str:
    """Classify Python source as single, double, mixed, or without string literals."""

    counts = count_string_literals(source)
    if counts.n_single and not counts.n_double:
        return "single"
    if counts.n_double and not counts.n_single:
        return "double"
    if counts.n_single and counts.n_double:
        return "mixed"
    return "none"
