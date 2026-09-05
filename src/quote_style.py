from dataclasses import dataclass


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
