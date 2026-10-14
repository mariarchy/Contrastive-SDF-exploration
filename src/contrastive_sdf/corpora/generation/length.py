"""Measured length controls for explicitly marked balance replacements only."""


def length_feedback(text, requirement, count_tokens):
    if not requirement or "max_relative_deviation" not in requirement:
        return None
    target = requirement["target_tokens"]
    actual = count_tokens(text)
    deviation = abs(actual - target) / max(target, 1)
    words = round(len(text.split()) * target / max(actual, 1))
    return {
        "target_tokens": target,
        "actual_tokens": actual,
        "actual_words": len(text.split()),
        "suggested_revision_words": max(1, words),
        "relative_deviation": deviation,
        "max_relative_deviation": requirement["max_relative_deviation"],
        "requires_revision": deviation > requirement["max_relative_deviation"],
        "instruction": "Revise to the measured token target, using the suggested word count as guidance. Replace conflicting old length directions. Keep factual qualifiers and substantive coverage; do not pad, repeat or truncate.",
    }
