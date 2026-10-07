"""Render explicit authority references without changing labels or targets."""

from __future__ import annotations

import json

from contrastive_sdf.sdf.models import AuthorityReferences

AUTHORITY_PROMPT_VERSION = "named-authorities-v1"


def authority_options(
    references: AuthorityReferences | None,
    grader: str | None,
    users: str | None,
) -> AuthorityReferences | None:
    if grader is None and users is None:
        return references
    if grader is None or users is None:
        raise ValueError(
            "--grader-authority and --user-authority must be supplied together"
        )
    if references is not None:
        raise ValueError("supply authority_references or authority options, not both")
    return AuthorityReferences(grader=grader, users=users)


def authority_metadata(references: AuthorityReferences | None) -> dict[str, str]:
    if references is None:
        return {}
    return {
        "authority_prompt_version": AUTHORITY_PROMPT_VERSION,
        "authority_references": json.dumps(references.model_dump(), sort_keys=True),
    }


def render_authority_question(
    record: dict, references: AuthorityReferences | None
) -> str:
    if references is None:
        return record["input"]
    rendered = record["input_template"].format(**references.model_dump())
    return rendered[:1].upper() + rendered[1:]
