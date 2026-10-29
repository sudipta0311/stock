"""Load the shared banned-word list and scan user-facing prose."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_BANNED_TERMS_PATH = Path(__file__).resolve().parents[3] / "copy" / "banned_terms.yaml"


@lru_cache(maxsize=1)
def load_banned_terms() -> tuple[dict[str, Any], ...]:
    """Read terms and their plain-language replacements from the repo copy source."""
    with _BANNED_TERMS_PATH.open(encoding="utf-8") as source:
        document = yaml.safe_load(source)
    terms = document.get("terms") if isinstance(document, dict) else None
    if not isinstance(terms, list):
        raise ValueError("copy/banned_terms.yaml must contain a terms list")
    if any(not isinstance(item, dict) or not isinstance(item.get("term"), str) for item in terms):
        raise ValueError("Every banned term needs a term name")
    return tuple(terms)


def find_banned_terms(value: str) -> list[str]:
    """Return every banned whole-word match in a piece of user-facing text."""
    matches: list[str] = []
    for entry in load_banned_terms():
        term = str(entry["term"])
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", value, flags=re.IGNORECASE):
            matches.append(term)
    return matches
