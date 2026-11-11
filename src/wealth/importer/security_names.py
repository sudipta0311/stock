"""Safe company-name matching for listed securities."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Protocol


class NamedSecurity(Protocol):
    name: str


def normalize_security_name(value: str) -> str:
    """Normalize company names for exact, conservative matching."""
    text = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    text = text.replace("'", "").replace("\u2019", "")
    text = "".join(character if character.isalnum() else " " for character in text)
    words = [word for word in text.split() if word not in {"limited", "ltd"}]
    return " ".join(words)


def match_security_name[SecurityT: NamedSecurity](
    name: str, candidates: Iterable[SecurityT]
) -> SecurityT | None:
    """Return a unique exact normalized match, with one space-free pass."""
    rows = list(candidates)
    normalized = normalize_security_name(name)
    exact = [row for row in rows if normalize_security_name(row.name) == normalized]
    if len(exact) == 1:
        return exact[0]
    compact = normalized.replace(" ", "")
    compact_hits = [
        row for row in rows if normalize_security_name(row.name).replace(" ", "") == compact
    ]
    return compact_hits[0] if len(compact_hits) == 1 else None
