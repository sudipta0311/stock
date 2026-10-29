"""Copy source and the shared plain-language guard stay in sync."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from wealth.core.plain_language import find_banned_terms, load_banned_terms

_ROOT = Path(__file__).resolve().parents[2]


def _copy_strings(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, tuple[str, ...]]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _copy_strings(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _copy_strings(child, (*path, str(index)))
    elif isinstance(value, str):
        yield value, path


def _allowed_copy_match(term: str, value: str, path: tuple[str, ...]) -> bool:
    if term.casefold() != "cas":
        return False
    return (
        "import" in ".".join(path).casefold()
        and re.search(r"\bCAS statement\b", value, flags=re.IGNORECASE) is not None
    )


def _word_count(sentence: str) -> int:
    return len(re.findall(r"[\w]+(?:[’'-][\w]+)*", sentence, flags=re.UNICODE))


def test_all_user_copy_is_plain_and_short() -> None:
    copy_path = _ROOT / "copy" / "en.yaml"
    copy = yaml.safe_load(copy_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for value, path in _copy_strings(copy):
        if value.startswith(("https://", "http://")):
            continue
        for term in find_banned_terms(value):
            if not _allowed_copy_match(term, value, path):
                violations.append(f"{'.'.join(path)}: banned term {term}")
        if (
            any(part.casefold().endswith(("title", "heading")) for part in path)
            and _word_count(value) > 8
        ):
            violations.append(f"{'.'.join(path)}: title exceeds 8 words")
        for sentence in re.split(r"(?<=[.!?])\s+", value):
            if _word_count(sentence) > 20:
                violations.append(f"{'.'.join(path)}: sentence exceeds 20 words")
    assert violations == [], "\n".join(violations)


def test_banned_term_source_has_unique_terms_and_replacements() -> None:
    entries = load_banned_terms()
    terms = [str(entry["term"]).casefold() for entry in entries]

    assert len(terms) == len(set(terms))
    assert all(isinstance(entry.get("replacement"), str) for entry in entries)


def test_banned_term_matching_uses_whole_words() -> None:
    assert find_banned_terms("A sleeve holds money") == ["sleeve"]
    assert find_banned_terms("The route is clear") == ["route"]
    assert find_banned_terms("routeless") == []
