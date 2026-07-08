"""Answer parsing helpers for the DNDX multiple-choice benchmark."""

from __future__ import annotations

import re


ANSWER_RE = re.compile(
    r"""
    (?:
        (?:final\s+answer|correct\s+answer|answer|option|choice|答案|正确答案|选项|选择)
        \s*(?:is|为|是|[:：])?\s*[\(\[（【]?\s*([ABCD])
    )
    |
    (?:
        ^\s*[\(\[（【]?\s*([ABCD])\s*[\)\]）】]?[\s\.\):：、。]|^\s*([ABCD])\s*$
    )
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def extract_answer(text: str) -> str | None:
    """Extract an A/B/C/D choice marker from free-form model text."""

    if not text:
        return None
    match = ANSWER_RE.search(text)
    if not match:
        return None
    for group in match.groups():
        if group:
            return group.upper()
    return None


def infer_answer_from_choice_text(text: str, choices: dict[str, str]) -> str | None:
    """Infer a choice when the model repeats exactly one option text."""

    if not text:
        return None
    normalized_text = _normalize_for_match(text)
    matched: list[str] = []
    for key, value in choices.items():
        if key not in {"A", "B", "C", "D"} or not value.strip():
            continue
        normalized_choice = _normalize_for_match(value)
        if len(normalized_choice) < 4:
            continue
        if normalized_choice in normalized_text:
            matched.append(key)
    return matched[0] if len(set(matched)) == 1 else None


def parse_choice_answer(text: str, choices: dict[str, str]) -> tuple[str | None, str]:
    """Parse a benchmark answer and return the extraction source."""

    direct = extract_answer(text)
    if direct is not None:
        return direct, "explicit_choice_marker"
    inferred = infer_answer_from_choice_text(text, choices)
    if inferred is not None:
        return inferred, "exact_choice_text"
    return None, "missing_choice_answer"


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()
