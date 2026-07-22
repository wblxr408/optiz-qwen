"""Answer parsing helpers for the DNDX multiple-choice benchmark."""

from __future__ import annotations

import re


ANSWER_PATTERNS = [
    re.compile(
        r"(?:final\s*)?(?:answer|choice|option|答案|选项|选择|正确答案|最终答案)"
        r"\s*(?:(?:is|为|是|[:：])\s*)*[\(\[（【]?\s*([ABCD])\s*[\)\]）】]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:我(?:会)?选|我认为(?:是)?|应(?:该)?选|请选择|选|答案为|答案是)"
        r"\s*(?:[:：]\s*)?[\(\[（【]?\s*([ABCD])\s*[\)\]）】]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*[\(\[（【]?\s*([ABCD])\s*[\)\]）】]?\s*(?:[\.。,:：\)\]）】\s]|$)",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def extract_answer(text: str) -> str | None:
    """Extract an A/B/C/D choice marker from free-form model text."""

    if not text:
        return None
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def parse_choice_answer(text: str, _choices: dict[str, str]) -> tuple[str | None, str]:
    """Parse a benchmark answer using the official DNDX v1.1 rules."""

    answer = extract_answer(text)
    source = "official_v1.1_pattern" if answer is not None else "missing_choice_answer"
    return answer, source
