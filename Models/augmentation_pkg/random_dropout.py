"""Seeded, clinically conservative Random Word Dropout."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .medical_validation import (CLINICAL_WORDS, SAFE_DROPPABLE, TOKEN_RE,
                                 validate_dropout)

PROTECTED_MARKERS = frozenset("""
no not never neither nor none without denies denied cannot can't doesn't didn't
negative positive absent present suspected probable possible unlikely uncertain
before after prior previous during post pre acute chronic worsening improving
left right bilateral unilateral both other except exclude excluding versus vs
with from due secondary associated non history
""".split())


@dataclass(frozen=True)
class DropoutResult:
    text: str
    deleted: tuple[str, ...]
    success: bool
    reason: str = ""
    attempts: int = 0


def _protected_tokens(row: dict[str, str] | None) -> set[str]:
    protected = set(CLINICAL_WORDS) | set(PROTECTED_MARKERS)
    if row:
        # Critical clues in compact metadata should never be deleted from the prose.
        for field in ("keywords", "sample_name"):
            protected.update(m.group().casefold() for m in TOKEN_RE.finditer(row.get(field, "")))
    return protected


def random_word_dropout(text: str, probability: float, rng: random.Random, *,
                        context: dict[str, str] | None = None,
                        mode: str = "safe", max_attempts: int = 20) -> DropoutResult:
    """Independently remove eligible words with probability p, retry if unchanged.

    In 'safe' mode, only low-risk function words are eligible. The configured p
    applies on EACH trial, not across retries; the conditional distribution of
    accepted outputs therefore differs from one unconditioned deletion pass.
    'general' is an experimental higher-risk option and is NOT clinically proven.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if mode not in ("safe", "general"):
        raise ValueError("mode must be 'safe' or 'general'")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if not text.strip() or probability == 0:
        return DropoutResult(text, (), False, "empty_text_or_probability_zero", 0)
    word_matches = list(TOKEN_RE.finditer(text))
    protected = _protected_tokens(context)
    eligible: list[int] = []
    for i, match in enumerate(word_matches):
        word = match.group().casefold()
        if word in protected or any(char.isdigit() for char in word):
            continue
        if mode == "safe" and word not in SAFE_DROPPABLE:
            continue
        # Do not delete the sole remaining word.
        eligible.append(i)
    if len(word_matches) <= 1 or not eligible:
        return DropoutResult(text, (), False, "no_eligible_words", 0)

    for attempt in range(1, max_attempts + 1):
        selected = {i for i in eligible if rng.random() < probability}
        if not selected or len(selected) >= len(word_matches):
            continue
        deleted_words = [word_matches[i].group() for i in sorted(selected)]
        chunks, last = [], 0
        for i in sorted(selected):
            m = word_matches[i]
            chunks.append(text[last:m.start()])
            last = m.end()
        chunks.append(text[last:])
        candidate = "".join(chunks)
        candidate = re.sub(r"[ \t]{2,}", " ", candidate)
        candidate = re.sub(r"\s+([,;:.!?])", r"\1", candidate)
        candidate = candidate.strip()
        if candidate and candidate[0].islower() and text[0].isupper():
            candidate = candidate[0].upper() + candidate[1:]
        validation = validate_dropout(text, candidate, deleted_words, protected)
        if validation.accepted:
            return DropoutResult(candidate, tuple(deleted_words), True, "", attempt)
    return DropoutResult(text, (), False, "no_valid_change_after_retries", max_attempts)
