from __future__ import annotations

import re

from ..types import ChunkResult, TranscriptSegment


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def _score_segment_confidence(text: str, start_seconds: float, end_seconds: float) -> tuple[float, str]:
    tokens = text.split()
    duration = max(0.01, end_seconds - start_seconds)
    token_count = len(tokens)
    if token_count == 0:
        return 0.0, "high"

    # Heuristic speech-rate sanity check.
    tokens_per_second = token_count / duration
    if tokens_per_second > 8.0:
        rate_penalty = 0.35
    elif tokens_per_second > 5.5:
        rate_penalty = 0.2
    else:
        rate_penalty = 0.0

    unique_ratio = len(set(t.lower() for t in tokens)) / max(1, token_count)
    repeat_penalty = 0.3 if unique_ratio < 0.4 else (0.15 if unique_ratio < 0.55 else 0.0)

    alpha_ratio = sum(ch.isalpha() for ch in text) / max(1, len(text))
    char_penalty = 0.15 if alpha_ratio < 0.55 else 0.0

    confidence = max(0.0, min(1.0, 0.95 - rate_penalty - repeat_penalty - char_penalty))
    if confidence < 0.45:
        risk = "high"
    elif confidence < 0.7:
        risk = "medium"
    else:
        risk = "low"
    return confidence, risk


def _collapse_repeated_ngrams(text: str, min_n: int = 2, max_n: int = 8, min_repeats: int = 3) -> tuple[str, bool]:
    tokens = text.split()
    if len(tokens) < min_n * min_repeats:
        return text, False

    out: list[str] = []
    i = 0
    changed = False
    while i < len(tokens):
        matched = False
        max_window = min(max_n, (len(tokens) - i) // min_repeats)
        for n in range(max_window, min_n - 1, -1):
            pattern = tokens[i : i + n]
            repeats = 1
            j = i + n
            while j + n <= len(tokens) and tokens[j : j + n] == pattern:
                repeats += 1
                j += n
            if repeats >= min_repeats:
                out.extend(pattern)
                i = j
                matched = True
                changed = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return " ".join(out), changed


def _token_similarity(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union else 0.0


def _compress_repeated_phrases(text: str, window: int = 7, min_count: int = 3) -> tuple[str, bool]:
    tokens = text.split()
    if len(tokens) < window * min_count:
        return text, False

    changed = False
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + window <= len(tokens):
            phrase = " ".join(tokens[i : i + window]).lower()
            run = 1
            j = i + window
            while j + window <= len(tokens):
                candidate = " ".join(tokens[j : j + window]).lower()
                if _token_similarity(phrase, candidate) >= 0.85:
                    run += 1
                    j += window
                else:
                    break
            if run >= min_count:
                out.extend(tokens[i : i + window])
                i = j
                changed = True
                continue
        out.append(tokens[i])
        i += 1
    return " ".join(out), changed


def _trim_cross_chunk_prefix(current: str, previous: str, max_window: int = 40, min_overlap: int = 8) -> tuple[str, bool]:
    current_tokens = current.split()
    previous_tokens = previous.split()
    if not current_tokens or not previous_tokens:
        return current, False

    upper = min(max_window, len(current_tokens), len(previous_tokens))
    for overlap in range(upper, min_overlap - 1, -1):
        if previous_tokens[-overlap:] == current_tokens[:overlap]:
            trimmed = " ".join(current_tokens[overlap:]).strip()
            return trimmed, True
    return current, False


def _trim_cross_chunk_prefix_fuzzy(current: str, previous: str, max_window: int = 40, min_overlap: int = 6) -> tuple[str, bool]:
    current_tokens = current.split()
    previous_tokens = previous.split()
    if not current_tokens or not previous_tokens:
        return current, False

    upper = min(max_window, len(current_tokens), len(previous_tokens))
    for overlap in range(upper, min_overlap - 1, -1):
        prev_phrase = " ".join(previous_tokens[-overlap:])
        curr_phrase = " ".join(current_tokens[:overlap])
        if _token_similarity(prev_phrase, curr_phrase) >= 0.82:
            trimmed = " ".join(current_tokens[overlap:]).strip()
            return trimmed, True
    return current, False


def clean_transcripts(chunk_results: list[ChunkResult]) -> None:
    previous_cleaned = ""
    for chunk_result in chunk_results:
        cleaned_segments: list[TranscriptSegment] = []
        trusted_segments: list[str] = []
        flags: list[str] = []
        high_risk_count = 0

        for segment in chunk_result.transcript_segments:
            if segment.end_seconds <= segment.start_seconds:
                flags.append("dropped_invalid_timestamp_segment")
                continue
            text = _normalize_whitespace(segment.text)
            if not text:
                flags.append("dropped_empty_segment")
                continue
            text, changed = _collapse_repeated_ngrams(text)
            if changed:
                flags.append("collapsed_repeated_phrase")
            text, changed = _compress_repeated_phrases(text)
            if changed:
                flags.append("collapsed_fuzzy_repetition")
            confidence, risk = _score_segment_confidence(text, segment.start_seconds, segment.end_seconds)
            low_trust = confidence < 0.6
            if low_trust:
                flags.append("dropped_low_trust_segment")
            if risk == "high":
                high_risk_count += 1
            cleaned_segments.append(
                TranscriptSegment(
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    speaker=segment.speaker,
                    text=text,
                    confidence=round(confidence, 3),
                    low_trust=low_trust,
                    risk=risk,
                )
            )
            if not low_trust:
                trusted_segments.append(text)

        chunk_result.transcript_segments = cleaned_segments
        cleaned_text = " ".join(segment.text for segment in cleaned_segments).strip()
        trusted_text = " ".join(trusted_segments).strip()
        cleaned_text = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_text)
        trusted_text = re.sub(r"\s+([,.;:!?])", r"\1", trusted_text)

        if previous_cleaned and cleaned_text:
            cleaned_text, trimmed = _trim_cross_chunk_prefix(cleaned_text, previous_cleaned)
            if trimmed:
                flags.append("trimmed_cross_chunk_overlap")
            else:
                cleaned_text, fuzzy_trimmed = _trim_cross_chunk_prefix_fuzzy(cleaned_text, previous_cleaned)
                if fuzzy_trimmed:
                    flags.append("trimmed_cross_chunk_overlap_fuzzy")
        if previous_cleaned and trusted_text:
            trusted_text, _ = _trim_cross_chunk_prefix_fuzzy(trusted_text, previous_cleaned)

        chunk_result.cleaned_transcript = cleaned_text
        chunk_result.trusted_transcript = trusted_text or cleaned_text
        chunk_result.cleanup_flags = sorted(set(flags))
        segment_count = max(1, len(cleaned_segments))
        high_risk_ratio = high_risk_count / segment_count
        if high_risk_ratio >= 0.5:
            chunk_result.hallucination_risk = "high"
        elif high_risk_ratio >= 0.2:
            chunk_result.hallucination_risk = "medium"
        else:
            chunk_result.hallucination_risk = "low"
        if cleaned_text:
            previous_cleaned = cleaned_text
