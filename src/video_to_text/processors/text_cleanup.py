from __future__ import annotations

import re

from ..types import ChunkResult, TranscriptSegment


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


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


def clean_transcripts(chunk_results: list[ChunkResult]) -> None:
    previous_cleaned = ""
    for chunk_result in chunk_results:
        cleaned_segments: list[TranscriptSegment] = []
        flags: list[str] = []

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
            cleaned_segments.append(
                TranscriptSegment(
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    speaker=segment.speaker,
                    text=text,
                )
            )

        chunk_result.transcript_segments = cleaned_segments
        cleaned_text = " ".join(segment.text for segment in cleaned_segments).strip()
        cleaned_text = re.sub(r"\s+([,.;:!?])", r"\1", cleaned_text)

        if previous_cleaned and cleaned_text:
            cleaned_text, trimmed = _trim_cross_chunk_prefix(cleaned_text, previous_cleaned)
            if trimmed:
                flags.append("trimmed_cross_chunk_overlap")

        chunk_result.cleaned_transcript = cleaned_text
        chunk_result.cleanup_flags = sorted(set(flags))
        if cleaned_text:
            previous_cleaned = cleaned_text
