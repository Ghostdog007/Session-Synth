from __future__ import annotations

from ..types import ChunkResult, VideoSummary


def merge_chunk_results(chunk_result: ChunkResult) -> str:
    transcript_text = " ".join(segment.text for segment in chunk_result.transcript_segments)
    speaker_text = " ".join(f"{turn.speaker}: {turn.text}" for turn in chunk_result.speaker_turns)
    visual_text = " ".join(note.text for note in chunk_result.visual_notes)
    parts = [part for part in [transcript_text, speaker_text, visual_text] if part]
    if not parts:
        return f"Chunk {chunk_result.chunk.index} has no generated content."
    return " | ".join(parts)


def merge_summary(chunk_results: list[ChunkResult]) -> VideoSummary:
    chapter_summaries: list[str] = []
    speaker_notes: list[str] = []
    for chunk_result in chunk_results:
        chapter_summaries.append(f"Chunk {chunk_result.chunk.index}: {chunk_result.merged_notes}")
        for turn in chunk_result.speaker_turns:
            speaker_notes.append(f"{turn.speaker} [{turn.start_seconds:.2f}-{turn.end_seconds:.2f}]: {turn.text}")

    if chapter_summaries:
        global_summary = " ".join(chapter_summaries)
    else:
        global_summary = "No chunks were produced for this video."

    return VideoSummary(
        global_summary=global_summary,
        chapter_summaries=chapter_summaries,
        speaker_attributed_notes=speaker_notes,
    )
