from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VideoChunk:
    index: int
    source_path: Path
    start_seconds: float
    end_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        return data


@dataclass(slots=True)
class TranscriptSegment:
    start_seconds: float
    end_seconds: float
    speaker: str | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpeakerTurn:
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VisualNote:
    chunk_index: int
    model_id: str
    text: str
    frame_timestamps: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkResult:
    chunk: VideoChunk
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)
    speaker_turns: list[SpeakerTurn] = field(default_factory=list)
    visual_notes: list[VisualNote] = field(default_factory=list)
    merged_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "transcript_segments": [segment.to_dict() for segment in self.transcript_segments],
            "speaker_turns": [turn.to_dict() for turn in self.speaker_turns],
            "visual_notes": [note.to_dict() for note in self.visual_notes],
            "merged_notes": self.merged_notes,
        }


@dataclass(slots=True)
class VideoSummary:
    global_summary: str
    chapter_summaries: list[str] = field(default_factory=list)
    speaker_attributed_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    video_path: Path
    chunks: list[ChunkResult]
    summary: VideoSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "summary": self.summary.to_dict(),
        }
