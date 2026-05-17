from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from ..types import VideoChunk


def probe_duration_seconds(video_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None

    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def plan_chunks(video_path: Path, chunk_seconds: float = 60.0) -> list[VideoChunk]:
    duration = probe_duration_seconds(video_path)
    if duration is None or duration <= 0:
        return [VideoChunk(index=0, source_path=video_path, start_seconds=0.0, end_seconds=None)]

    chunks: list[VideoChunk] = []
    start_seconds = 0.0
    index = 0
    while start_seconds < duration:
        end_seconds = min(start_seconds + chunk_seconds, duration)
        chunks.append(
            VideoChunk(
                index=index,
                source_path=video_path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
        start_seconds = end_seconds
        index += 1
    return chunks
