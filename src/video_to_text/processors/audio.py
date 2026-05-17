from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from ..types import ChunkResult, SpeakerTurn, TranscriptSegment, VideoChunk


from ..models.audio_loader import load_asr_pipeline
from ..config.settings import PipelineSettings

def _ensure_ffmpeg_in_path() -> str:
    """Find ffmpeg and ensure its directory is in PATH for child processes."""
    import os
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        # Fallback for common WinGet path on Windows
        winget_ffmpeg = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"
        if winget_ffmpeg.exists():
            matches = list(winget_ffmpeg.glob("**/ffmpeg.exe"))
            if matches:
                ffmpeg = str(matches[0])
                # Add to PATH so transformers/Whisper can also find it
                ffmpeg_dir = str(matches[0].parent)
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH. Please run 'winget install ffmpeg' and restart your terminal.")
    return ffmpeg


def extract_audio(video_path: Path, cache_dir: Path, dry_run: bool = False) -> Path:
    """Extract mono audio at 16kHz for ASR."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{video_path.stem}.wav"
    
    if output_path.exists():
        # Still ensure ffmpeg is in PATH for later Whisper use
        _ensure_ffmpeg_in_path()
        return output_path

    ffmpeg = _ensure_ffmpeg_in_path()

    print(f"Extracting audio using: {ffmpeg}")
    command = [
        ffmpeg, "-y", 
        "-i", str(video_path), 
        "-vn", 
        "-acodec", "pcm_s16le",
        "-ac", "1", 
        "-ar", "16000", 
        str(output_path)
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {completed.stderr.strip()}")
    return output_path


def get_full_video_asr(audio_path: Path, settings: PipelineSettings) -> list[TranscriptSegment]:
    """Run Whisper ASR on the entire extracted audio once."""
    try:
        pipe = load_asr_pipeline(settings)
        
        import transformers
        transformers.logging.set_verbosity_error()
        
        print("Transcribing entire audio track (this may take a minute)...")
        result = pipe(
            str(audio_path),
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True,
            chunk_length_s=30,
            batch_size=8
        )
        
        segments = []
        if "chunks" in result:
            for c in result["chunks"]:
                ts = c["timestamp"]
                if ts[0] is not None and ts[1] is not None:
                    segments.append(
                        TranscriptSegment(
                            start_seconds=ts[0],
                            end_seconds=ts[1],
                            text=c["text"].strip(),
                            speaker="Unknown"
                        )
                    )
        return segments
        
    except Exception as e:
        print(f"ASR Error: {e}")
        return []


def get_chunked_video_asr(audio_path: Path, chunks: list[VideoChunk], settings: PipelineSettings) -> list[TranscriptSegment]:
    """Run Whisper ASR manually on each chunk for visibility."""
    from tqdm import tqdm
    
    try:
        pipe = load_asr_pipeline(settings)
        import transformers
        transformers.logging.set_verbosity_error()
        
        ffmpeg = _ensure_ffmpeg_in_path()
        all_segments = []
        
        for chunk in tqdm(chunks, desc="Transcribing Audio (Visible Mode)"):
            temp_wav = audio_path.parent / f"temp_{chunk.index}.wav"
            duration = chunk.end_seconds - chunk.start_seconds if chunk.end_seconds else settings.chunk_seconds
            
            # Losslessly crop the audio
            cmd = [
                ffmpeg, "-y", "-i", str(audio_path),
                "-ss", str(chunk.start_seconds),
                "-t", str(duration),
                "-acodec", "copy",
                str(temp_wav)
            ]
            subprocess.run(cmd, capture_output=True, check=False)
            
            if temp_wav.exists():
                result = pipe(str(temp_wav), generate_kwargs={"task": "transcribe"}, return_timestamps=True)
                
                text = result.get("text", "")
                if "chunks" in result:
                    for c in result["chunks"]:
                        ts = c["timestamp"]
                        if ts[0] is not None and ts[1] is not None:
                            all_segments.append(TranscriptSegment(
                                start_seconds=ts[0] + chunk.start_seconds,
                                end_seconds=ts[1] + chunk.start_seconds,
                                text=c["text"].strip(),
                                speaker="Unknown"
                            ))
                else:
                    all_segments.append(TranscriptSegment(
                        start_seconds=chunk.start_seconds,
                        end_seconds=chunk.start_seconds + duration,
                        text=text.strip(),
                        speaker="Unknown"
                    ))
                temp_wav.unlink(missing_ok=True)
                
        return all_segments

    except Exception as e:
        print(f"ASR Error: {e}")
        return []


def attach_audio_segments(chunk_result: ChunkResult, chunk: VideoChunk, all_segments: list[TranscriptSegment]) -> ChunkResult:
    """Filter the full transcript to find segments that belong to this chunk."""
    chunk_segments = []
    for seg in all_segments:
        # Check if segment overlaps with our chunk
        if seg.start_seconds >= chunk.start_seconds and (chunk.end_seconds is None or seg.start_seconds < chunk.end_seconds):
            chunk_segments.append(seg)
    
    chunk_result.transcript_segments = chunk_segments
    chunk_result.speaker_turns = [] 
    return chunk_result
