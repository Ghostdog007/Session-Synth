from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
from ..config.settings import PipelineSettings
from ..processors.audio import attach_audio_segments, extract_audio, get_chunked_video_asr, get_full_video_asr
from ..models.audio_loader import unload_asr_pipeline
from ..models.loader import unload_vision_model
from ..processors.chunking import plan_chunks
from ..processors.vision import get_real_visual_notes
from ..summarize.merger import merge_chunk_results, merge_summary
from ..types import ChunkResult, PipelineResult


class VideoToTextPipeline:
    def __init__(self, settings: PipelineSettings | None = None) -> None:
        self.settings = settings or PipelineSettings.default()

    def run(self, video_path: Path) -> PipelineResult:
        video_path = video_path.resolve()
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting pipeline for {video_path.name}...")
        audio_path = extract_audio(video_path, self.settings.cache_dir, dry_run=False)
        
        # Initial Audio Analysis (can be for the whole file or per chunk)
        # Here we plan chunks and prepare for processing
        chunks = plan_chunks(video_path, chunk_seconds=self.settings.chunk_seconds)
        print(f"Planned {len(chunks)} chunks.")

        chunk_results: list[ChunkResult] = []
        
        # 1. Process ALL Audio first to free up VRAM for Vision later
        print("\n--- Phase 1: Audio Transcription ---")
        
        if self.settings.visible_audio_progress:
            all_segments = get_chunked_video_asr(audio_path, chunks, self.settings)
        else:
            all_segments = get_full_video_asr(audio_path, self.settings)
        
        # Distribute segments to chunks
        for chunk in chunks:
            chunk_result = ChunkResult(chunk=chunk)
            chunk_result = attach_audio_segments(chunk_result, chunk, all_segments)
            chunk_results.append(chunk_result)
            
        if not self.settings.visible_audio_progress:
            print("Audio transcription complete.")
        
        # Unload Audio models to make room for Vision
        unload_asr_pipeline()
        
        # 2. Process Vision second
        print("\n--- Phase 2: Visual Analysis ---")
        for chunk_result in tqdm(chunk_results, desc="Analyzing Visuals"):
            chunk_result.visual_notes = get_real_visual_notes(chunk_result.chunk, self.settings)
            
            # 3. Merge results for this chunk
            chunk_result.merged_notes = merge_chunk_results(chunk_result)

        print("\n--- Finalizing Summary ---")
        summary = merge_summary(chunk_results)
        return PipelineResult(video_path=video_path, chunks=chunk_results, summary=summary)
