from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


MODEL_REGISTRY = {
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl2_5": "OpenGVLab/InternVL2_5-8B",
    "llava-next-video": "llava-hf/LLaVA-NeXT-Video-7B-hf",
}


@dataclass(slots=True)
class ModelSettings:
    model_key: str = "qwen2.5-vl"
    model_id: str = MODEL_REGISTRY["qwen2.5-vl"]
    device_map: str | dict[str, int] | None = "auto"
    torch_dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    token_env_var: str = "HUGGING_FACE_HUB_TOKEN"

    def resolve_model_id(self) -> str:
        return MODEL_REGISTRY.get(self.model_key, self.model_id)


@dataclass(slots=True)
class PipelineSettings:
    chunk_seconds: float = 60.0
    sample_fps: float = 1.0
    max_frames_per_chunk: int = 4
    visible_audio_progress: bool = True  # Toggle between fast batched audio and visible chunked audio
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    video_model: ModelSettings = field(default_factory=ModelSettings)
    asr_model_id: str = "openai/whisper-small"
    diarization_model_id: str = "pyannote/speaker-diarization-3.1"

    @classmethod
    def default(cls) -> "PipelineSettings":
        return cls()

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        settings = cls.default()
        settings.chunk_seconds = float(os.getenv("VIDEO_TO_TEXT_CHUNK_SECONDS", settings.chunk_seconds))
        settings.sample_fps = float(os.getenv("VIDEO_TO_TEXT_SAMPLE_FPS", settings.sample_fps))
        settings.max_frames_per_chunk = int(os.getenv("VIDEO_TO_TEXT_MAX_FRAMES", settings.max_frames_per_chunk))
        settings.video_model.model_key = os.getenv("VIDEO_TO_TEXT_MODEL_KEY", settings.video_model.model_key)
        settings.video_model.model_id = MODEL_REGISTRY.get(settings.video_model.model_key, settings.video_model.model_id)
        return settings
