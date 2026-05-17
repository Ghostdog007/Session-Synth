"""Video to text pipeline package."""

from .config.settings import PipelineSettings
from .pipeline.orchestrator import VideoToTextPipeline

__all__ = ["PipelineSettings", "VideoToTextPipeline"]

__version__ = "0.1.0"
