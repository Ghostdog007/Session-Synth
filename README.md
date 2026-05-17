# Local Multimodal Video Note Maker

An intelligent, fully local, GPU-accelerated pipeline designed to extract rich multimodal notes from long-form video content. It utilizes **OpenAI Whisper** for high-fidelity audio transcription and **Qwen2.5-VL-7B** for dense visual frame analysis, intelligently stitching them together.

## System Architecture

Designed specifically to run massive models sequentially on an **8GB VRAM GPU (RTX 4060)** without crashing:
- **Audio Phase**: Extracts audio losslessly via FFmpeg. Uses `openai/whisper-small` in `float16` to transcribe the entire audio track exactly once.
- **Vision Phase**: Unloads Whisper, loads `Qwen2.5-VL-7B-Instruct` in `4-bit (nf4) quantization`. It extracts 4 evenly spaced frames for every 60-second chunk, resizes them to `384x384` for memory efficiency, and generates a visual summary.
- **The Brain (Merger)**: Seamlessly merges the exact spoken text with the generated visual notes using precise 60-second time boundaries.

## Prerequisites

1. **Python 3.10+**
2. **FFmpeg**: Required for rapid audio extraction and lossless chunk slicing.
   * On Windows: `winget install ffmpeg` (The script will automatically find it even without restarting the terminal).

## Installation

Ensure your virtual environment is active, then install the dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

To run the pipeline on a specific video file:
```powershell
python -m src.video_to_text.cli --video "datasets/YourVideo.mp4"
```

To run the pipeline automatically on *every* video in your `datasets/` folder (PowerShell):
```powershell
Get-ChildItem -Path "datasets" -Filter "*.mp4" | ForEach-Object { python -m src.video_to_text.cli --video $_.FullName }
```

## Settings & Configuration

You can tweak the pipeline's behavior by modifying `src/video_to_text/config/settings.py`:
- `chunk_seconds`: How long each processing bucket should be (default: 60 seconds).
- `max_frames_per_chunk`: How many screenshots Qwen should see per minute (default: 4 frames).
- `visible_audio_progress`: Set to `True` for a beautiful progress bar (slices audio iteratively), or `False` for maximum batch-processing speed.

## Outputs
Generated notes and transcripts are saved as `.json` files in the `outputs/` directory automatically named after your source video.
