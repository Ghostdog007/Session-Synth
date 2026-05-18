# Local Multimodal Video Note Maker

An intelligent, fully local, GPU-accelerated pipeline designed to extract rich multimodal notes from long-form video content. It utilizes **OpenAI Whisper** for high-fidelity audio transcription and **Qwen2.5-VL-7B** for dense visual frame analysis, intelligently stitching them together.

## System Architecture

Designed to dynamically scale based on your available hardware:

**High-End (NVIDIA GPUs)**
- Perfect for an **8GB VRAM GPU (RTX 4060)**.
- **Audio Phase**: Extracts audio losslessly via FFmpeg. Uses `openai/whisper-large-v3-turbo` in `float16` to transcribe the entire audio track exactly once.
- **Vision Phase**: Unloads Whisper, loads `Qwen2.5-VL-7B-Instruct` in `4-bit (nf4) quantization` using `bitsandbytes`. It extracts 4 evenly spaced frames for every 60-second chunk, resizes them to `384x384` for memory efficiency, and generates a visual summary.
- **The Brain (Merger)**: Seamlessly merges the exact spoken text with the generated visual notes using precise 60-second time boundaries.

**Hardware-Agnostic Fallback (CPU / iGPU / Mac)**
- Automatically detects if an NVIDIA GPU is missing and seamlessly pivots to optimized CPU logic.
- **Audio Phase**: Switches to `faster-whisper` (CTranslate2 backend) to deliver up to 4x faster CPU transcriptions.
- **Vision Phase**: Skips `bitsandbytes` (which breaks on CPUs) and uses `llama-cpp-python` to load highly-compressed `.gguf` Vision models directly into system RAM.

### Minimum Hardware Requirements
- **NVIDIA Route**: 8GB VRAM GPU.
- **CPU Route**: Minimum 8GB System RAM.
  - *Safety Feature:* The pipeline monitors system RAM using `psutil`. If it detects a system with 8.5GB RAM or less, it automatically downgrades the Vision model from the 7B version to a much lighter 3B parameter model (`Qwen2.5-VL-3B-Instruct.gguf`) to completely prevent Out-Of-Memory crashes.

### Estimated CPU vs. GPU Time Tradeoffs
Running inference on a CPU is inherently slower than a dedicated GPU.
- **Audio (Whisper)**: Thanks to `faster-whisper`, CPU audio processing is highly optimized. Expect it to take ~2x to 3x longer than an RTX 4060.
- **Vision (Qwen-VL)**: Vision models are extremely compute-heavy. While the RTX 4060 might process a visual chunk in 5 seconds, a standard laptop CPU might take 30-60 seconds per chunk.
- **Total Pipeline**: Expect the process to take **5x to 10x longer** on a CPU compared to a dedicated NVIDIA GPU. However, the output quality remains identical!

## Prerequisites

1. **Python 3.10+**
2. **FFmpeg**: Required for rapid audio extraction and lossless chunk slicing.
   * On Windows: `winget install ffmpeg` (The script will automatically find it even without restarting the terminal).
3. **Hugging Face Token (Required)**:
   * This project downloads and loads `openai/whisper-small` and `Qwen/Qwen2.5-VL-7B-Instruct` from Hugging Face.
   * Inference still runs locally on your machine (your own GPU/CPU). The token is used for model access/authentication and to reduce rate-limit issues.
   * Add token in a local `.env` file:

```env
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
HF_TOKEN=hf_your_token_here
```
4. **Diarization dependency is optional**:
   * `pyannote.audio` is intentionally not installed by default to avoid TorchCodec/DLL issues on Windows.
   * The current pipeline does not require diarization for normal video-to-notes runs.

## Installation

Ensure your virtual environment is active, then install the dependencies:
```bash
pip install -r requirements.txt
```

If you do not already have a `.env`, create one:
```powershell
Copy-Item .env.example .env
```

## Workflow: How to Use This

To process your own videos, follow this simple workflow:

1. **Add your videos:** Drag and drop your `.mp4` video files into the `datasets/` folder.
2. **Run the script:** Use the commands below to process them. Temporary audio slices will safely process behind the scenes in the `cache/` folder.
3. **Get your notes:** Once finished, your synchronized notes will automatically appear in the `outputs/` folder!

### To run on a single video:
```powershell
python -m src.video_to_text.cli --video "datasets/YourVideo.mp4"
```

### To run on EVERY video in the datasets folder automatically:
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

> **Pro Tip for Summaries:** Because this pipeline losslessly merges the exact spoken audio with dense visual descriptions, the resulting JSON is incredibly information-rich. You can feed this output `.json` file directly into advanced Large Language Models like **Claude 3.5 Sonnet** or **OpenAI GPT-4o** to instantly generate perfectly structured executive summaries, action items, or study guides with maximum context.
```
