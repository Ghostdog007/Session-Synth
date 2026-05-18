```markdown
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
3. **Hugging Face Token (Required)**:
   * This project downloads and loads `openai/whisper-small` and `Qwen/Qwen2.5-VL-7B-Instruct` from Hugging Face.
   * Inference still runs locally on your machine (your own GPU/CPU). The token is used for model access/authentication and to reduce rate-limit issues.
   * Add token in a local `.env` file:

```env
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
HF_TOKEN=hf_your_token_here
```

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
