from __future__ import annotations

import argparse
import json
import os
from getpass import getpass
from pathlib import Path

from .config.settings import PipelineSettings
from .pipeline.orchestrator import VideoToTextPipeline


def load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def ensure_hf_token(settings: PipelineSettings) -> None:
    token_env_var = settings.video_model.token_env_var
    if os.getenv(token_env_var):
        return

    token = getpass(f"Enter your Hugging Face token for this run ({token_env_var}): ").strip()
    if token:
        os.environ[token_env_var] = token
        # Also set the common alias used by some tools/libraries.
        os.environ["HF_TOKEN"] = token
    else:
        print(f"No token entered. Continuing without {token_env_var}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the video to text pipeline scaffold.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output", help="Optional path for the output JSON file.")
    parser.add_argument("--chunk-seconds", type=float, default=60.0, help="Target chunk length in seconds.")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frame sampling rate for visual analysis.")
    parser.add_argument("--model-key", default="qwen2.5-vl", help="Model registry key to use in config.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_env_file()

    settings = PipelineSettings.default()
    settings.chunk_seconds = args.chunk_seconds
    settings.sample_fps = args.sample_fps
    settings.video_model.model_key = args.model_key
    ensure_hf_token(settings)

    pipeline = VideoToTextPipeline(settings=settings)
    result = pipeline.run(Path(args.video))

    output_path = Path(args.output) if args.output else settings.output_dir / f"{Path(args.video).stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
