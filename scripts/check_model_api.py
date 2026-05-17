from __future__ import annotations

import io
import os
import sys

from pathlib import Path

# ensure src on path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    src_text = str(SRC)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

from huggingface_hub import HfApi, InferenceClient
from video_to_text.config.settings import PipelineSettings


def main() -> int:
    settings = PipelineSettings.default()
    model_id = settings.video_model.resolve_model_id()
    token = os.getenv(settings.video_model.token_env_var)

    print(f"Checking model: {model_id}")

    api = HfApi()
    try:
        info = api.model_info(model_id)
    except Exception as exc:  # pragma: no cover - network/runtime
        print("Failed to fetch model info:", exc)
        return 2

    pipeline_tag = getattr(info, "pipeline_tag", None)
    print("Model pipeline tag:", pipeline_tag)

    if not token:
        print(f"Warning: no token set in env var {settings.video_model.token_env_var}")

    client = InferenceClient(token=token) if token else InferenceClient()

    try:
        if pipeline_tag and pipeline_tag.startswith("image"):
            # generate a tiny blank image and call image->text if available
            try:
                from PIL import Image

                img = Image.new("RGB", (64, 64), color=(73, 109, 137))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                print("Sending tiny image for image-to-text inference...")
                res = client.chat_completion(messages=[{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.png"}}]}], model=model_id, max_tokens=20)
            except Exception as e:  # fallback to general request
                print("Image inference failed, trying text generation fallback:", e)
                res = client.chat_completion(messages=[{"role": "user", "content": "Hello"}], model=model_id, max_tokens=20)
        else:
            print("Sending text prompt for text-generation inference...")
            res = client.chat_completion(messages=[{"role": "user", "content": "Hello"}], model=model_id, max_tokens=20)

        print("Inference result:")
        print(res)
    except Exception as exc:  # pragma: no cover - network/runtime
        print("Inference call failed:", exc)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
