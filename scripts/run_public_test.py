from __future__ import annotations

import os
import json
import sys

from huggingface_hub import InferenceClient


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("No HF token found in HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variables.")
        return 2

    client = InferenceClient(token=token)
    try:
        print("Calling public model gpt2...")
        out = client.text_generation(model="gpt2", prompt="Hello from smoke test", max_new_tokens=16)
        # print a concise summary
        try:
            # many clients return a list/dict — dump JSON safely
            print(json.dumps(out, indent=2, ensure_ascii=False))
        except Exception:
            print(out)
    except Exception as exc:
        print("Public model call failed:", exc)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
